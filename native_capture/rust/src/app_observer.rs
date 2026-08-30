//! Event-driven app-change detection via NSWorkspace notifications.
//!
//! Mirrors `CaptureManager.wireAppChanges` in the Swift native_mac helper.
//! Observes `NSWorkspaceDidActivateApplicationNotification` and sends app
//! change info (app_name, bundle_id, pid) to the capture thread via a
//! channel, which emits the `app_change` event and triggers a screenshot.
//!
//! This is what catches a terminal switch that a 1fps poll might miss —
//! the notification fires immediately on app activation.

#[cfg(target_os = "macos")]
mod platform {
    use std::ptr::NonNull;
    use std::sync::mpsc;

    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, NSObjectProtocol, ProtocolObject};
    use objc2_foundation::NSNotification;
    use objc2_app_kit::{
        NSRunningApplication, NSWorkspace, NSWorkspaceApplicationKey,
        NSWorkspaceDidActivateApplicationNotification,
    };

    /// App change info sent from the notification callback to the capture thread.
    #[derive(Debug, Clone)]
    pub struct AppChangeEvent {
        pub app_name: Option<String>,
        pub bundle_id: Option<String>,
        #[allow(dead_code)]
        pub pid: i32,
    }

    /// Observes NSWorkspace app-activation notifications. The observer token
    /// and block must stay alive for the duration of the session; dropping
    /// this struct removes the observer.
    pub struct AppObserver {
        _block: Option<block2::RcBlock<dyn Fn(NonNull<NSNotification>) + 'static>>,
        token: Option<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
        center: Option<Retained<objc2_foundation::NSNotificationCenter>>,
    }

    // SAFETY: The observer is registered on the main thread (before the run
    // loop starts). The block captures an mpsc::Sender which is Send.
    unsafe impl Send for AppObserver {}

    impl AppObserver {
        /// Registers for NSWorkspaceDidActivateApplicationNotification.
        /// Returns the observer and a receiver for AppChangeEvent messages.
        /// Returns None if the notification center can't be accessed.
        pub fn start() -> Option<(Self, mpsc::Receiver<AppChangeEvent>)> {
            let (tx, rx) = mpsc::channel::<AppChangeEvent>();

            let workspace = NSWorkspace::sharedWorkspace();
            let center = workspace.notificationCenter();

            // The block receives a NonNull<NSNotification> (owned pointer)
            // per the objc2 DynBlock signature. We dereference it inside.
            let block =
                block2::RcBlock::new(move |notification: NonNull<NSNotification>| {
                    let notification = unsafe { notification.as_ref() };
                    let app = extract_running_application(notification);
                    let event = AppChangeEvent {
                        app_name: app
                            .as_ref()
                            .and_then(|a| a.localizedName())
                            .map(|s| s.to_string()),
                        bundle_id: app
                            .as_ref()
                            .and_then(|a| a.bundleIdentifier())
                            .map(|s| s.to_string()),
                        pid: app.as_ref().map(|a| a.processIdentifier()).unwrap_or(0),
                    };
                    let _ = tx.send(event);
                });

            // Register the observer. &*block derefs RcBlock to &DynBlock.
            // NSWorkspaceDidActivateApplicationNotification is an extern static
            // that's safe to read (it's a constant CFString reference).
            let token = unsafe {
                center.addObserverForName_object_queue_usingBlock(
                    Some(NSWorkspaceDidActivateApplicationNotification),
                    None,
                    None,
                    &*block,
                )
            };

            Some((
                Self {
                    _block: Some(block),
                    token: Some(token),
                    center: Some(center),
                },
                rx,
            ))
        }

        /// Removes the observer from the notification center.
        pub fn stop(&mut self) {
            if let (Some(center), Some(token)) = (&self.center, &self.token) {
                // removeObserver expects &AnyObject. ProtocolObject implements
                // AsRef<AnyObject>, so we use the explicit AsRef call to convert.
                let any: &AnyObject = AsRef::<AnyObject>::as_ref(&**token);
                unsafe {
                    center.removeObserver(any);
                }
            }
            self.token = None;
            self.center = None;
            self._block = None;
        }
    }

    impl Drop for AppObserver {
        fn drop(&mut self) {
            self.stop();
        }
    }

    /// Extracts the NSRunningApplication from a NSWorkspace notification's
    /// userInfo dictionary, using the NSWorkspaceApplicationKey.
    fn extract_running_application(
        notification: &NSNotification,
    ) -> Option<Retained<NSRunningApplication>> {
        let user_info = notification.userInfo()?;
        let key = unsafe { NSWorkspaceApplicationKey };
        let app_obj = user_info.objectForKey(key)?;
        app_obj.downcast::<NSRunningApplication>().ok()
    }
}

#[cfg(target_os = "macos")]
pub use platform::{AppChangeEvent, AppObserver};

// -- Windows: foreground-window polling (no NSWorkspace on Windows) --

#[cfg(target_os = "windows")]
mod platform {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::Duration;

    use crate::foreground_window;

    /// App change info sent from the polling thread to the capture thread.
    /// Same shape as the macOS `AppChangeEvent` so `handle_app_change` in
    /// main.rs works unchanged.
    #[derive(Debug, Clone)]
    pub struct AppChangeEvent {
        pub app_name: Option<String>,
        pub bundle_id: Option<String>,
        #[allow(dead_code)]
        pub pid: i32,
    }

    /// Polls `GetForegroundWindow` on a dedicated thread (~300 ms interval)
    /// and sends an `AppChangeEvent` when the foreground process changes.
    /// This is the Windows analog of the macOS NSWorkspace notification
    /// observer — it catches app switches that the 1 fps screen sampler
    /// might miss between ticks.
    ///
    /// Window title changes are detected by the main capture loop's
    /// `accessibility::window_title` sampling at ~1 fps (matching the C#
    /// helper's `Tick()` behavior), not by this observer.
    pub struct AppObserver {
        running: Arc<AtomicBool>,
        thread: Option<thread::JoinHandle<()>>,
    }

    impl AppObserver {
        /// Starts the polling thread. Returns the observer and a receiver
        /// for `AppChangeEvent` messages. Returns `None` if the thread
        /// can't be spawned (no crash).
        pub fn start() -> Option<(Self, mpsc::Receiver<AppChangeEvent>)> {
            let (tx, rx) = mpsc::channel::<AppChangeEvent>();
            let running = Arc::new(AtomicBool::new(true));
            let running_clone = running.clone();

            let handle = thread::Builder::new()
                .name("mentor-capture.app-observer".to_string())
                .spawn(move || {
                    let mut last_app: Option<String> = None;
                    while running_clone.load(Ordering::SeqCst) {
                        if let Some(fg) = foreground_window::try_get() {
                            let app_changed = Some(&fg.app_name) != last_app.as_ref();
                            if app_changed {
                                last_app = Some(fg.app_name.clone());
                                let _ = tx.send(AppChangeEvent {
                                    app_name: Some(fg.app_name.clone()),
                                    // On Windows the process name serves as
                                    // the bundle_id (no bundle identifiers).
                                    bundle_id: Some(fg.app_name.clone()),
                                    pid: fg.pid as i32,
                                });
                            }
                        }
                        // Poll at ~300 ms — fast enough to catch a terminal
                        // switch between 1 fps ticks, cheap enough to not
                        // burden the CPU.
                        thread::sleep(Duration::from_millis(300));
                    }
                })
                .ok()?;

            Some((
                Self {
                    running,
                    thread: Some(handle),
                },
                rx,
            ))
        }

        /// Signals the polling thread to stop and joins it.
        pub fn stop(&mut self) {
            self.running.store(false, Ordering::SeqCst);
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }

    impl Drop for AppObserver {
        fn drop(&mut self) {
            self.stop();
        }
    }
}

#[cfg(target_os = "windows")]
pub use platform::{AppChangeEvent, AppObserver};

// -- Stubs for other platforms (neither macOS nor Windows) --

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod stubs {
    use std::sync::mpsc;
    #[derive(Debug, Clone)]
    pub struct AppChangeEvent {
        pub app_name: Option<String>,
        pub bundle_id: Option<String>,
        pub pid: i32,
    }
    pub struct AppObserver;
    impl AppObserver {
        pub fn start() -> Option<(Self, mpsc::Receiver<AppChangeEvent>)> {
            let (tx, rx) = mpsc::channel();
            let _ = tx;
            None
        }
        pub fn stop(&mut self) {}
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub use stubs::{AppChangeEvent, AppObserver};
