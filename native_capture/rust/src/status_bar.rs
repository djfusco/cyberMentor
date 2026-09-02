//! macOS menu-bar recording indicator.
//!
//! Creates an NSStatusItem showing "● Recording" while a capture session is
//! active, mirroring the behaviour of the Swift helper's CaptureManager.
//! The item is removed automatically when the guard is dropped.

#[cfg(target_os = "macos")]
mod platform {
    use objc2::rc::Retained;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSStatusBar, NSStatusItem,
        NSVariableStatusItemLength,
    };
    use objc2_foundation::NSString;

    /// Holds the status item for the duration of a session.
    /// Dropping this removes the item from the menu bar.
    pub struct StatusBarGuard {
        item: Retained<NSStatusItem>,
        bar: Retained<NSStatusBar>,
    }

    // NSStatusItem / NSStatusBar are Objective-C objects. We only ever touch
    // them from the main thread (run_start runs on main), so this is safe.
    unsafe impl Send for StatusBarGuard {}

    impl StatusBarGuard {
        /// Creates the "● Recording" menu-bar item.
        pub fn new() -> Self {
            // SAFETY: called from the main thread (run_start, before the loop).
            let mtm = unsafe { MainThreadMarker::new_unchecked() };

            // Ensure NSApplication is initialised and won't show a Dock icon.
            let app = NSApplication::sharedApplication(mtm);
            let _ = app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);

            let bar = NSStatusBar::systemStatusBar();
            let item = bar.statusItemWithLength(NSVariableStatusItemLength);

            // Use the deprecated title setter — it avoids needing the full
            // NSStatusBarButton feature chain and matches what the Swift
            // helper does (button.title = "● Recording").
            #[allow(deprecated)]
            item.setTitle(Some(&NSString::from_str("● Recording")));

            Self { item, bar }
        }
    }

    impl Drop for StatusBarGuard {
        fn drop(&mut self) {
            self.bar.removeStatusItem(&self.item);
        }
    }
}

#[cfg(target_os = "macos")]
pub use platform::StatusBarGuard;

// -- Stub for non-macOS platforms -------------------------------------------

#[cfg(not(target_os = "macos"))]
pub struct StatusBarGuard;

#[cfg(not(target_os = "macos"))]
impl StatusBarGuard {
    pub fn new() -> Self {
        Self
    }
}
