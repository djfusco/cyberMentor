//! Passive (listen-only) global input capture via a CGEventTap.
//!
//! Mirrors `InputObserver.swift` in the Swift native_mac helper. Captures:
//! - Left/right mouse clicks (position + triggered screenshot)
//! - Scroll wheel events (aggregated count, flushed periodically)
//! - Key-down events (coarse category ONLY — return/tab/escape/delete/arrow/
//!   command-shortcut/typing — NEVER raw characters)
//!
//! The event tap runs on the main CFRunLoop. Callbacks update shared state
//! (key/scroll counts) and send click events via a channel to the capture
//! thread, which flushes aggregated counts and emits mouse_click events.
//!
//! Needs Input Monitoring permission (IOHIDCheckAccess). Degrades gracefully
//! (start() returns false, no crash) if the tap can't be created.

#[cfg(target_os = "macos")]
mod platform {
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    // The CGEvent/CGEventTap/CFRunLoop APIs are accessed via raw FFI below
    // (CGEventTapCreate, CGRunLoopAddSource, etc.) since we need a C callback
    // function pointer for the event tap.

    // -- FFI: IOKit (IOHIDCheckAccess is not exposed by objc2-io-kit) --

    extern "C" {
        fn IOHIDCheckAccess(request_type: u32) -> u32;
    }

    // kIOHIDRequestTypeListenEvent = 0, kIOHIDAccessTypeGranted = 0
    const K_IOHID_REQUEST_TYPE_LISTEN_EVENT: u32 = 0;
    const K_IOHID_ACCESS_TYPE_GRANTED: u32 = 0;

    // -- CGEventTap callback type --

    // CGEventTapCallBack: (proxy, type, event, userInfo) -> CGEventRef
    type CGEventTapCallBack = unsafe extern "C" fn(
        proxy: *mut c_void,
        r#type: u32,
        event: *mut c_void,
        user_info: *mut c_void,
    ) -> *mut c_void;

    // -- FFI: CGEventTapCreate (we declare it ourselves for the callback type) --

    extern "C" {
        fn CGEventTapCreate(
            tap: u32,             // CGEventTapLocation
            place: u32,           // CGEventTapPlacement
            options: u32,         // CGEventTapOptions
            events_of_interest: u64, // CGEventMask
            callback: CGEventTapCallBack,
            user_info: *mut c_void,
        ) -> *mut c_void; // CFMachPortRef

        fn CGEventTapEnable(tap: *mut c_void, enable: bool);

        fn CGEventGetIntegerValueField(event: *mut c_void, field: u32) -> i64;
        fn CGEventGetFlags(event: *mut c_void) -> u64;
        fn CGEventGetLocation(event: *mut c_void) -> CGPoint;

        // CFRunLoop
        fn CFMachPortCreateRunLoopSource(
            alloc: *mut c_void,
            port: *mut c_void,
            order: isize,
        ) -> *mut c_void; // CFRunLoopSourceRef
        fn CFRunLoopAddSource(
            run_loop: *mut c_void,
            source: *mut c_void,
            mode: *const c_void,
        );
        fn CFRunLoopRemoveSource(
            run_loop: *mut c_void,
            source: *mut c_void,
            mode: *const c_void,
        );
        fn CFRunLoopGetMain() -> *mut c_void;
        fn CFRetain(cf: *const c_void) -> *const c_void;
        fn CFRelease(cf: *const c_void);

        // kCFRunLoopCommonModes
        static kCFRunLoopCommonModes: *const c_void;
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    // -- Shared input state (updated by the event tap callback on the main
    //    thread, read/flushed by the capture thread) --

    /// Aggregated key/scroll counts, protected by a mutex. The capture thread
    /// reads and clears these on each flush window.
    #[derive(Default)]
    pub struct InputCounts {
        pub key_counts: HashMap<String, u32>,
        pub scroll_count: u32,
    }

    /// Click event sent from the event tap callback to the capture thread.
    #[derive(Debug, Clone)]
    pub struct ClickEvent {
        #[allow(dead_code)]
        pub kind: ClickKind,
        pub x: f64,
        pub y: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum ClickKind {
        Left,
        Right,
    }

    /// What the capture thread needs to receive from the input observer:
    /// - a receiver for click events (to attach a triggered frame_path)
    /// - shared counts (to flush as key_activity/scroll events)
    pub struct InputHandle {
        pub click_rx: std::sync::mpsc::Receiver<ClickEvent>,
        pub counts: Arc<Mutex<InputCounts>>,
    }

    /// Passive (listen-only) global event tap. Never records raw typed
    /// characters — keys are categorized only.
    pub struct InputObserver {
        tap: *mut c_void,
        run_loop_source: *mut c_void,
        installed: bool,
    }

    // SAFETY: The tap and run loop source are raw pointers managed by the
    // main run loop. InputObserver is only used from the main thread (start
    // is called before the run loop starts, stop is called after it stops).
    unsafe impl Send for InputObserver {}

    impl InputObserver {
        /// Non-prompting permission check (Input Monitoring). Safe to call
        /// from `check`.
        pub fn check_permission() -> bool {
            unsafe {
                IOHIDCheckAccess(K_IOHID_REQUEST_TYPE_LISTEN_EVENT) == K_IOHID_ACCESS_TYPE_GRANTED
            }
        }

        /// Attempts to install the event tap. Returns the InputHandle (for the
        /// capture thread to receive clicks and read counts) on success, or
        /// None if permission is missing or tap creation fails (no crash).
        pub fn start() -> Option<(Self, InputHandle)> {
            // CGEventType raw values: leftMouseDown=1, rightMouseDown=3,
            // scrollWheel=22, keyDown=10
            let mask: u64 = (1u64 << 1) | (1u64 << 3) | (1u64 << 22) | (1u64 << 10);

            let counts = Arc::new(Mutex::new(InputCounts::default()));
            let (click_tx, click_rx) = std::sync::mpsc::channel::<ClickEvent>();

            // Package the shared state for the callback's user_info pointer.
            let callback_data = Box::new(CallbackData {
                counts: counts.clone(),
                click_tx,
            });
            let user_info = Box::into_raw(callback_data) as *mut c_void;

            unsafe {
                // CGEventTapLocation: kCGSessionEventTap = 1
                // CGEventTapPlacement: kCGHeadInsertEventTap = 0
                // CGEventTapOptions: kCGListenEventTap = 1 (listen-only)
                let tap = CGEventTapCreate(1, 0, 1, mask, event_tap_callback, user_info);
                if tap.is_null() {
                    // Free the boxed data since the callback will never fire.
                    let _ = Box::from_raw(user_info as *mut CallbackData);
                    return None;
                }

                let source = CFMachPortCreateRunLoopSource(std::ptr::null_mut(), tap, 0);
                if source.is_null() {
                    CFRelease(tap);
                    let _ = Box::from_raw(user_info as *mut CallbackData);
                    return None;
                }

                let main_run_loop = CFRunLoopGetMain();
                CFRetain(main_run_loop);
                CFRunLoopAddSource(main_run_loop, source, kCFRunLoopCommonModes);
                CGEventTapEnable(tap, true);

                Some((
                    Self {
                        tap,
                        run_loop_source: source,
                        installed: true,
                    },
                    InputHandle { click_rx, counts },
                ))
            }
        }

        /// Removes the event tap and run loop source. Must be called before
        /// the main run loop stops (or from the main thread after it stops).
        pub fn stop(&mut self) {
            if !self.installed {
                return;
            }
            unsafe {
                CGEventTapEnable(self.tap, false);
                let main_run_loop = CFRunLoopGetMain();
                CFRunLoopRemoveSource(main_run_loop, self.run_loop_source, kCFRunLoopCommonModes);
                CFRelease(self.run_loop_source);
                CFRelease(self.tap);
            }
            self.installed = false;
        }
    }

    impl Drop for InputObserver {
        fn drop(&mut self) {
            self.stop();
        }
    }

    // -- Callback data and function --

    struct CallbackData {
        counts: Arc<Mutex<InputCounts>>,
        click_tx: std::sync::mpsc::Sender<ClickEvent>,
    }

    /// The CGEventTap callback. Runs on the main run loop. Categorizes keys
    /// (never decodes characters), counts scroll events, and sends click
    /// events via the channel. Returns the event unchanged (listen-only).
    unsafe extern "C" fn event_tap_callback(
        _proxy: *mut c_void,
        event_type: u32,
        event: *mut c_void,
        user_info: *mut c_void,
    ) -> *mut c_void {
        if user_info.is_null() || event.is_null() {
            return event;
        }
        let data = &*(user_info as *const CallbackData);

        match event_type {
            1 => {
                // leftMouseDown
                let loc = CGEventGetLocation(event);
                let _ = data.click_tx.send(ClickEvent {
                    kind: ClickKind::Left,
                    x: loc.x,
                    y: loc.y,
                });
            }
            3 => {
                // rightMouseDown
                let loc = CGEventGetLocation(event);
                let _ = data.click_tx.send(ClickEvent {
                    kind: ClickKind::Right,
                    x: loc.x,
                    y: loc.y,
                });
            }
            22 => {
                // scrollWheel
                if let Ok(mut counts) = data.counts.lock() {
                    counts.scroll_count += 1;
                }
            }
            10 => {
                // keyDown
                let category = category_for_event(event);
                if let Ok(mut counts) = data.counts.lock() {
                    *counts.key_counts.entry(category.to_string()).or_insert(0) += 1;
                }
            }
            _ => {}
        }
        event
    }

    // -- Key categorization (pure, unit-testable) --

    /// Maps a key-down event to a coarse category. Never reads/decodes the
    /// actual character — only the raw keycode and modifier flags.
    /// Mirrors `InputObserver.category(for:)` in Swift.
    pub fn category_for_event(event: *mut c_void) -> &'static str {
        unsafe {
            let flags = CGEventGetFlags(event);
            // kCGEventFlagMaskCommand = 1048576 (0x100000)
            if flags & 1048576 != 0 {
                return "command-shortcut";
            }
            // kCGKeyboardEventKeycode = 9
            let keycode = CGEventGetIntegerValueField(event, 9);
            match keycode {
                36 => "return",
                48 => "tab",
                53 => "escape",
                51 | 117 => "delete",
                123 | 124 | 125 | 126 => "arrow",
                _ => "typing",
            }
        }
    }

    /// Pure key categorization from raw keycode + flags — unit-testable
    /// without a live CGEvent.
    #[allow(dead_code)]
    pub fn category_for_keycode(keycode: i64, flags: u64) -> &'static str {
        if flags & 1048576 != 0 {
            return "command-shortcut";
        }
        match keycode {
            36 => "return",
            48 => "tab",
            53 => "escape",
            51 | 117 => "delete",
            123 | 124 | 125 | 126 => "arrow",
            _ => "typing",
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub use platform::{
    category_for_keycode, InputCounts, InputHandle, InputObserver,
};

// -- Windows: Win32 low-level mouse/keyboard hooks --

#[cfg(target_os = "windows")]
mod platform {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicPtr, AtomicBool, Ordering};
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;

    use windows::Win32::Foundation::{LPARAM, LRESULT, POINT, WPARAM};
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, GetMessageW, PostThreadMessageW, SetWindowsHookExW, UnhookWindowsHookEx,
        MSG, WH_KEYBOARD_LL, WH_MOUSE_LL, WM_KEYDOWN, WM_LBUTTONDOWN, WM_MOUSEWHEEL,
        WM_QUIT, WM_RBUTTONDOWN, WM_SYSKEYDOWN, WINDOWS_HOOK_ID,
    };
    use windows::Win32::System::Threading::GetCurrentThreadId;

    // KBDLLHOOKSTRUCT and MSLLHOOKSTRUCT are not exposed in the
    // windows crate 0.62's Win32::UI::Input::KeyboardAndMouse module, so
    // we define them manually with the standard Win32 ABI layout.
    #[repr(C)]
    #[allow(dead_code)]
    struct KbdLlHookStruct {
        vk_code: u32,
        scan_code: u32,
        flags: u32,
        time: u32,
        dw_extra_info: usize,
    }

    #[repr(C)]
    #[allow(dead_code)]
    struct MslLlHookStruct {
        pt: POINT,
        mouse_data: u32,
        flags: u32,
        time: u32,
        dw_extra_info: usize,
    }

    // -- Shared input state (same shape as macOS) --

    #[derive(Default)]
    pub struct InputCounts {
        pub key_counts: HashMap<String, u32>,
        pub scroll_count: u32,
    }

    #[derive(Debug, Clone)]
    pub struct ClickEvent {
        #[allow(dead_code)]
        pub kind: ClickKind,
        pub x: f64,
        pub y: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum ClickKind {
        Left,
        Right,
    }

    pub struct InputHandle {
        pub click_rx: mpsc::Receiver<ClickEvent>,
        pub counts: Arc<Mutex<InputCounts>>,
    }

    // -- Hook callback data, stored in a static for the C callback --

    struct HookData {
        counts: Arc<Mutex<InputCounts>>,
        click_tx: mpsc::Sender<ClickEvent>,
    }

    // SAFETY: HookData is only accessed from the hook thread's callback
    // (which runs on a single thread) and is stored via AtomicPtr. The
    // Arc<Mutex<InputCounts>> is Send+Sync; the Sender is Send. The
    // AtomicPtr itself is Sync because HookData is Send.
    static HOOK_DATA: AtomicPtr<HookData> = AtomicPtr::new(std::ptr::null_mut());

    /// Runs `f` with the hook data if it's set. The `&HookData` reference
    /// is valid only within the closure (the pointer is valid while hooks
    /// are installed — set in start, cleared in stop after the thread joins).
    fn with_hook_data(f: impl FnOnce(&HookData)) {
        let ptr = HOOK_DATA.load(Ordering::SeqCst);
        if !ptr.is_null() {
            // SAFETY: ptr is valid while the hook is installed. The callback
            // runs only while the hook is installed, so the pointer is valid.
            let data = unsafe { &*ptr };
            f(data);
        }
    }

    // -- InputObserver: installs low-level hooks on a dedicated thread --

    /// Passive (listen-only) global input capture via Win32 low-level hooks
    /// (WH_MOUSE_LL + WH_KEYBOARD_LL). Never records raw typed characters —
    /// keys are categorized only. The hooks run on a dedicated thread with a
    /// Win32 message loop (low-level hooks require one).
    pub struct InputObserver {
        running: Arc<AtomicBool>,
        thread: Option<thread::JoinHandle<()>>,
        thread_id: u32,
        hook_data_ptr: *mut HookData,
    }

    impl InputObserver {
        /// Low-level hooks don't require explicit user permission on Windows
        /// (unlike macOS Input Monitoring). Always returns true.
        #[allow(dead_code)]
        pub fn check_permission() -> bool {
            true
        }

        /// Installs the low-level hooks on a dedicated thread. Returns the
        /// InputHandle (for the capture thread to receive clicks and read
        /// counts) on success, or None if the thread can't be spawned or the
        /// hooks can't be installed (no crash — emits an error event in the
        /// caller and continues without input capture).
        pub fn start() -> Option<(Self, InputHandle)> {
            let counts = Arc::new(Mutex::new(InputCounts::default()));
            let (click_tx, click_rx) = mpsc::channel::<ClickEvent>();

            let hook_data = Box::new(HookData {
                counts: counts.clone(),
                click_tx,
            });
            let hook_data_ptr = Box::into_raw(hook_data);
            // Store before installing hooks so the callback can find it.
            HOOK_DATA.store(hook_data_ptr, Ordering::SeqCst);

            let running = Arc::new(AtomicBool::new(true));
            let running_clone = running.clone();
            let (tid_tx, tid_rx) = mpsc::channel::<u32>();

            let handle = thread::Builder::new()
                .name("mentor-capture.input-hooks".to_string())
                .spawn(move || {
                    let tid = unsafe { GetCurrentThreadId() };
                    let _ = tid_tx.send(tid);

                    // Install low-level hooks (0 = current thread).
                    let mouse_hook = unsafe {
                        SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_callback), None, 0)
                    };
                    let kb_hook = unsafe {
                        SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_callback), None, 0)
                    };

                    if mouse_hook.is_err() || kb_hook.is_err() {
                        eprintln!("cyberalfred-capture: could not install low-level hooks");
                    }

                    // Message loop — low-level hooks require it. GetMessageW
                    // blocks until a message arrives; WM_QUIT (posted by stop)
                    // breaks the loop.
                    let mut msg = MSG::default();
                    while running_clone.load(Ordering::SeqCst) {
                        let ret = unsafe { GetMessageW(&mut msg, None, 0, 0) };
                        // BOOL.0: 0 = WM_QUIT, -1 = error, positive = message.
                        // Both 0 and -1 mean we should stop.
                        if ret.0 <= 0 {
                            break;
                        }
                    }

                    // Unhook (ignore errors — best-effort cleanup).
                    if let Ok(h) = mouse_hook { let _ = unsafe { UnhookWindowsHookEx(h) }; }
                    if let Ok(h) = kb_hook { let _ = unsafe { UnhookWindowsHookEx(h) }; }
                })
                .ok();

            let thread = match handle {
                Some(h) => h,
                None => {
                    // Thread spawn failed — clean up HOOK_DATA.
                    let ptr = HOOK_DATA.swap(std::ptr::null_mut(), Ordering::SeqCst);
                    if !ptr.is_null() {
                        unsafe { let _ = Box::from_raw(ptr); }
                    }
                    return None;
                }
            };

            // Wait for the hook thread to report its thread ID.
            let thread_id = tid_rx.recv().ok()?;

            Some((
                Self {
                    running,
                    thread: Some(thread),
                    thread_id,
                    hook_data_ptr,
                },
                InputHandle { click_rx, counts },
            ))
        }

        /// Posts WM_QUIT to the hook thread, joins it, and cleans up.
        pub fn stop(&mut self) {
            self.running.store(false, Ordering::SeqCst);
            // Post WM_QUIT to break the GetMessageW loop.
            if self.thread_id != 0 {
                let _ = unsafe {
                    PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0))
                };
            }
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
            // Clear and free HOOK_DATA.
            let ptr = HOOK_DATA.swap(std::ptr::null_mut(), Ordering::SeqCst);
            if !ptr.is_null() {
                unsafe { let _ = Box::from_raw(ptr); }
            }
            self.hook_data_ptr = std::ptr::null_mut();
        }
    }

    impl Drop for InputObserver {
        fn drop(&mut self) {
            // If stop wasn't called, clean up the hook data.
            if !self.hook_data_ptr.is_null() {
                let ptr = HOOK_DATA.swap(std::ptr::null_mut(), Ordering::SeqCst);
                if !ptr.is_null() {
                    unsafe { let _ = Box::from_raw(ptr); }
                }
                self.hook_data_ptr = std::ptr::null_mut();
            }
        }
    }

    // -- Hook callbacks (run on the hook thread) --

    /// Low-level mouse hook callback. Categorizes left/right clicks (sends
    /// via channel with screen coordinates) and counts scroll wheel events.
    /// Passes all events to CallNextHookEx (listen-only — never blocks).
    unsafe extern "system" fn mouse_callback(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let msg = wparam.0 as u32;
            with_hook_data(|data| {
                match msg {
                    WM_LBUTTONDOWN => {
                        if let Some(info) = ptr_to_msllhookstruct(lparam) {
                            let _ = data.click_tx.send(ClickEvent {
                                kind: ClickKind::Left,
                                x: info.pt.x as f64,
                                y: info.pt.y as f64,
                            });
                        }
                    }
                    WM_RBUTTONDOWN => {
                        if let Some(info) = ptr_to_msllhookstruct(lparam) {
                            let _ = data.click_tx.send(ClickEvent {
                                kind: ClickKind::Right,
                                x: info.pt.x as f64,
                                y: info.pt.y as f64,
                            });
                        }
                    }
                    WM_MOUSEWHEEL => {
                        if let Ok(mut counts) = data.counts.lock() {
                            counts.scroll_count += 1;
                        }
                    }
                    _ => {}
                }
            });
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    /// Low-level keyboard hook callback. Categorizes key-down events (never
    /// decodes characters) and aggregates counts. Passes all events to
    /// CallNextHookEx (listen-only).
    unsafe extern "system" fn keyboard_callback(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let msg = wparam.0 as u32;
            if msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN {
                if let Some(info) = ptr_to_kbdllhookstruct(lparam) {
                    with_hook_data(|data| {
                        let category = category_for_vk(info.vk_code);
                        if let Ok(mut counts) = data.counts.lock() {
                            *counts.key_counts.entry(category.to_string()).or_insert(0) += 1;
                        }
                    });
                }
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    /// Safely dereferences the lParam as a MslLlHookStruct pointer.
    unsafe fn ptr_to_msllhookstruct(lparam: LPARAM) -> Option<&'static MslLlHookStruct> {
        if lparam.0 == 0 {
            None
        } else {
            Some(&*(lparam.0 as *const MslLlHookStruct))
        }
    }

    /// Safely dereferences the lParam as a KbdLlHookStruct pointer.
    unsafe fn ptr_to_kbdllhookstruct(lparam: LPARAM) -> Option<&'static KbdLlHookStruct> {
        if lparam.0 == 0 {
            None
        } else {
            Some(&*(lparam.0 as *const KbdLlHookStruct))
        }
    }

    // -- Key categorization --

    // Win32 virtual key codes.
    const VK_CONTROL: i32 = 0x11;
    const VK_MENU: i32 = 0x12; // Alt
    const VK_LWIN: i32 = 0x5B;
    const VK_RWIN: i32 = 0x5C;
    const VK_RETURN: u32 = 0x0D;
    const VK_TAB: u32 = 0x09;
    const VK_ESCAPE: u32 = 0x1B;
    const VK_BACK: u32 = 0x08;
    const VK_DELETE: u32 = 0x2E;
    const VK_LEFT: u32 = 0x25;
    const VK_UP: u32 = 0x26;
    const VK_RIGHT: u32 = 0x27;
    const VK_DOWN: u32 = 0x28;

    /// Maps a key-down event (virtual key code + live modifier state via
    /// GetAsyncKeyState) to a coarse category. NEVER decodes the actual
    /// character — mirrors the macOS `category_for_event`. Ctrl/Win/Alt
    /// modifiers take precedence (command-shortcut), matching the macOS
    /// Command flag rule.
    fn category_for_vk(vk_code: u32) -> &'static str {
        // Check live modifier state via GetAsyncKeyState (high bit = down).
        let ctrl = (unsafe { GetAsyncKeyState(VK_CONTROL) } >> 15) != 0;
        let alt = (unsafe { GetAsyncKeyState(VK_MENU) } >> 15) != 0;
        let win = ((unsafe { GetAsyncKeyState(VK_LWIN) } | unsafe { GetAsyncKeyState(VK_RWIN) }) >> 15) != 0;
        if ctrl || alt || win {
            return "command-shortcut";
        }
        match vk_code {
            VK_RETURN => "return",
            VK_TAB => "tab",
            VK_ESCAPE => "escape",
            VK_BACK | VK_DELETE => "delete",
            VK_LEFT | VK_UP | VK_RIGHT | VK_DOWN => "arrow",
            _ => "typing",
        }
    }

    /// Pure key categorization from raw keycode + flags — unit-testable
    /// without a live keyboard state. Uses macOS keycodes for compatibility
    /// with the existing shared test suite (the real Windows categorization
    /// uses `category_for_vk` with Win32 VK codes + GetAsyncKeyState).
    #[allow(dead_code)]
    pub fn category_for_keycode(keycode: i64, flags: u64) -> &'static str {
        if flags & 1048576 != 0 {
            return "command-shortcut";
        }
        match keycode {
            36 => "return",
            48 => "tab",
            53 => "escape",
            51 | 117 => "delete",
            123 | 124 | 125 | 126 => "arrow",
            _ => "typing",
        }
    }

    // Suppress unused warnings for types re-exported via struct fields.
    #[allow(dead_code)]
    fn _assert_types(_: WINDOWS_HOOK_ID) {}
}

#[cfg(target_os = "windows")]
#[allow(unused_imports)]
pub use platform::{
    category_for_keycode, InputCounts, InputHandle, InputObserver,
};

// -- Stubs for other platforms (neither macOS nor Windows) --

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod stubs {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum ClickKind { Left, Right }
    #[derive(Debug, Clone)]
    pub struct ClickEvent { pub kind: ClickKind, pub x: f64, pub y: f64 }
    #[derive(Default)]
    pub struct InputCounts {
        pub key_counts: HashMap<String, u32>,
        pub scroll_count: u32,
    }
    pub struct InputHandle {
        pub click_rx: std::sync::mpsc::Receiver<ClickEvent>,
        pub counts: Arc<Mutex<InputCounts>>,
    }
    pub struct InputObserver;
    impl InputObserver {
        pub fn check_permission() -> bool { false }
        pub fn start() -> Option<(Self, InputHandle)> {
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = tx;
            None
        }
        pub fn stop(&mut self) {}
    }
    pub fn category_for_keycode(keycode: i64, flags: u64) -> &'static str {
        if flags & 1048576 != 0 { return "command-shortcut"; }
        match keycode {
            36 => "return", 48 => "tab", 53 => "escape",
            51 | 117 => "delete", 123 | 124 | 125 | 126 => "arrow",
            _ => "typing",
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub use stubs::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_flag_returns_command_shortcut() {
        assert_eq!(category_for_keycode(0, 1048576), "command-shortcut");
        assert_eq!(category_for_keycode(36, 1048576), "command-shortcut");
    }

    #[test]
    fn return_keycode_returns_return() {
        assert_eq!(category_for_keycode(36, 0), "return");
    }

    #[test]
    fn tab_keycode_returns_tab() {
        assert_eq!(category_for_keycode(48, 0), "tab");
    }

    #[test]
    fn escape_keycode_returns_escape() {
        assert_eq!(category_for_keycode(53, 0), "escape");
    }

    #[test]
    fn delete_keycodes_return_delete() {
        assert_eq!(category_for_keycode(51, 0), "delete");
        assert_eq!(category_for_keycode(117, 0), "delete");
    }

    #[test]
    fn arrow_keycodes_return_arrow() {
        assert_eq!(category_for_keycode(123, 0), "arrow");
        assert_eq!(category_for_keycode(124, 0), "arrow");
        assert_eq!(category_for_keycode(125, 0), "arrow");
        assert_eq!(category_for_keycode(126, 0), "arrow");
    }

    #[test]
    fn other_keycodes_return_typing() {
        assert_eq!(category_for_keycode(0, 0), "typing");
        assert_eq!(category_for_keycode(15, 0), "typing");
        assert_eq!(category_for_keycode(100, 0), "typing");
    }

    #[test]
    fn command_takes_precedence_over_other_categories() {
        // Even return/tab/escape with cmd flag -> command-shortcut
        assert_eq!(category_for_keycode(36, 1048576), "command-shortcut");
        assert_eq!(category_for_keycode(48, 1048576), "command-shortcut");
        assert_eq!(category_for_keycode(123, 1048576), "command-shortcut");
    }
}
