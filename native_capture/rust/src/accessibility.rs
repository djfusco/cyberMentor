//! Accessibility text extraction via the macOS Accessibility API (AXUIElement).
//!
//! Mirrors `AccessibilityObserver.swift` in the Swift native_mac helper — the
//! quality bar this binary matches. Reads the focused element's value/title/
//! selection, the focused window's title, and a shallow, bounded walk of the
//! focused window's children (depth 5, 60 nodes, 10k chars).
//!
//! Uses raw C FFI to the ApplicationServices framework (AXUIElement is a C
//! API, not ObjC) plus CoreFoundation helpers for CFString/CFArray/CFType.
//! The objc2 crates are used only for ObjC APIs (NSWorkspace, Vision).

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::os::raw::c_float;

    // -- CoreFoundation type aliases --

    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFMutableDictionaryRef = *mut c_void;
    type CFArrayRef = *const c_void;
    type CFTypeID = usize;
    type CFIndex = isize;
    type Boolean = u8;

    // -- AX type aliases --

    pub type AXUIElementRef = *const c_void;
    type AXError = c_int;
    #[allow(non_camel_case_types)]
    type pid_t = i32;

    // -- AX error codes --

    const AX_SUCCESS: AXError = 0;

    // -- AXValue types (for reading CGPoint/CGSize from AXPosition/AXSize) --

    type AXValueRef = CFTypeRef;
    type AXValueType = u32;

    // kAXValueCGPointType = 1, kAXValueCGSizeType = 2 (AXValue.h).
    const K_AX_VALUE_CG_POINT_TYPE: AXValueType = 1;
    const K_AX_VALUE_CG_SIZE_TYPE: AXValueType = 2;

    /// CGPoint (CGFloat is f64 on 64-bit macOS).
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    /// CGSize (CGFloat is f64 on 64-bit macOS).
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct CGSize {
        width: f64,
        height: f64,
    }

    // -- AX attribute string constants --

    const AX_FOCUSED_WINDOW: &str = "AXFocusedWindow";
    const AX_FOCUSED_UI_ELEMENT: &str = "AXFocusedUIElement";
    const AX_TITLE: &str = "AXTitle";
    const AX_VALUE: &str = "AXValue";
    const AX_SELECTED_TEXT: &str = "AXSelectedText";
    const AX_CHILDREN: &str = "AXChildren";
    const AX_PARENT: &str = "AXParent";
    const AX_POSITION: &str = "AXPosition";
    const AX_SIZE: &str = "AXSize";

    // -- FFI: CoreFoundation --

    extern "C" {
        fn CFRelease(cf: CFTypeRef);
        fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
        fn CFEqual(cf1: CFTypeRef, cf2: CFTypeRef) -> Boolean;
        fn CFGetTypeID(cf: CFTypeRef) -> CFTypeID;
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            cstr: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFStringGetCStringPtr(
            the_string: CFStringRef,
            encoding: u32,
        ) -> *const c_char;
        fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: CFIndex,
            encoding: u32,
        ) -> Boolean;
        fn CFArrayGetCount(the_array: CFArrayRef) -> CFIndex;
        fn CFArrayGetValueAtIndex(the_array: CFArrayRef, idx: CFIndex) -> *const c_void;
        fn CFDictionaryCreateMutable(
            alloc: *const c_void,
            capacity: CFIndex,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> CFMutableDictionaryRef;
        fn CFDictionarySetValue(
            dict: CFMutableDictionaryRef,
            key: *const c_void,
            value: *const c_void,
        );

        // kCFBooleanTrue is a CFBooleanRef extern constant in CoreFoundation.
        static kCFBooleanTrue: CFTypeRef;
    }

    // kCFStringEncodingUTF8
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;

    // -- FFI: ApplicationServices / HIServices (AX) --

    extern "C" {
        fn AXIsProcessTrusted() -> Boolean;
        fn AXIsProcessTrustedWithOptions(options: CFTypeRef) -> Boolean;
        fn AXUIElementCreateApplication(pid: pid_t) -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        fn AXUIElementSetMessagingTimeout(
            element: AXUIElementRef,
            timeout: c_float,
        ) -> AXError;
        fn AXUIElementGetTypeID() -> CFTypeID;
        fn AXValueGetType(value: AXValueRef) -> AXValueType;
        fn AXValueGetValue(
            value: AXValueRef,
            the_type: AXValueType,
            value_ptr: *mut c_void,
        ) -> Boolean;

        // kAXTrustedCheckOptionPrompt is a CFStringRef extern constant
        // in HIServices.framework (linked via ApplicationServices).
        static kAXTrustedCheckOptionPrompt: CFStringRef;
    }

    // -- AxElement: owned AXUIElementRef wrapper --

    /// Owns an AXUIElement reference, releasing it on drop. AXUIElement is a
    /// CFType (reference-counted, immutable ref), so it's safe to share across
    /// threads — we implement Send+Sync manually.
    pub struct AxElement {
        raw: AXUIElementRef,
    }

    // SAFETY: AXUIElement is a CoreFoundation reference-counted type. The ref
    // itself is immutable; AXUIElementCopyAttributeValue is thread-safe. Safe
    // to send and share across threads.
    unsafe impl Send for AxElement {}
    unsafe impl Sync for AxElement {}

    impl AxElement {
        /// Wraps a raw AXUIElementRef, taking ownership (will CFRelease on drop).
        /// Returns None if the ref is null.
        pub fn from_raw(raw: AXUIElementRef) -> Option<Self> {
            if raw.is_null() {
                None
            } else {
                Some(Self { raw })
            }
        }

        pub fn raw(&self) -> AXUIElementRef {
            self.raw
        }

        /// Creates an AXUIElement for the application with the given PID.
        pub fn create_application(pid: pid_t) -> Option<Self> {
            unsafe {
                let raw = AXUIElementCreateApplication(pid);
                let el = Self::from_raw(raw)?;
                // Set a 1-second messaging timeout so a hung app can't block us.
                AXUIElementSetMessagingTimeout(el.raw, 1.0);
                Some(el)
            }
        }

        /// Reads an attribute that is itself an AXUIElement (e.g. AXFocusedWindow,
        /// AXParent, AXFocusedUIElement). Returns None if the attribute is
        /// missing or not an AXUIElement.
        pub fn element_attribute(&self, attr: &str) -> Option<AxElement> {
            let value = self.copy_attribute(attr)?;
            unsafe {
                if CFGetTypeID(value) == AXUIElementGetTypeID() {
                    AxElement::from_raw(value as AXUIElementRef)
                } else {
                    CFRelease(value);
                    None
                }
            }
        }

        /// Reads a string attribute (e.g. AXTitle, AXValue, AXSelectedText).
        pub fn string_attribute(&self, attr: &str) -> Option<String> {
            let value = self.copy_attribute(attr)?;
            unsafe {
                let s = cf_string_to_rust(value as CFStringRef);
                CFRelease(value);
                s
            }
        }

        /// Reads the AXChildren attribute as a Vec of child AXUIElements.
        /// Each child is CFRetain'd before wrapping so it survives the array's
        /// release (CFArrayGetValueAtIndex returns a non-retained borrow).
        pub fn children(&self) -> Vec<AxElement> {
            let value = match self.copy_attribute(AX_CHILDREN) {
                Some(v) => v,
                None => return vec![],
            };
            unsafe {
                let count = CFArrayGetCount(value as CFArrayRef);
                let mut result = Vec::with_capacity(count.min(128) as usize);
                for i in 0..count {
                    let child = CFArrayGetValueAtIndex(value as CFArrayRef, i) as AXUIElementRef;
                    if !child.is_null() {
                        // CFRetain the child so it has its own ownership stake
                        // independent of the array. Without this, CFRelease(value)
                        // below would release the array's elements, and then
                        // AxElement::drop would double-release them.
                        CFRetain(child);
                        if let Some(el) = AxElement::from_raw(child) {
                            result.push(el);
                        }
                    }
                }
                CFRelease(value);
                result
            }
        }

        /// Reads the element's on-screen position (AXPosition) and size
        /// (AXSize) via the AX C FFI and returns them as a screen-space
        /// `Rect` in pixel coordinates. Returns `None` if either attribute
        /// is missing, the value isn't a CGPoint/CGSize, or the rect is
        /// empty. Used to scope OCR to the foreground window's bounds so
        /// only the foreground window is OCR'd instead of the whole screen.
        ///
        /// Runs inside `objc2::exception::catch` (via `ax_catch`) so an
        /// ObjC exception can't abort the process. Both AXValue refs
        /// (position + size) are released exactly once regardless of
        /// success or failure (no double-free, no leak).
        pub fn window_bounds(&self) -> Option<crate::ocr::Rect> {
            ax_catch(|| {
                let position_ref = self.copy_attribute(AX_POSITION)?;
                let size_ref = self.copy_attribute(AX_SIZE);
                unsafe {
                    let result = (|| {
                        let size_ref = size_ref?;
                        let point = extract_cgpoint(position_ref)?;
                        let size = extract_cgsize(size_ref)?;
                        // CGFloat -> u32, clamping negatives to 0.
                        let x = point.x.max(0.0) as u32;
                        let y = point.y.max(0.0) as u32;
                        let width = size.width.max(0.0) as u32;
                        let height = size.height.max(0.0) as u32;
                        if width == 0 || height == 0 {
                            return None;
                        }
                        Some(crate::ocr::Rect { x, y, width, height })
                    })();
                    // Always release both refs (copy_attribute returns a
                    // +1 retained ref that we own — exactly one CFRelease
                    // each, no double-free).
                    CFRelease(position_ref);
                    if let Some(sr) = size_ref {
                        CFRelease(sr);
                    }
                    result
                }
            })?
        }

        /// Low-level: copies an attribute value (caller must CFRelease the
        /// returned ref). Returns None on error or null.
        fn copy_attribute(&self, attr: &str) -> Option<CFTypeRef> {
            let cf_attr = cf_string_from_rust(attr)?;
            unsafe {
                let mut value: CFTypeRef = std::ptr::null();
                let err = AXUIElementCopyAttributeValue(self.raw, cf_attr, &mut value);
                CFRelease(cf_attr as CFTypeRef);
                if err == AX_SUCCESS && !value.is_null() {
                    Some(value)
                } else {
                    None
                }
            }
        }
    }

    impl Drop for AxElement {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { CFRelease(self.raw); }
            }
        }
    }

    impl Clone for AxElement {
        fn clone(&self) -> Self {
            // CFRetain the ref so the clone has its own ownership stake.
            unsafe { CFRetain(self.raw); }
            Self { raw: self.raw }
        }
    }

    // -- CFString helpers --

    /// Creates a CFString from a Rust &str. Caller must CFRelease the result.
    fn cf_string_from_rust(s: &str) -> Option<CFStringRef> {
        let cstr = CString::new(s).ok()?;
        unsafe {
            let cf = CFStringCreateWithCString(
                std::ptr::null(),
                cstr.as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            if cf.is_null() { None } else { Some(cf) }
        }
    }

    /// Converts a CFStringRef to a Rust String. Does NOT release the CFString.
    fn cf_string_to_rust(cf: CFStringRef) -> Option<String> {
        if cf.is_null() {
            return None;
        }
        unsafe {
            // Fast path: direct C string pointer
            let ptr = CFStringGetCStringPtr(cf, K_CF_STRING_ENCODING_UTF8);
            if !ptr.is_null() {
                return std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string());
            }
            // Slow path: copy into a buffer
            let max_buf = 4096;
            let mut buf = vec![0i8; max_buf];
            if CFStringGetCString(
                cf,
                buf.as_mut_ptr(),
                max_buf as CFIndex,
                K_CF_STRING_ENCODING_UTF8,
            ) != 0
            {
                std::ffi::CStr::from_ptr(buf.as_ptr())
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            } else {
                None
            }
        }
    }

    /// Extracts a CGPoint from an AXValue ref. Does NOT release the ref.
    /// Returns `None` if the value is null or not a CGPoint-typed AXValue.
    unsafe fn extract_cgpoint(value: CFTypeRef) -> Option<CGPoint> {
        if value.is_null() {
            return None;
        }
        let the_type = AXValueGetType(value);
        if the_type != K_AX_VALUE_CG_POINT_TYPE {
            return None;
        }
        let mut point = CGPoint::default();
        if AXValueGetValue(value, the_type, &mut point as *mut CGPoint as *mut c_void) != 0 {
            Some(point)
        } else {
            None
        }
    }

    /// Extracts a CGSize from an AXValue ref. Does NOT release the ref.
    /// Returns `None` if the value is null or not a CGSize-typed AXValue.
    unsafe fn extract_cgsize(value: CFTypeRef) -> Option<CGSize> {
        if value.is_null() {
            return None;
        }
        let the_type = AXValueGetType(value);
        if the_type != K_AX_VALUE_CG_SIZE_TYPE {
            return None;
        }
        let mut size = CGSize::default();
        if AXValueGetValue(value, the_type, &mut size as *mut CGSize as *mut c_void) != 0 {
            Some(size)
        } else {
            None
        }
    }

    // -- Exception-safe wrapper --
    //
    // AX APIs (and objc2 msg_send! calls) can raise ObjC exceptions like
    // NSAccessibilityException. If such an exception unwinds across the FFI
    // boundary into Rust, Rust cannot catch foreign exceptions and aborts
    // (SIGABRT). We wrap every AX/objc2 call in objc2::exception::catch,
    // which sets up an ObjC @try/@catch that intercepts the exception before
    // it reaches Rust, converting it to a logged warning + None.

    /// Runs `closure` inside objc2::exception::catch. If an ObjC exception is
    /// raised, logs it to stderr and returns None. Otherwise returns Some(R).
    fn ax_catch<R>(closure: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
        match objc2::exception::catch(closure) {
            Ok(value) => Some(value),
            Err(exc) => {
                if let Some(exc) = exc {
                    eprintln!(
                        "cyberalfred-capture: AX exception caught: {}",
                        exc
                    );
                } else {
                    eprintln!("cyberalfred-capture: AX exception caught (nil)");
                }
                None
            }
        }
    }

    // -- Public API (mirrors AccessibilityObserver.swift) --

    /// Non-prompting permission check, safe to call from `check`.
    pub fn check_permission() -> bool {
        ax_catch(|| unsafe { AXIsProcessTrusted() != 0 }).unwrap_or(false)
    }

    /// Triggers the standard macOS Accessibility consent prompt if not yet
    /// granted. Call at most once, at session start.
    pub fn request_permission_prompt_if_needed() {
        let _ = ax_catch(|| unsafe {
            let dict = CFDictionaryCreateMutable(
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if !dict.is_null() {
                CFDictionarySetValue(
                    dict,
                    kAXTrustedCheckOptionPrompt as *const c_void,
                    kCFBooleanTrue as *const c_void,
                );
                AXIsProcessTrustedWithOptions(dict as CFTypeRef);
                CFRelease(dict as CFTypeRef);
            }
        });
    }

    /// The raw focused-window AXUIElement for a PID — exposed so callers can
    /// compare window *identity* (via CFEqual) rather than title text.
    pub fn focused_window_element(pid: i32) -> Option<AxElement> {
        ax_catch(|| {
            let app = AxElement::create_application(pid)?;
            app.element_attribute(AX_FOCUSED_WINDOW)
        })?
    }

    /// The focused window's title.
    pub fn window_title(window: &AxElement) -> Option<String> {
        ax_catch(|| window.string_attribute(AX_TITLE))?
    }

    /// The element's AX parent, if any — used for ancestor walks (spawned
    /// dialog recognition).
    pub fn parent_element(element: &AxElement) -> Option<AxElement> {
        ax_catch(|| element.element_attribute(AX_PARENT))?
    }

    /// Compares two AXUIElement refs by identity (CFEqual).
    pub fn elements_equal(a: &AxElement, b: &AxElement) -> bool {
        ax_catch(|| unsafe { CFEqual(a.raw(), b.raw()) != 0 }).unwrap_or(false)
    }

    /// A small, bounded snapshot of visible text for the frontmost app:
    /// the focused element's value/title/selection, the focused window's
    /// title, and a shallow, capped walk of its immediate descendants.
    ///
    /// Mirrors `AccessibilityObserver.focusedText(for:)`:
    /// - depth limit: 5, node limit: 60, char limit: 10k
    /// - cycle dedup via pointer-address HashSet
    /// - values > 4000 chars or titles > 500 chars are skipped (same as Swift)
    ///
    /// The entire function runs inside objc2::exception::catch so any ObjC
    /// exception (NSAccessibilityException, etc.) is caught at the FFI
    /// boundary and converted to None, not allowed to abort the process.
    pub fn focused_text(pid: i32) -> Option<String> {
        ax_catch(|| {
            let app = AxElement::create_application(pid)?;
            let mut collected: Vec<String> = Vec::new();

            // Focused element: value / title / selected text
            if let Some(focused) = app.element_attribute(AX_FOCUSED_UI_ELEMENT) {
                for attr in [AX_VALUE, AX_TITLE, AX_SELECTED_TEXT] {
                    if let Some(s) = focused.string_attribute(attr) {
                        if !s.is_empty() {
                            collected.push(s);
                        }
                    }
                }
            }

            // Focused window: title + bounded walk of children
            if let Some(window) = app.element_attribute(AX_FOCUSED_WINDOW) {
                if let Some(title) = window.string_attribute(AX_TITLE) {
                    if !title.is_empty() {
                        collected.push(title);
                    }
                }
                let mut node_count = 0usize;
                let mut visited = std::collections::HashSet::new();
                collect_shallow(
                    &window,
                    0,
                    &mut node_count,
                    &mut collected,
                    &mut visited,
                );
            }

            if collected.is_empty() {
                return None;
            }
            let mut joined = collected.join("\n");
            if joined.len() > crate::events::MAX_TEXT_LENGTH {
                joined.truncate(crate::events::MAX_TEXT_LENGTH);
            }
            Some(joined)
        })?
    }

    /// Bounded recursive walk of an element's children, collecting value/title
    /// text. Mirrors `collectShallow` in AccessibilityObserver.swift.
    fn collect_shallow(
        element: &AxElement,
        depth: usize,
        node_count: &mut usize,
        collected: &mut Vec<String>,
        visited: &mut std::collections::HashSet<usize>,
    ) {
        if depth >= crate::events::AX_MAX_DEPTH || *node_count >= crate::events::AX_MAX_NODES {
            return;
        }

        let key = element.raw() as usize;
        if visited.contains(&key) {
            return;
        }
        visited.insert(key);

        for child in element.children() {
            if *node_count >= crate::events::AX_MAX_NODES {
                return;
            }
            *node_count += 1;

            if let Some(value) = child.string_attribute(AX_VALUE) {
                if !value.is_empty() && value.len() < 4000 {
                    collected.push(value);
                }
            } else if let Some(title) = child.string_attribute(AX_TITLE) {
                if !title.is_empty() && title.len() < 500 {
                    collected.push(title);
                }
            }

            collect_shallow(&child, depth + 1, node_count, collected, visited);
        }
    }

    /// Returns the bundle identifier of the app with the given PID, via
    /// NSRunningApplication. None if the app can't be found.
    #[allow(dead_code)]
    pub fn bundle_id_for_pid(pid: i32) -> Option<String> {
        ax_catch(|| {
            use objc2_app_kit::NSRunningApplication;
            let app = NSRunningApplication::runningApplicationWithProcessIdentifier(pid);
            app?.bundleIdentifier()?.to_string().into()
        })?
    }
}

#[cfg(target_os = "macos")]
pub use platform::{
    check_permission, elements_equal, focused_text, focused_window_element,
    parent_element, request_permission_prompt_if_needed, window_title, AxElement,
};

// -- Windows: UI Automation text extraction (analog of macOS Accessibility) --

#[cfg(target_os = "windows")]
mod platform {
    use std::cell::OnceCell;

    use uiautomation::types::{Handle, UIProperty};
    use uiautomation::{UIAutomation, UIElement, UITreeWalker};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{GetWindow, GetWindowTextLengthW, GetWindowTextW, GW_OWNER};

    /// The `text_source` label for AX-equivalent text on this platform.
    /// macOS uses "accessibility"; Windows uses "ui_automation" (matching
    /// the C# helper's `TextSource = "ui_automation"`). Kept for API parity
    /// with the macOS module (main.rs uses a separate cfg-gated constant).
    #[allow(dead_code)]
    pub const TEXT_SOURCE: &str = "ui_automation";

    // UIAutomation is !Send/!Sync (it owns a COM interface pointer), so we
    // keep it in a thread_local OnceCell. The capture loop runs on the main
    // thread, so this is single-threaded. UIAutomation::new() CoInitializes
    // COINIT_MULTITHREADED on first access; subsequent calls reuse the cached
    // instance. If COM init or CUIAutomation instantiation fails (e.g. UI
    // Automation not registered), the cell caches None and all text
    // extraction degrades to None — never crashes.
    thread_local! {
        static UIA: OnceCell<Option<UIAutomation>> = OnceCell::new();
    }

    /// Runs `f` with the cached UIAutomation instance for this thread,
    /// initializing it lazily on first call. Returns None if UI Automation
    /// is unavailable.
    fn with_uia<R>(f: impl FnOnce(&UIAutomation) -> Option<R>) -> Option<R> {
        UIA.with(|cell| {
            let opt = cell.get_or_init(|| UIAutomation::new().ok());
            opt.as_ref().and_then(|uia| f(uia))
        })
    }

    /// AxElement: an owned HWND identity wrapper. On Windows the foreground
    /// window's HWND is the stable identity (analog of macOS AXUIElement).
    /// Naturally Send+Sync (isize), so no manual unsafe impl needed.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct AxElement {
        hwnd: isize,
    }

    impl AxElement {
        pub fn from_hwnd(hwnd: isize) -> Option<Self> {
            if hwnd == 0 { None } else { Some(Self { hwnd }) }
        }

        pub fn hwnd(&self) -> isize {
            self.hwnd
        }
    }

    /// True if CUIAutomation is available (instantiable). Used by `check`.
    /// Mirrors the C# `UiAutomation.IsAvailable()`.
    pub fn check_permission() -> bool {
        with_uia(|_| Some(())).is_some()
    }

    /// No-op on Windows: UI Automation does not require explicit user
    /// permission (unlike macOS Accessibility). Provided for API parity.
    #[allow(dead_code)]
    pub fn request_permission_prompt_if_needed() {}

    /// The foreground window's HWND as an AxElement. The `pid` parameter is
    /// accepted for macOS API parity but not used — on Windows the foreground
    /// window IS the element we want, identified by HWND (not PID).
    pub fn focused_window_element(_pid: i32) -> Option<AxElement> {
        crate::foreground_window::try_get().map(|fg| AxElement { hwnd: fg.hwnd })
    }

    /// The window's title text via `GetWindowTextW`.
    pub fn window_title(element: &AxElement) -> Option<String> {
        unsafe {
            let hwnd = HWND(element.hwnd as *mut std::ffi::c_void);
            let len = GetWindowTextLengthW(hwnd);
            if len <= 0 {
                return None;
            }
            let capacity = (len + 1) as usize;
            let mut buf = vec![0u16; capacity];
            let copied = GetWindowTextW(hwnd, &mut buf);
            if copied <= 0 {
                return None;
            }
            let end = buf[..copied as usize]
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(copied as usize);
            let title = String::from_utf16_lossy(&buf[..end]);
            if title.is_empty() { None } else { Some(title) }
        }
    }

    /// The window's owner/parent HWND, via `GetWindow(GW_OWNER)`. Used for
    /// the ancestor walk in self-capture exclusion (recognizing spawned
    /// dialogs as belonging to the launcher window). Returns None for
    /// top-level windows with no owner.
    pub fn parent_element(element: &AxElement) -> Option<AxElement> {
        unsafe {
            let hwnd = HWND(element.hwnd as *mut std::ffi::c_void);
            let owner = GetWindow(hwnd, GW_OWNER).ok()?;
            if owner.is_invalid() {
                return None;
            }
            AxElement::from_hwnd(owner.0.expose_provenance() as isize)
        }
    }

    /// Compares two AxElements by HWND identity.
    pub fn elements_equal(a: &AxElement, b: &AxElement) -> bool {
        a.hwnd == b.hwnd
    }

    /// Best-effort, bounded snapshot of visible text for the foreground
    /// window via UI Automation. Mirrors `UiAutomation.TryGetFocusedText`:
    /// the focused element's Name + Value, plus a shallow, capped walk of
    /// the foreground window's descendants (depth 5, 60 nodes, 10k chars).
    /// The `pid` parameter is accepted for macOS API parity but not used.
    ///
    /// Every UI Automation call is wrapped in Result handling (`.ok()`), so
    /// a failure on one element never aborts the walk — matching the C#
    /// helper's per-element try/catch. Returns None if UI Automation is
    /// unavailable or yields nothing.
    pub fn focused_text(_pid: i32) -> Option<String> {
        let fg = crate::foreground_window::try_get()?;
        with_uia(|uia| {
            let mut collected: Vec<String> = Vec::new();

            // Focused element: Name + Value (mirrors C# GetFocusedElement).
            if let Ok(focused) = uia.get_focused_element() {
                if let Ok(name) = focused.get_name() {
                    append_if_useful(&mut collected, &name, usize::MAX);
                }
                if let Ok(value) = focused.get_property_value(UIProperty::ValueValue) {
                    if let Ok(s) = value.get_string() {
                        append_if_useful(&mut collected, &s, 4000);
                    }
                }
            }

            // Bounded walk of the foreground window's descendants via
            // RawViewWalker (mirrors C# RawViewWalker).
            if let Ok(window) = uia.element_from_handle(Handle::from(fg.hwnd)) {
                if let Ok(walker) = uia.get_raw_view_walker() {
                    let mut node_count = 0usize;
                    walk(&walker, &window, 0, &mut node_count, &mut collected);
                }
            }

            if collected.is_empty() {
                return None;
            }
            let mut joined = collected.join("\n");
            if joined.len() > crate::events::MAX_TEXT_LENGTH {
                joined.truncate(crate::events::MAX_TEXT_LENGTH);
            }
            Some(joined)
        })
    }

    /// Bounded depth-first walk: first child -> process -> recurse -> next
    /// sibling. Mirrors `Walk` in UiAutomation.cs. Per-element errors are
    /// swallowed (`.ok()` / match on Err) so one bad element never aborts
    /// the walk.
    fn walk(
        walker: &UITreeWalker,
        element: &UIElement,
        depth: usize,
        node_count: &mut usize,
        collected: &mut Vec<String>,
    ) {
        if depth >= crate::events::AX_MAX_DEPTH || *node_count >= crate::events::AX_MAX_NODES {
            return;
        }

        let mut child = match walker.get_first_child(element) {
            Ok(c) => c,
            Err(_) => return,
        };

        loop {
            if *node_count >= crate::events::AX_MAX_NODES {
                return;
            }
            *node_count += 1;

            // Try Value first (max 4000 chars), fall back to Name (max 500).
            let got_value = match child.get_property_value(UIProperty::ValueValue) {
                Ok(value) => match value.get_string() {
                    Ok(s) => append_if_useful(collected, &s, 4000),
                    Err(_) => false,
                },
                Err(_) => false,
            };
            if !got_value {
                if let Ok(name) = child.get_name() {
                    append_if_useful(collected, &name, 500);
                }
            }

            // Recurse into children.
            walk(walker, &child, depth + 1, node_count, collected);

            // Next sibling.
            child = match walker.get_next_sibling(&child) {
                Ok(next) => next,
                Err(_) => break,
            };
        }
    }

    /// Appends a trimmed, non-empty string (capped at `max_len`) to
    /// `collected`. Returns true if appended. Mirrors C# `AppendIfUseful`.
    fn append_if_useful(collected: &mut Vec<String>, value: &str, max_len: usize) -> bool {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return false;
        }
        let s = if trimmed.len() > max_len {
            &trimmed[..max_len]
        } else {
            trimmed
        };
        collected.push(s.to_string());
        true
    }

    /// Returns the process name (without extension) for the given PID —
    /// the Windows analog of macOS's bundle identifier. Used as `bundle_id`.
    #[allow(dead_code)]
    pub fn bundle_id_for_pid(pid: i32) -> Option<String> {
        crate::foreground_window::process_name_for_pid(pid as u32)
    }
}

#[cfg(target_os = "windows")]
#[allow(unused_imports)]
pub use platform::{
    check_permission, elements_equal, focused_text, focused_window_element,
    parent_element, request_permission_prompt_if_needed, window_title, AxElement,
    TEXT_SOURCE,
};

// -- Stubs for other platforms (neither macOS nor Windows) --

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod stubs {
    pub fn check_permission() -> bool { false }
    pub fn request_permission_prompt_if_needed() {}
    pub fn focused_text(_pid: i32) -> Option<String> { None }
    pub fn focused_window_element(_pid: i32) -> Option<()> { None }
    pub fn window_title(_el: &()) -> Option<String> { None }
    pub fn parent_element(_el: &()) -> Option<()> { None }
    pub fn elements_equal(_a: &(), _b: &()) -> bool { false }
    pub fn bundle_id_for_pid(_pid: i32) -> Option<String> { None }
    pub struct AxElement;
    pub const TEXT_SOURCE: &str = "accessibility";
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub use stubs::*;

#[cfg(test)]
mod tests {
    // The AX logic requires Accessibility permission and a live app, so we
    // can't unit-test the actual text extraction. The pure decision logic
    // (is_launcher_window etc.) is tested in self_capture.rs. Here we only
    // test that the module compiles and the public API exists.

    #[cfg(target_os = "macos")]
    #[test]
    fn check_permission_does_not_crash() {
        let _ = super::check_permission();
    }
}
