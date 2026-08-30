//! Vision OCR via the macOS Vision framework.
//!
//! Mirrors `OCRService.swift` in the Swift native_mac helper. Runs
//! VNRecognizeTextRequest on a retained frame's CGImage:
//! - recognitionLevel = accurate
//! - usesLanguageCorrection = false
//! - joins topCandidates(1) with newlines
//! - 10k char cap
//!
//! Runs ONLY when AX text is empty for the current app, OR always when the
//! frontmost app is a terminal (terminals don't expose text via AX).
//! Runs inside an autorelease pool.

// -- Shared (cross-platform): Rect + BGRA crop helper --

/// A rectangle in screen pixel coordinates. Used to scope OCR to the
/// foreground window's bounding box on both macOS and Windows, so only
/// the foreground window is OCR'd instead of the whole screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Crops a raw BGRA frame buffer to `rect`, clamping to the frame bounds.
/// Returns the cropped pixel buffer plus its width and height, or `None`
/// if the rect is empty or entirely out of bounds. The returned buffer has
/// tightly packed rows (no stride padding): each row is `width * 4` bytes.
pub fn crop_bgra_to_window(
    bgra: &[u8],
    frame_width: u32,
    frame_height: u32,
    rect: Rect,
) -> Option<(Vec<u8>, u32, u32)> {
    if frame_width == 0 || frame_height == 0 {
        return None;
    }
    // Clamp the rect to the frame bounds.
    let x0 = rect.x.min(frame_width);
    let y0 = rect.y.min(frame_height);
    let x1 = rect.x.saturating_add(rect.width).min(frame_width);
    let y1 = rect.y.saturating_add(rect.height).min(frame_height);
    let crop_w = x1.checked_sub(x0)?;
    let crop_h = y1.checked_sub(y0)?;
    if crop_w == 0 || crop_h == 0 {
        return None;
    }
    let fw = frame_width as usize;
    let fh = frame_height as usize;
    // Guard against a short buffer (shouldn't happen with scap, but be safe).
    if bgra.len() < fw * fh * 4 {
        return None;
    }
    let row_bytes = crop_w as usize * 4;
    let mut out = Vec::with_capacity(row_bytes * crop_h as usize);
    for y in y0..y1 {
        let src_start = ((y as usize) * fw + (x0 as usize)) * 4;
        let src_end = src_start + row_bytes;
        out.extend_from_slice(&bgra[src_start..src_end]);
    }
    Some((out, crop_w, crop_h))
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::c_void;

    use objc2::rc::Retained;
    use objc2::AllocAnyThread;
    use objc2_core_foundation::CFRetained;
    use objc2_core_graphics::{
        CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage,
    };
    use objc2_foundation::{NSArray, NSDictionary, NSString};
    use objc2_vision::{
        VNImageRequestHandler, VNRecognizeTextRequest, VNRequestTextRecognitionLevel,
    };

    // kCGImageAlphaPremultipliedLast = 1
    const K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST: u32 = 1;

    /// Recognizes text in a raw BGRA frame buffer using Vision.
    /// Returns the joined text (newlines between lines), capped at 10k chars,
    /// or None if OCR fails or produces no text.
    ///
    /// The entire Vision pipeline runs inside objc2::exception::catch so
    /// any ObjC exception is caught at the FFI boundary and converted to
    /// None, not allowed to abort the process.
    ///
    /// `bgra` is the raw pixel data in BGRA order, `width` x `height`.
    pub fn recognize_text(bgra: &[u8], width: u32, height: u32) -> Option<String> {
        if width == 0 || height == 0 || bgra.is_empty() {
            return None;
        }

        // Convert BGRA -> RGBA (CGImage expects RGBA with premultiplied alpha).
        let mut rgba = bgra.to_vec();
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2); // B <-> R
            pixel[3] = 255; // Ensure alpha is opaque for screen captures
        }

        // Wrap the unsafe CGImage + Vision call in exception::catch.
        // AssertUnwindSafe is needed because the closure captures `rgba`
        // by move, which isn't inherently UnwindSafe.
        let result = objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
            unsafe {
                let cg_image = create_cg_image(&rgba, width as usize, height as usize)?;
                run_vision_ocr(&cg_image)
            }
        }));

        match result {
            Ok(text) => text,
            Err(exc) => {
                if let Some(exc) = exc {
                    eprintln!("cyberalfred-capture: OCR exception caught: {}", exc);
                } else {
                    eprintln!("cyberalfred-capture: OCR exception caught (nil)");
                }
                None
            }
        }
    }

    /// Creates a CGImage from raw RGBA bytes using objc2-core-graphics typed
    /// APIs. Returns a CFRetained<CGImage> that auto-releases on drop.
    unsafe fn create_cg_image(rgba: &[u8], width: usize, height: usize) -> Option<CFRetained<CGImage>> {
        let bytes_per_row = width * 4;
        let data_size = bytes_per_row * height;

        // Create a data provider that borrows `rgba`. No release callback
        // since `rgba` is a local Vec that outlives the CGImage (the
        // CFRetained<CGImage> is dropped before recognize_text returns).
        let provider = CGDataProvider::with_data(
            std::ptr::null_mut(),
            rgba.as_ptr() as *const c_void,
            data_size,
            None, // no release callback — data is borrowed
        )?;

        let color_space = CGColorSpace::new_device_rgb()?;

        let bitmap_info = CGBitmapInfo(K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST);

        CGImage::new(
            width,
            height,
            8,    // bits per component
            32,   // bits per pixel
            bytes_per_row,
            Some(&color_space),
            bitmap_info,
            Some(&provider),
            std::ptr::null(), // decode
            true,             // should interpolate
            CGColorRenderingIntent::RenderingIntentDefault,
        )
    }

    /// Runs VNRecognizeTextRequest on a CGImage and returns the recognized
    /// text (topCandidates(1) joined with newlines, capped at 10k chars).
    unsafe fn run_vision_ocr(cg_image: &CGImage) -> Option<String> {
        // Create the text recognition request.
        let request = VNRecognizeTextRequest::new();
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        request.setUsesLanguageCorrection(false);

        // Keep a clone to read results after performRequests.
        let request_for_results = request.clone();

        // Upcast VNRecognizeTextRequest -> VNImageBasedRequest -> VNRequest
        // so we can put it in an NSArray<VNRequest> for performRequests.
        // These are ObjC types, so use Retained::into_super (not CFRetained).
        let request_img: Retained<objc2_vision::VNImageBasedRequest> =
            Retained::into_super(request);
        let request_obj: Retained<objc2_vision::VNRequest> =
            Retained::into_super(request_img);
        let requests_array: Retained<NSArray<objc2_vision::VNRequest>> =
            NSArray::from_retained_slice(&[request_obj]);

        // Create VNImageRequestHandler using the typed init method.
        // VNImageOption = NSString, so the options dict type is
        // NSDictionary<NSString, AnyObject>. Create it directly with
        // the correct generic types (type inference from annotation).
        let empty_dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::new();

        let handler = VNImageRequestHandler::initWithCGImage_options(
            VNImageRequestHandler::alloc(),
            cg_image,
            &empty_dict,
        );

        // Perform the request (synchronous).
        match handler.performRequests_error(&requests_array) {
            Ok(()) => {}
            Err(err) => {
                eprintln!("cyberalfred-capture: OCR failed: {}", err);
                return None;
            }
        }

        // Extract observations from the request's results().
        let observations = request_for_results.results()?;

        let mut lines: Vec<String> = Vec::new();
        for obs in observations.iter() {
            let candidates = obs.topCandidates(1);
            if let Some(candidate) = candidates.firstObject() {
                lines.push(candidate.string().to_string());
            }
        }

        if lines.is_empty() {
            return None;
        }
        let mut joined = lines.join("\n");
        if joined.len() > crate::events::MAX_TEXT_LENGTH {
            joined.truncate(crate::events::MAX_TEXT_LENGTH);
        }
        Some(joined)
    }

    /// Returns true if the given bundle_id is a known terminal app (terminals
    /// don't expose text via AX, so OCR is always used for them).
    pub fn is_terminal_bundle_id(bundle_id: Option<&str>) -> bool {
        match bundle_id {
            Some(bid) => {
                const TERMINAL_BUNDLES: &[&str] = &[
                    "com.apple.Terminal",
                    "com.googlecode.iterm2",
                    "com.todesktop.682d041e3c80b9da", // Warp
                    "com.microsoft.VSCode",
                    "com.jetbrains.intellij",
                    "com.jetbrains.pycharm",
                    "com.jetbrains.rustrover",
                    "com.googlecode.go",
                    "org.gnu.Emacs",
                    "com.neovide.neovide",
                    "io.alacritty",
                    "net.kovidgoyal.kitty",
                    "com.github.wez.wezterm",
                    "com.mitchellh.ghostty",
                ];
                TERMINAL_BUNDLES.contains(&bid)
            }
            None => false,
        }
    }
}

#[cfg(target_os = "macos")]
pub use platform::{is_terminal_bundle_id, recognize_text};

// -- Windows: WinRT OCR via Windows.Media.Ocr --

#[cfg(target_os = "windows")]
mod platform {
    use std::time::Duration;

    use windows::core::{Interface, RuntimeType};
    use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine;
    use windows::Storage::Streams::Buffer;
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
    use windows::Win32::System::WinRT::IBufferByteAccess;
    use windows_future::{AsyncStatus, IAsyncOperation};

    /// Recognizes text in a raw BGRA frame buffer using WinRT OCR
    /// (`Windows.Media.Ocr`). Mirrors the macOS Vision OCR path: joins
    /// recognized lines with newlines, capped at 10k chars. Returns `None`
    /// if OCR is unavailable, the image can't be decoded, or no text is
    /// recognized — never crashes.
    ///
    /// The BGRA pixels are copied directly into a `SoftwareBitmap` via
    /// `SoftwareBitmap::CreateCopyFromBuffer` (synchronous — no PNG
    /// encode/decode round-trip). The OCR engine is created synchronously
    /// via `TryCreateFromUserProfileLanguages`. Only `RecognizeAsync` is
    /// async; we block on it by polling `Status()` until it completes (the
    /// capture loop thread is MTA, so this is safe).
    pub fn recognize_text(bgra: &[u8], width: u32, height: u32) -> Option<String> {
        if width == 0 || height == 0 || bgra.is_empty() {
            return None;
        }

        // Ensure COM is initialized on this thread (MTA). Idempotent — if
        // already initialized (e.g. by the UI Automation module), this is a
        // no-op. RPC_E_CHANGED_MODE is ignored (we degrade to None).
        let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };

        // Create a SoftwareBitmap directly from the BGRA pixel data.
        let bitmap = create_software_bitmap(bgra, width, height)?;

        // Create the OCR engine for the user's preferred languages
        // (synchronous in windows-rs 0.62).
        let engine = OcrEngine::TryCreateFromUserProfileLanguages().ok()?;

        // Run recognition (async — block via Status() polling).
        let op = engine.RecognizeAsync(&bitmap).ok()?;
        let result = block_async(op)?;

        // Join recognized lines with newlines.
        let lines = result.Lines().ok()?;
        let mut text_parts: Vec<String> = Vec::new();
        let count = lines.Size().ok()?;
        for i in 0..count {
            if let Ok(line) = lines.GetAt(i) {
                if let Ok(text) = line.Text() {
                    text_parts.push(text.to_string());
                }
            }
        }
        if text_parts.is_empty() {
            return None;
        }
        let mut joined = text_parts.join("\n");
        if joined.len() > crate::events::MAX_TEXT_LENGTH {
            joined.truncate(crate::events::MAX_TEXT_LENGTH);
        }
        Some(joined)
    }

    /// Creates a `SoftwareBitmap` from raw BGRA pixels via
    /// `SoftwareBitmap::CreateCopyFromBuffer`. Uses `IBufferByteAccess` to
    /// write the pixel bytes into a WinRT `Buffer`. Returns `None` on any
    /// WinRT/COM failure.
    fn create_software_bitmap(bgra: &[u8], width: u32, height: u32) -> Option<SoftwareBitmap> {
        unsafe {
            let buffer = Buffer::Create(bgra.len() as u32).ok()?;
            // Write the BGRA data into the buffer via IBufferByteAccess.
            let byte_access: IBufferByteAccess = buffer.cast().ok()?;
            let ptr = byte_access.Buffer().ok()?;
            if ptr.is_null() {
                return None;
            }
            std::ptr::copy_nonoverlapping(bgra.as_ptr(), ptr, bgra.len());
            // Set the logical length (Buffer::Create sets capacity, not length).
            buffer.SetLength(bgra.len() as u32).ok()?;
            // Create the SoftwareBitmap from the buffer (Bgra8 matches scap's
            // BGRA frame format).
            SoftwareBitmap::CreateCopyFromBuffer(
                &buffer,
                BitmapPixelFormat::Bgra8,
                width as i32,
                height as i32,
            )
            .ok()
        }
    }

    /// Blocks on a WinRT `IAsyncOperation<T>` by polling `Status()` until it
    /// completes, then returns `GetResults()`. Returns `None` on error or
    /// cancellation. The capture loop thread is MTA (COM initialized with
    /// COINIT_MULTITHREADED), so polling is safe.
    fn block_async<T: Interface + RuntimeType>(op: IAsyncOperation<T>) -> Option<T> {
        loop {
            let status = op.Status().ok()?;
            match status {
                AsyncStatus::Completed => return op.GetResults().ok(),
                AsyncStatus::Error | AsyncStatus::Canceled => return None,
                _ => std::thread::sleep(Duration::from_millis(5)),
            }
        }
    }

    /// Returns true if the given process name (used as `bundle_id` on
    /// Windows) is a known terminal or terminal-hosting editor. Terminals
    /// don't expose text via UI Automation, so OCR is always used for them
    /// — same rule as the macOS `is_terminal_bundle_id`.
    pub fn is_terminal_bundle_id(bundle_id: Option<&str>) -> bool {
        match bundle_id {
            Some(name) => {
                const TERMINAL_NAMES: &[&str] = &[
                    "cmd",             // Command Prompt
                    "conhost",         // Console Host
                    "WindowsTerminal", // Windows Terminal
                    "powershell",      // Windows PowerShell
                    "pwsh",            // PowerShell Core
                    "bash",            // WSL / Git Bash
                    "wsl",             // WSL host
                    "Code",            // VS Code
                    "Code - Insiders", // VS Code Insiders
                    "devenv",          // Visual Studio
                    "idea64",          // IntelliJ IDEA (64-bit)
                    "pycharm64",       // PyCharm (64-bit)
                    "rustrover64",     // RustRover (64-bit)
                ];
                TERMINAL_NAMES.contains(&name)
            }
            None => false,
        }
    }
}

#[cfg(target_os = "windows")]
pub use platform::{is_terminal_bundle_id, recognize_text};

// -- Stubs for other platforms (neither macOS nor Windows) --

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod stubs {
    pub fn recognize_text(_bgra: &[u8], _width: u32, _height: u32) -> Option<String> {
        None
    }
    pub fn is_terminal_bundle_id(_bundle_id: Option<&str>) -> bool {
        false
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub use stubs::*;

#[cfg(test)]
mod tests {
    use super::*;

    // macOS-specific terminal bundle IDs. gated to macOS because the macOS
    // is_terminal_bundle_id checks Apple bundle identifiers.
    #[cfg(target_os = "macos")]
    #[test]
    fn is_terminal_app_recognizes_known_terminals() {
        assert!(is_terminal_bundle_id(Some("com.apple.Terminal")));
        assert!(is_terminal_bundle_id(Some("com.googlecode.iterm2")));
        assert!(is_terminal_bundle_id(Some("com.todesktop.682d041e3c80b9da")));
    }

    // Windows-specific terminal process names. gated to Windows because the
    // Windows is_terminal_bundle_id checks process names (used as bundle_id
    // on Windows, which has no bundle identifiers).
    #[cfg(target_os = "windows")]
    #[test]
    fn is_terminal_app_recognizes_known_terminals() {
        assert!(is_terminal_bundle_id(Some("cmd")));
        assert!(is_terminal_bundle_id(Some("conhost")));
        assert!(is_terminal_bundle_id(Some("WindowsTerminal")));
        assert!(is_terminal_bundle_id(Some("powershell")));
        assert!(is_terminal_bundle_id(Some("pwsh")));
    }

    // macOS-specific non-terminal bundle IDs (gated to macOS so the assertions
    // are meaningful on the platform whose is_terminal_bundle_id checks them).
    #[cfg(target_os = "macos")]
    #[test]
    fn is_terminal_app_rejects_non_terminals() {
        assert!(!is_terminal_bundle_id(Some("com.apple.Safari")));
        assert!(!is_terminal_bundle_id(Some("com.google.Chrome")));
    }

    // Windows-specific non-terminal process names (gated to Windows).
    #[cfg(target_os = "windows")]
    #[test]
    fn is_terminal_app_rejects_non_terminals() {
        assert!(!is_terminal_bundle_id(Some("explorer")));
        assert!(!is_terminal_bundle_id(Some("notepad")));
        assert!(!is_terminal_bundle_id(Some("chrome")));
    }

    #[test]
    fn is_terminal_app_handles_none() {
        assert!(!is_terminal_bundle_id(None));
    }

    // -- crop_bgra_to_window tests (cross-platform, no cfg gate) --

    #[test]
    fn crop_returns_none_for_empty_rect() {
        let buf = vec![0u8; 4 * 4 * 4]; // 4x4 BGRA
        let rect = Rect { x: 0, y: 0, width: 0, height: 0 };
        assert!(crop_bgra_to_window(&buf, 4, 4, rect).is_none());
    }

    #[test]
    fn crop_returns_none_for_zero_frame() {
        let rect = Rect { x: 0, y: 0, width: 10, height: 10 };
        assert!(crop_bgra_to_window(&[], 0, 0, rect).is_none());
    }

    #[test]
    fn crop_returns_none_for_out_of_bounds_rect() {
        let buf = vec![0u8; 4 * 4 * 4];
        let rect = Rect { x: 10, y: 10, width: 5, height: 5 };
        assert!(crop_bgra_to_window(&buf, 4, 4, rect).is_none());
    }

    #[test]
    fn crop_exact_full_frame() {
        // 4x4 frame, fill with a known pattern.
        let mut buf = vec![0u8; 4 * 4 * 4];
        for i in 0..buf.len() {
            buf[i] = (i % 256) as u8;
        }
        let rect = Rect { x: 0, y: 0, width: 4, height: 4 };
        let (out, w, h) = crop_bgra_to_window(&buf, 4, 4, rect).unwrap();
        assert_eq!((w, h), (4, 4));
        assert_eq!(out, buf);
    }

    #[test]
    fn crop_subregion_preserves_pixels() {
        // 4x4 frame. Each pixel is (v, v, v, 255) where v = x + y*4.
        let mut buf = vec![0u8; 4 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let idx = ((y * 4 + x) * 4) as usize;
                let val = (x + y * 4) as u8;
                buf[idx] = val;
                buf[idx + 1] = val;
                buf[idx + 2] = val;
                buf[idx + 3] = 255;
            }
        }
        // Crop to x=1..3, y=1..3 (2x2).
        let rect = Rect { x: 1, y: 1, width: 2, height: 2 };
        let (out, w, h) = crop_bgra_to_window(&buf, 4, 4, rect).unwrap();
        assert_eq!((w, h), (2, 2));
        // Row y=1: x=1 -> val 5, x=2 -> val 6
        // Row y=2: x=1 -> val 9, x=2 -> val 10
        let expected: Vec<u8> = vec![
            5, 5, 5, 255, 6, 6, 6, 255,
            9, 9, 9, 255, 10, 10, 10, 255,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn crop_clamps_rect_to_frame_bounds() {
        // 4x4 frame. Rect extends beyond frame: x=2, y=2, w=10, h=10.
        // Should clamp to x=2..4, y=2..4 (2x2).
        let buf = vec![1u8; 4 * 4 * 4];
        let rect = Rect { x: 2, y: 2, width: 10, height: 10 };
        let (out, w, h) = crop_bgra_to_window(&buf, 4, 4, rect).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(out.len(), 2 * 2 * 4);
        assert!(out.iter().all(|&b| b == 1));
    }

    #[test]
    fn crop_returns_none_for_short_buffer() {
        // Claim a 4x4 frame but provide only 1 byte.
        let rect = Rect { x: 0, y: 0, width: 2, height: 2 };
        assert!(crop_bgra_to_window(&[0u8], 4, 4, rect).is_none());
    }
}
