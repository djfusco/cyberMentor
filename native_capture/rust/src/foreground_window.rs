//! Win32 foreground-window tracking — the Windows analog of the macOS
//! CaptureManager's app/window observation. Mirrors `ForegroundWindow.cs`
//! in the C# helper: `GetForegroundWindow` -> HWND, `GetWindowText` ->
//! title, `GetWindowThreadProcessId` -> pid -> process name (used as both
//! `app_name` and `bundle_id`, since Windows doesn't have bundle
//! identifiers).
//!
//! All calls are best-effort: a failure (no foreground window, process
//! lookup failure) returns `None` or a fallback value — never crashes.
//! This is the identity source for the self-capture exclusion (the HWND
//! recorded at `start` time) and for the app-change polling thread.

#![cfg(target_os = "windows")]

use windows::Win32::Foundation::{CloseHandle, RECT};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_FORMAT,
    PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowRect,
    GetWindowThreadProcessId,
};
use windows::core::PWSTR;

/// Info about the current foreground window: its HWND (identity), the owning
/// process's name (used as both `app_name` and `bundle_id`), and the window
/// title. Mirrors the C# `ForegroundInfo` record struct.
#[derive(Debug, Clone)]
pub struct ForegroundInfo {
    /// The raw HWND value, used for identity comparison (self-capture
    /// exclusion) and passed to UI Automation's `element_from_handle`.
    pub hwnd: isize,
    /// The owning process's executable name without extension (e.g.
    /// "cmd", "WindowsTerminal", "chrome"). Used as `app_name`.
    pub app_name: String,
    /// The foreground window's title text (via `GetWindowTextW`).
    pub window_title: String,
    /// The owning process's PID.
    pub pid: u32,
}

/// Retrieves the foreground window's HWND, title, and owning process name.
/// Returns `None` if there is no foreground window. Mirrors
/// `ForegroundWindow.TryGet()` in the C# helper — never crashes; a process
/// name lookup failure falls back to "unknown".
pub fn try_get() -> Option<ForegroundInfo> {
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_invalid() {
            return None;
        }
        let hwnd_val = hwnd.0.expose_provenance() as isize;
        let title = window_title(hwnd);
        let pid = process_id(hwnd);
        let app = process_name(pid).unwrap_or_else(|| "unknown".to_string());
        Some(ForegroundInfo {
            hwnd: hwnd_val,
            app_name: app,
            window_title: title,
            pid,
        })
    }
}

/// Returns the foreground window's HWND as an `isize`, or 0 if there is no
/// foreground window. Convenience wrapper used by `check` and self-capture.
#[allow(dead_code)]
pub fn foreground_hwnd() -> isize {
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_invalid() {
            0
        } else {
            hwnd.0.expose_provenance() as isize
        }
    }
}

/// Returns the window's bounding rectangle in screen coordinates, via
/// `GetWindowRect`. Returns `None` if the call fails or the rect is empty
/// (after clamping negatives to 0). Used to scope OCR to the foreground
/// window instead of the whole screen, so terminal output isn't buried in
/// taskbar/explorer noise.
pub fn window_rect(hwnd: isize) -> Option<crate::ocr::Rect> {
    if hwnd == 0 {
        return None;
    }
    unsafe {
        let mut rect = RECT::default();
        GetWindowRect(HWND(hwnd as *mut std::ffi::c_void), &mut rect).ok()?;
        let left = rect.left.max(0) as u32;
        let top = rect.top.max(0) as u32;
        let right = rect.right.max(0) as u32;
        let bottom = rect.bottom.max(0) as u32;
        if right <= left || bottom <= top {
            return None;
        }
        Some(crate::ocr::Rect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        })
    }
}

/// Returns the process's executable name (without extension) for the given
/// PID. Public wrapper around the internal `process_name` — used by
/// `accessibility::bundle_id_for_pid` as the Windows analog of a bundle ID.
pub fn process_name_for_pid(pid: u32) -> Option<String> {
    process_name(pid)
}

/// Reads the window title via `GetWindowTextLengthW` + `GetWindowTextW`.
/// Returns an empty string if the window has no title (never crashes).
unsafe fn window_title(hwnd: HWND) -> String {
    let len = GetWindowTextLengthW(hwnd);
    if len <= 0 {
        return String::new();
    }
    let capacity = (len + 1) as usize;
    let mut buf = vec![0u16; capacity];
    let copied = GetWindowTextW(hwnd, &mut buf);
    if copied <= 0 {
        return String::new();
    }
    let len = copied as usize;
    // Truncate at the first NUL in case GetWindowTextW didn't fully fill.
    let end = buf[..len].iter().position(|&c| c == 0).unwrap_or(len);
    String::from_utf16_lossy(&buf[..end])
}

/// Returns the PID that owns the window, via `GetWindowThreadProcessId`.
unsafe fn process_id(hwnd: HWND) -> u32 {
    let mut pid: u32 = 0;
    let _thread_id = GetWindowThreadProcessId(hwnd, Some(&mut pid));
    pid
}

/// Returns the process's executable name (without extension) for the given
/// PID, via `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` +
/// `QueryFullProcessImageNameW`. Returns `None` if the process can't be
/// opened (e.g. an elevated process from a non-elevated capture). This is
/// the Windows analog of the C# helper's `Process.GetProcessById(pid).ProcessName`.
fn process_name(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let result = query_process_name(handle);
        let _ = CloseHandle(handle);
        result
    }
}

/// Queries the full process image path and extracts the file name without
/// extension. Returns `None` if the path can't be queried.
unsafe fn query_process_name(handle: windows::Win32::Foundation::HANDLE) -> Option<String> {
    let mut buf = vec![0u16; 1024];
    let mut size = buf.len() as u32;
    QueryFullProcessImageNameW(
        handle,
        PROCESS_NAME_FORMAT(0),
        PWSTR(buf.as_mut_ptr()),
        &mut size,
    )
    .ok()?;
    if size == 0 {
        return None;
    }
    let end = buf[..size as usize]
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(size as usize);
    let full_path = String::from_utf16_lossy(&buf[..end]);
    // Extract the file name (last path component) and strip the extension.
    let file_name = full_path.rsplit(['\\', '/']).next().unwrap_or(&full_path);
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name);
    Some(stem.to_string())
}
