//! cyberalfred-capture -- a Rust screen capture agent for the mentor app.
//!
//! Session-bound CLI mirroring the native_mac Swift helper (the quality bar):
//! the Python provider launches `cyberalfred-capture start --session <id>
//! --output <dir>` and reads newline-delimited JSON events from stdout (also
//! appended to `<output>/<session>/events.jsonl`). It stops the session with
//! `cyberalfred-capture stop --session <id>` (sentinel file) or SIGINT (Ctrl+C),
//! and checks permissions with `cyberalfred-capture check`.
//!
//! This is a drop-in replacement for the Swift `mentor-capture` helper:
//! same event schema (Models.swift), same evidence quality (Accessibility
//! text + Vision OCR + event-driven app changes + self-capture exclusion +
//! coarse input capture). See native_capture/README.md and
//! app/services/native_common.py for the integration contract.

mod accessibility;
mod app_observer;
mod capture;
mod change_detection;
mod events;
mod input;
mod ocr;
mod self_capture;
mod status_bar;
mod stop;

// Windows-only module: Win32 foreground-window tracking (GetForegroundWindow,
// GetWindowText, process name). cfg-gated so it's absent on macOS.
#[cfg(target_os = "windows")]
mod foreground_window;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capture::{CapturedFrame, ScreenCapture};
use events::{CaptureEvent, EventWriter};

/// Target capture rate (scap throttles to this via the OS capture API).
const TARGET_FPS: u32 = 1;
/// Minimum time between two retained screenshots (matches native_mac).
const MIN_RETAINED_INTERVAL: Duration = Duration::from_secs(events::MIN_RETAINED_INTERVAL_SECS);
/// Force a retained screenshot at least this often (matches native_mac).
const SAFETY_CHECKPOINT: Duration = Duration::from_secs(events::SAFETY_CHECKPOINT_SECS);
/// How often aggregated key/scroll counts are flushed as events.
const KEY_FLUSH_INTERVAL: Duration = Duration::from_millis(1500);

// -- Platform-specific labels (cfg-gated so the macOS values are unchanged) --

/// The `text_source` label for AX-equivalent text. macOS uses
/// "accessibility"; Windows uses "ui_automation" (matching the C# helper).
#[cfg(target_os = "macos")]
const AX_TEXT_SOURCE: &str = "accessibility";
#[cfg(target_os = "windows")]
const AX_TEXT_SOURCE: &str = "ui_automation";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const AX_TEXT_SOURCE: &str = "accessibility";

/// Diagnostic label for the AX-equivalent permission/capability.
#[cfg(target_os = "macos")]
const AX_NAME: &str = "Accessibility";
#[cfg(target_os = "windows")]
const AX_NAME: &str = "UI Automation";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const AX_NAME: &str = "Accessibility";

/// Diagnostic label for the screen-capture permission/capability.
#[cfg(target_os = "macos")]
const SCREEN_NAME: &str = "Screen Recording";
#[cfg(target_os = "windows")]
const SCREEN_NAME: &str = "Screen Capture";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const SCREEN_NAME: &str = "Screen Recording";

/// Diagnostic label for the input-capture capability.
#[cfg(target_os = "macos")]
const INPUT_NAME: &str = "Input Monitoring";
#[cfg(target_os = "windows")]
const INPUT_NAME: &str = "Input Hooks";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const INPUT_NAME: &str = "Input Monitoring";

/// Error message when the input observer can't start.
#[cfg(target_os = "macos")]
const INPUT_ERROR_MSG: &str = "event tap could not be created (permission likely missing)";
#[cfg(target_os = "windows")]
const INPUT_ERROR_MSG: &str = "low-level hooks could not be installed";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const INPUT_ERROR_MSG: &str = "event tap could not be created (permission likely missing)";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = run(&args);
    std::process::exit(code);
}

fn run(args: &[String]) -> i32 {
    if args.len() < 2 {
        print_usage();
        return 1;
    }
    match args[1].as_str() {
        "start" => run_start(&args[2..]),
        "check" => run_check(),
        "stop" => run_stop(&args[2..]),
        "-h" | "--help" | "help" => {
            print_usage();
            0
        }
        other => {
            eprintln!("cyberalfred-capture: unknown command '{other}'");
            print_usage();
            1
        }
    }
}

fn print_usage() {
    eprintln!("Usage: cyberalfred-capture <start|stop|check> [options]");
    eprintln!("  start --session SESSION_ID --output OUTPUT_DIRECTORY");
    eprintln!("  stop  --session SESSION_ID");
    eprintln!("  check");
}

// -- start ----------------------------------------------------------------

struct StartArgs {
    session: String,
    output: PathBuf,
}

fn parse_start_args(args: &[String]) -> Option<StartArgs> {
    let mut session: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--session" => {
                i += 1;
                session = args.get(i).cloned();
            }
            "--output" => {
                i += 1;
                output = args.get(i).map(PathBuf::from);
            }
            other => {
                eprintln!("cyberalfred-capture: unknown argument '{other}'");
            }
        }
        i += 1;
    }
    Some(StartArgs {
        session: session?,
        output: output?,
    })
}

fn run_start(args: &[String]) -> i32 {
    let parsed = match parse_start_args(args) {
        Some(a) => a,
        None => {
            eprintln!(
                "Usage: cyberalfred-capture start --session SESSION_ID --output OUTPUT_DIRECTORY"
            );
            return 1;
        }
    };

    let session_dir = parsed.output.join(&parsed.session);
    let frames_dir = session_dir.join("frames");
    if let Err(err) = fs::create_dir_all(&frames_dir) {
        eprintln!("cyberalfred-capture: could not create output directory: {err}");
        return 1;
    }

    let mut writer = match EventWriter::new(&session_dir) {
        Ok(w) => w,
        Err(err) => {
            eprintln!("cyberalfred-capture: {err}");
            return 1;
        }
    };

    let running = Arc::new(AtomicBool::new(true));
    let stop_guard = match stop::install(&parsed.session, running.clone()) {
        Ok(g) => g,
        Err(err) => {
            writer.emit(&CaptureEvent::error("startup", &err));
            return 1;
        }
    };

    // Screen capture: degrade gracefully (no screenshots) if permission missing.
    let mut capture: Option<ScreenCapture> = match ScreenCapture::start(TARGET_FPS) {
        Ok(c) => Some(c),
        Err(err) => {
            writer.emit(&CaptureEvent::error("capture", &err));
            eprintln!("cyberalfred-capture: screen capture unavailable: {err}");
            None
        }
    };
    let screen_available = capture.as_ref().map(|c| c.is_running()).unwrap_or(false);

    writer.emit(&CaptureEvent::session_start());
    eprintln!(
        "cyberalfred-capture: session '{}' -> {}",
        parsed.session,
        session_dir.display()
    );
    eprintln!(
        "cyberalfred-capture: run 'cyberalfred-capture stop --session {}' or press Ctrl+C to stop",
        parsed.session
    );

    // Request permissions (prompt once at start).
    #[cfg(target_os = "macos")]
    {
        accessibility::request_permission_prompt_if_needed();
    }

    // Check which permissions are available (degraded mode, not fatal).
    let ax_available = accessibility::check_permission();
    if !ax_available {
        eprintln!("cyberalfred-capture: {AX_NAME} unavailable: window titles and AX text unavailable.");
    }
    if !screen_available {
        eprintln!("cyberalfred-capture: {SCREEN_NAME} unavailable: screenshots and OCR disabled.");
    }

    // Self-capture exclusion: record the launcher context at start.
    let own_pid = std::process::id() as i32;
    let launcher_ctx = capture_launcher_context(own_pid, ax_available);
    if let Some(ref ctx) = launcher_ctx {
        eprintln!(
            "cyberalfred-capture: excluding launcher window from text evidence ({}: {})",
            ctx.bundle_id.as_deref().unwrap_or("?"),
            ctx.window_title.as_deref().unwrap_or("untitled")
        );
    }

    // Start the input observer (event tap on the main run loop).
    let mut input_handle = start_input(&mut writer);

    // State for the capture loop.
    let mut frame_counter = highest_existing_frame_number(&frames_dir);
    let mut last_retained_grid: Option<Vec<u8>> = None;
    let mut last_retained_at: Option<Instant> = None;
    let mut last_app: Option<String> = None;
    let mut last_bundle_id: Option<String> = None;
    let mut last_window_title: Option<String> = None;
    let mut last_window_element: Option<accessibility::AxElement> = None;
    let mut last_text: Option<String> = None;
    let mut last_key_flush = Instant::now();

    // Start the app-change observer (owned by the loop so it stays alive).
    let (app_observer, app_rx) = match app_observer::AppObserver::start() {
        Some(pair) => (Some(pair.0), Some(pair.1)),
        None => (None, None),
    };

    // Show a "● Recording" indicator in the macOS menu bar for the session.
    let _status_bar = status_bar::StatusBarGuard::new();

    // Run the capture loop.
    let result = capture_loop(
        &running,
        &parsed.session,
        &mut capture,
        &mut writer,
        &frames_dir,
        &mut frame_counter,
        &mut last_retained_grid,
        &mut last_retained_at,
        &mut last_app,
        &mut last_bundle_id,
        &mut last_window_title,
        &mut last_window_element,
        &mut last_text,
        &mut last_key_flush,
        &launcher_ctx,
        own_pid,
        ax_available,
        screen_available,
        input_handle.as_ref(),
        app_rx.as_ref(),
    );

    // Cleanup: stop input observer, flush remaining counts, stop capture.
    if let Some((ref mut observer, ref handle)) = input_handle {
        observer.stop();
        flush_input_counts(&handle.counts, &mut writer, &last_app, &last_bundle_id, &last_window_title);
    }
    if let Some(ref mut c) = capture {
        c.stop();
    }
    drop(app_observer); // remove notification registration
    writer.emit(&CaptureEvent::session_stop());
    drop(stop_guard);
    eprintln!("cyberalfred-capture: stopped");
    result
}

/// Records the launcher context: the frontmost app's bundle_id and focused
/// window AXUIElement identity at start time, so the launcher's own output
/// never becomes learner evidence. Mirrors `CaptureManager.captureLauncherContext`.
/// Runs inside objc2::exception::catch so ObjC exceptions can't abort.
#[cfg(target_os = "macos")]
fn capture_launcher_context(own_pid: i32, ax_available: bool) -> Option<LauncherContext> {
    use objc2_app_kit::NSWorkspace;

    objc2::exception::catch(|| {
        let workspace = NSWorkspace::sharedWorkspace();
        let frontmost = workspace.frontmostApplication()?;
        let pid = frontmost.processIdentifier();
        if pid == own_pid {
            return None;
        }

        let bundle_id = frontmost.bundleIdentifier().map(|s| s.to_string());
        let app_name = frontmost.localizedName().map(|s| s.to_string());

        let (window_element, window_title) = if ax_available {
            let element = accessibility::focused_window_element(pid);
            let title = element.as_ref().and_then(|e| accessibility::window_title(e));
            (element, title)
        } else {
            (None, None)
        };

        Some(LauncherContext {
            bundle_id,
            app_name,
            window_title,
            window_element,
        })
    })
    .ok()
    .flatten()
}

// -- Windows: records the foreground HWND + process name + title at start --

#[cfg(target_os = "windows")]
fn capture_launcher_context(_own_pid: i32, ax_available: bool) -> Option<LauncherContext> {
    let fg = foreground_window::try_get()?;
    // Skip if the foreground process is our own (unlikely for a console
    // app, but matches the macOS own-PID skip).
    if fg.pid as i32 == _own_pid {
        return None;
    }

    let bundle_id = Some(fg.app_name.clone());
    let app_name = Some(fg.app_name.clone());

    let (window_element, window_title) = if ax_available {
        let element = accessibility::AxElement::from_hwnd(fg.hwnd);
        let title = element.as_ref().and_then(|e| accessibility::window_title(e));
        (element, title)
    } else {
        (None, None)
    };

    // Use the foreground window's title as a fallback if AX couldn't read it.
    let window_title = window_title.or_else(|| {
        if fg.window_title.is_empty() { None } else { Some(fg.window_title.clone()) }
    });

    Some(LauncherContext {
        bundle_id,
        app_name,
        window_title,
        window_element,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn capture_launcher_context(_own_pid: i32, _ax_available: bool) -> Option<LauncherContext> {
    None
}

/// The recorded launcher context for self-capture exclusion.
struct LauncherContext {
    bundle_id: Option<String>,
    #[allow(dead_code)]
    app_name: Option<String>,
    window_title: Option<String>,
    window_element: Option<accessibility::AxElement>,
}

/// Starts the input observer. Returns the observer + handle, or None if
/// the event tap couldn't be created (permission missing). Emits an error
/// event in that case but does not crash.
fn start_input(writer: &mut EventWriter) -> Option<(input::InputObserver, input::InputHandle)> {
    match input::InputObserver::start() {
        Some(pair) => Some(pair),
        None => {
            eprintln!("cyberalfred-capture: {INPUT_NAME} unavailable: mouse/keyboard activity capture disabled.");
            writer.emit(&CaptureEvent::error(
                "input",
                INPUT_ERROR_MSG,
            ));
            None
        }
    }
}

/// The main capture loop. On macOS, pumps the main CFRunLoop between ticks
/// so the event tap and NSWorkspace notifications fire.
#[allow(clippy::too_many_arguments)]
fn capture_loop(
    running: &Arc<AtomicBool>,
    session_id: &str,
    capture: &mut Option<ScreenCapture>,
    writer: &mut EventWriter,
    frames_dir: &Path,
    frame_counter: &mut u32,
    last_retained_grid: &mut Option<Vec<u8>>,
    last_retained_at: &mut Option<Instant>,
    last_app: &mut Option<String>,
    last_bundle_id: &mut Option<String>,
    last_window_title: &mut Option<String>,
    last_window_element: &mut Option<accessibility::AxElement>,
    last_text: &mut Option<String>,
    last_key_flush: &mut Instant,
    launcher_ctx: &Option<LauncherContext>,
    own_pid: i32,
    ax_available: bool,
    screen_available: bool,
    input_handle: Option<&(input::InputObserver, input::InputHandle)>,
    app_rx: Option<&std::sync::mpsc::Receiver<app_observer::AppChangeEvent>>,
) -> i32 {
    while running.load(Ordering::SeqCst) {
        // Cross-process stop check.
        if stop::stop_requested(session_id) {
            running.store(false, Ordering::SeqCst);
            break;
        }

        // Pump the main run loop briefly so event tap + notifications fire.
        #[cfg(target_os = "macos")]
        {
            pump_main_run_loop(Duration::from_millis(100));
        }

        // On Windows, hooks and the app-observer run on separate threads —
        // there's no run loop to pump. Sleep briefly to avoid busy-looping
        // between ticks (scap's next_frame blocks for the ~1 fps interval).
        #[cfg(target_os = "windows")]
        {
            std::thread::sleep(Duration::from_millis(100));
        }

        // Process app-change events from the notification observer.
        if let Some(rx) = app_rx {
            while let Ok(event) = rx.try_recv() {
                handle_app_change(
                    event,
                    writer,
                    capture,
                    frames_dir,
                    frame_counter,
                    last_retained_grid,
                    last_retained_at,
                    last_app,
                    last_bundle_id,
                    last_window_title,
                    last_window_element,
                    last_text,
                    launcher_ctx,
                    screen_available,
                );
            }
        }

        // Process click events from the input observer.
        if let Some((_, handle)) = input_handle {
            while let Ok(click) = handle.click_rx.try_recv() {
                let frame_path = attempt_triggered_screenshot(
                    capture,
                    frames_dir,
                    frame_counter,
                    last_retained_grid,
                    last_retained_at,
                    screen_available,
                );
                let mut ev = CaptureEvent::mouse_click(click.x, click.y, frame_path.as_deref());
                if let Some(app) = last_app.as_ref() {
                    ev = ev.with_app_name(app);
                }
                if let Some(bid) = last_bundle_id.as_ref() {
                    ev = ev.with_bundle_id(bid);
                }
                if let Some(title) = last_window_title.as_ref() {
                    ev = ev.with_window_title(title);
                }
                writer.emit(&ev);
            }
        }

        // Flush key/scroll counts on the aggregation window.
        if last_key_flush.elapsed() >= KEY_FLUSH_INTERVAL {
            if let Some((_, handle)) = input_handle {
                flush_input_counts(
                    &handle.counts,
                    writer,
                    last_app,
                    last_bundle_id,
                    last_window_title,
                );
            }
            *last_key_flush = Instant::now();
        }

        // --- One capture tick ---

        // Get the frontmost app PID (skip our own PID).
        let (pid, app_name, bundle_id) = frontmost_app_info(own_pid);
        if pid == 0 {
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }

        // App/window/text sampling (AX) — independent of frame retention.
        if ax_available {
            let window_element = accessibility::focused_window_element(pid);

            // Window title change detection.
            let current_title = window_element
                .as_ref()
                .and_then(|e| accessibility::window_title(e));

            if let Some(ref title) = current_title {
                if Some(title.as_str()) != last_window_title.as_deref() {
                    *last_window_title = Some(title.clone());
                    let ev = CaptureEvent::window_change(
                        last_app.as_deref(),
                        last_bundle_id.as_deref(),
                        title,
                    );
                    writer.emit(&ev);
                }
            }

            *last_window_element = window_element;

            // Check self-capture exclusion.
            let is_launcher = is_launcher_window(
                &bundle_id,
                &current_title,
                last_window_element,
                launcher_ctx,
            );

            if !is_launcher {
                // AX text extraction.
                if let Some(text) = accessibility::focused_text(pid) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() && Some(trimmed) != last_text.as_deref() {
                    *last_text = Some(trimmed.to_string());
                        let mut ev = CaptureEvent::text_observed(trimmed, AX_TEXT_SOURCE);
                        if let Some(app) = &app_name {
                            ev = ev.with_app_name(app);
                        }
                        if let Some(bid) = &bundle_id {
                            ev = ev.with_bundle_id(bid);
                        }
                        if let Some(title) = &current_title {
                            ev = ev.with_window_title(title);
                        }
                        writer.emit(&ev);
                    }
                }
            }
        }

        // Update app tracking (for app_change events from polling).
        if let Some(ref name) = app_name {
            if Some(name.as_str()) != last_app.as_deref() {
                *last_app = Some(name.clone());
                *last_bundle_id = bundle_id.clone();
                *last_text = None;
                let ev = CaptureEvent::app_change(name, bundle_id.as_deref());
                writer.emit(&ev);
            }
        }

        // Screen capture + change detection + retention.
        if !screen_available {
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }

        let frame = match capture.as_mut() {
            Some(c) => match c.next_frame() {
                Ok(frame) => frame,
                Err(err) => {
                    writer.emit(&CaptureEvent::error("capture", &err));
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            },
            None => {
                std::thread::sleep(Duration::from_secs(1));
                continue;
            }
        };

        let grid =
            change_detection::downscale_grayscale_bgra(&frame.bgra, frame.width, frame.height);
        let screen_diff = change_detection::difference(&grid, last_retained_grid.as_deref());

        let elapsed = last_retained_at
            .map(|t| t.elapsed())
            .unwrap_or(SAFETY_CHECKPOINT + MIN_RETAINED_INTERVAL);
        let safety_due = elapsed >= SAFETY_CHECKPOINT;
        let min_interval_due = elapsed >= MIN_RETAINED_INTERVAL;
        let first_frame = last_retained_at.is_none();

        let should_retain =
            first_frame || (screen_diff > change_detection::CHANGE_THRESHOLD && min_interval_due)
                || safety_due;

        if !should_retain {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        *frame_counter += 1;
        let file_name = format!("{frame_counter:06}.jpg");
        let file_path = frames_dir.join(&file_name);

        if let Err(err) = save_jpeg(&file_path, &frame, events::JPEG_QUALITY) {
            writer.emit(&CaptureEvent::error("capture", &format!("failed to save frame: {err}")));
            continue;
        }

        *last_retained_grid = Some(grid);
        *last_retained_at = Some(Instant::now());

        let relative_path = format!("frames/{file_name}");
        let mut ev = CaptureEvent::screen_change(&relative_path, screen_diff);
        if let Some(app) = &last_app {
            ev = ev.with_app_name(app);
        }
        if let Some(bid) = &last_bundle_id {
            ev = ev.with_bundle_id(bid);
        }
        if let Some(title) = &last_window_title {
            ev = ev.with_window_title(title);
        }
        writer.emit(&ev);

        // OCR fallback: only when AX text is empty (or terminal app), and
        // only on already-retained screenshots, and never on the launcher window.
        maybe_run_ocr(
            &frame,
            last_text,
            last_app,
            last_bundle_id,
            last_window_title,
            last_window_element,
            launcher_ctx,
            writer,
        );
    }

    0
}

/// Handles an app-change notification: emits an app_change event and triggers
/// a retained screenshot.
#[allow(clippy::too_many_arguments)]
fn handle_app_change(
    event: app_observer::AppChangeEvent,
    writer: &mut EventWriter,
    capture: &mut Option<ScreenCapture>,
    frames_dir: &Path,
    frame_counter: &mut u32,
    last_retained_grid: &mut Option<Vec<u8>>,
    last_retained_at: &mut Option<Instant>,
    last_app: &mut Option<String>,
    last_bundle_id: &mut Option<String>,
    last_window_title: &mut Option<String>,
    last_window_element: &mut Option<accessibility::AxElement>,
    last_text: &mut Option<String>,
    launcher_ctx: &Option<LauncherContext>,
    screen_available: bool,
) {
    // Update app tracking.
    *last_app = event.app_name.clone();
    *last_bundle_id = event.bundle_id.clone();
    *last_text = None;

    // Pre-seed with the launcher's known title when this app matches the launcher.
    if let Some(ctx) = launcher_ctx {
        if event.bundle_id == ctx.bundle_id {
            *last_window_title = ctx.window_title.clone();
            *last_window_element = ctx.window_element.clone();
        } else {
            *last_window_title = None;
            *last_window_element = None;
        }
    }

    // Emit app_change event.
    let ev = CaptureEvent::app_change(
        event.app_name.as_deref().unwrap_or("Unknown"),
        event.bundle_id.as_deref(),
    );
    writer.emit(&ev);

    // Trigger a screenshot.
    let _ = attempt_triggered_screenshot(
        capture,
        frames_dir,
        frame_counter,
        last_retained_grid,
        last_retained_at,
        screen_available,
    );
}

/// Attempts to retain a fresh frame immediately (triggered by click/app-change),
/// subject only to the minimum-interval rate limit. Returns the relative path
/// if a frame was saved.
fn attempt_triggered_screenshot(
    capture: &mut Option<ScreenCapture>,
    frames_dir: &Path,
    frame_counter: &mut u32,
    last_retained_grid: &mut Option<Vec<u8>>,
    last_retained_at: &mut Option<Instant>,
    screen_available: bool,
) -> Option<String> {
    if !screen_available {
        return None;
    }
    if let Some(t) = *last_retained_at {
        if t.elapsed() < MIN_RETAINED_INTERVAL {
            return None;
        }
    }

    let capturer = capture.as_mut()?;
    let frame = capturer.next_frame().ok()?;
    let grid =
        change_detection::downscale_grayscale_bgra(&frame.bgra, frame.width, frame.height);

    *frame_counter += 1;
    let file_name = format!("{frame_counter:06}.jpg");
    let file_path = frames_dir.join(&file_name);

    if save_jpeg(&file_path, &frame, events::JPEG_QUALITY).is_err() {
        return None;
    }

    *last_retained_grid = Some(grid);
    *last_retained_at = Some(Instant::now());

    Some(format!("frames/{file_name}"))
}

/// OCR fallback: runs Vision OCR on a retained frame when AX text is empty
/// (or the frontmost app is a terminal). Never OCRs the launcher window.
fn maybe_run_ocr(
    frame: &CapturedFrame,
    last_text: &Option<String>,
    last_app: &Option<String>,
    last_bundle_id: &Option<String>,
    last_window_title: &Option<String>,
    last_window_element: &Option<accessibility::AxElement>,
    launcher_ctx: &Option<LauncherContext>,
    writer: &mut EventWriter,
) {
    // Skip if this is the launcher window.
    let is_launcher = is_launcher_window(
        last_bundle_id,
        last_window_title,
        last_window_element,
        launcher_ctx,
    );
    if is_launcher {
        return;
    }

    // Skip if we already have AX text (unless terminal app).
    let is_terminal = ocr::is_terminal_bundle_id(last_bundle_id.as_deref());
    let has_text = last_text.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    if !is_terminal && has_text {
        return;
    }

    // Scope OCR to the foreground window's bounding rectangle. Both
    // platforms crop the BGRA frame to the window's screen rect before
    // running OCR, so only the foreground window is OCR'd instead of the
    // whole screen (which buries terminal output in taskbar/explorer
    // noise on Windows). If the window bounds are unavailable, fall back
    // to whole-frame OCR (graceful degradation, no crash).
    #[cfg(target_os = "macos")]
    let window_rect = last_window_element.as_ref().and_then(|e| e.window_bounds());

    #[cfg(target_os = "windows")]
    let window_rect = last_window_element
        .as_ref()
        .and_then(|e| crate::foreground_window::window_rect(e.hwnd()));

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let window_rect: Option<ocr::Rect> = None;

    let ocr_text = match window_rect {
        Some(rect) => {
            ocr::crop_bgra_to_window(&frame.bgra, frame.width, frame.height, rect)
                .and_then(|(cropped, w, h)| ocr::recognize_text(&cropped, w, h))
        }
        None => ocr::recognize_text(&frame.bgra, frame.width, frame.height),
    };

    if let Some(text) = ocr_text {
        let trimmed = text.trim();
        if trimmed.is_empty() || Some(trimmed) == last_text.as_deref() {
            return;
        }
        let mut ev = CaptureEvent::text_observed(trimmed, "ocr");
        if let Some(app) = last_app {
            ev = ev.with_app_name(app);
        }
        if let Some(bid) = last_bundle_id {
            ev = ev.with_bundle_id(bid);
        }
        if let Some(title) = last_window_title {
            ev = ev.with_window_title(title);
        }
        writer.emit(&ev);
    }
}

/// Checks if the current window is the launcher window (self-capture exclusion).
fn is_launcher_window(
    bundle_id: &Option<String>,
    window_title: &Option<String>,
    window_element: &Option<accessibility::AxElement>,
    launcher_ctx: &Option<LauncherContext>,
) -> bool {
    let Some(ctx) = launcher_ctx else {
        return false;
    };

    #[cfg(target_os = "macos")]
    {
        self_capture::is_launcher_window(
            bundle_id.as_deref(),
            window_title.as_deref(),
            window_element.as_ref(),
            ctx.bundle_id.as_deref(),
            ctx.window_title.as_deref(),
            ctx.window_element.as_ref(),
            |a, b| accessibility::elements_equal(a, b),
            |e| accessibility::parent_element(e),
            3,
        )
    }

    // Windows: HWND identity + owner-chain walk (GetWindow GW_OWNER) +
    // title fallback. Mirrors the macOS AX identity + parent walk.
    #[cfg(target_os = "windows")]
    {
        self_capture::is_launcher_window(
            bundle_id.as_deref(),
            window_title.as_deref(),
            window_element.as_ref(),
            ctx.bundle_id.as_deref(),
            ctx.window_title.as_deref(),
            ctx.window_element.as_ref(),
            |a, b| accessibility::elements_equal(a, b),
            |e| accessibility::parent_element(e),
            3,
        )
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        self_capture::is_launcher_window_by_title(
            bundle_id.as_deref(),
            window_title.as_deref(),
            ctx.bundle_id.as_deref(),
            ctx.window_title.as_deref(),
        )
    }
}

/// Flushes aggregated key/scroll counts as key_activity and scroll events.
fn flush_input_counts(
    counts: &Arc<Mutex<input::InputCounts>>,
    writer: &mut EventWriter,
    last_app: &Option<String>,
    last_bundle_id: &Option<String>,
    last_window_title: &Option<String>,
) {
    let mut counts = counts.lock().unwrap();
    if counts.scroll_count > 0 {
        let mut ev = CaptureEvent::scroll(counts.scroll_count);
        if let Some(app) = last_app {
            ev = ev.with_app_name(app);
        }
        if let Some(bid) = last_bundle_id {
            ev = ev.with_bundle_id(bid);
        }
        if let Some(title) = last_window_title {
            ev = ev.with_window_title(title);
        }
        writer.emit(&ev);
        counts.scroll_count = 0;
    }
    for (category, count) in counts.key_counts.drain() {
        let mut ev = CaptureEvent::key_activity(&category, count);
        if let Some(app) = last_app {
            ev = ev.with_app_name(app);
        }
        if let Some(bid) = last_bundle_id {
            ev = ev.with_bundle_id(bid);
        }
        if let Some(title) = last_window_title {
            ev = ev.with_window_title(title);
        }
        writer.emit(&ev);
    }
}

/// Returns the frontmost app's PID, app_name, and bundle_id.
/// Skips our own PID. Returns (0, None, None) if no suitable app is found.
/// Runs inside objc2::exception::catch so ObjC exceptions can't abort.
#[cfg(target_os = "macos")]
fn frontmost_app_info(own_pid: i32) -> (i32, Option<String>, Option<String>) {
    use objc2_app_kit::NSWorkspace;
    objc2::exception::catch(|| {
        let workspace = NSWorkspace::sharedWorkspace();
        let frontmost = match workspace.frontmostApplication() {
            Some(app) => app,
            None => return (0, None, None),
        };
        let pid = frontmost.processIdentifier();
        if pid == own_pid {
            return (0, None, None);
        }
        let app_name = frontmost.localizedName().map(|s| s.to_string());
        let bundle_id = frontmost.bundleIdentifier().map(|s| s.to_string());
        (pid, app_name, bundle_id)
    })
    .unwrap_or((0, None, None))
}

// -- Windows: foreground window -> PID + process name (as app_name + bundle_id) --

#[cfg(target_os = "windows")]
fn frontmost_app_info(own_pid: i32) -> (i32, Option<String>, Option<String>) {
    let fg = match foreground_window::try_get() {
        Some(fg) => fg,
        None => return (0, None, None),
    };
    if fg.pid as i32 == own_pid {
        return (0, None, None);
    }
    let app_name = Some(fg.app_name.clone());
    // On Windows the process name serves as the bundle_id.
    let bundle_id = Some(fg.app_name.clone());
    (fg.pid as i32, app_name, bundle_id)
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn frontmost_app_info(_own_pid: i32) -> (i32, Option<String>, Option<String>) {
    (0, None, None)
}

/// Pumps the main CFRunLoop for a short duration so the event tap and
/// NSWorkspace notifications fire.
#[cfg(target_os = "macos")]
fn pump_main_run_loop(duration: Duration) {
    extern "C" {
        fn CFRunLoopRunInMode(
            mode: *const std::ffi::c_void,
            duration: f64,
            return_after_source_handled: bool,
        ) -> i32;
        static kCFRunLoopDefaultMode: *const std::ffi::c_void;
    }
    unsafe {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, duration.as_secs_f64(), false);
    }
}

// -- check ----------------------------------------------------------------

/// Prints human-readable permission/capability lines to STDOUT and exits 0.
/// Reports all three permissions: Screen Recording, Accessibility, Input
/// Monitoring — matching native_mac's check output. The Python provider
/// parses these lines (lowercase prefix + "granted" substring).
///
/// Deliberately exits 0 regardless of permission status — missing permissions
/// are a degraded-but-running state, not a health failure.
fn run_check() -> i32 {
    // Screen capture (scap: ScreenCaptureKit on macOS, Windows.Graphics.Capture
    // on Windows). The label is platform-specific ("Screen Recording" on
    // macOS, "Screen Capture" on Windows) so the Python provider's
    // permission_status() parser matches the right prefix.
    let screen_granted = scap::is_supported() && scap::has_permission();
    if screen_granted {
        println!("{SCREEN_NAME}: granted");
    } else {
        println!("{SCREEN_NAME}: unavailable");
    }

    // AX-equivalent text capability (Accessibility on macOS, UI Automation
    // on Windows).
    #[cfg(target_os = "macos")]
    {
        if accessibility::check_permission() {
            println!("Accessibility: granted");
        } else {
            println!("Accessibility: unavailable");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if accessibility::check_permission() {
            println!("UI Automation: granted");
        } else {
            println!("UI Automation: unavailable");
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        println!("Accessibility: unavailable");
    }

    // Input capture capability (Input Monitoring on macOS, low-level hooks
    // on Windows). On Windows, hooks don't require explicit permission, so
    // this always reports granted. Not parsed by the Python provider on
    // Windows (it only looks for "screen capture" and "active window").
    #[cfg(target_os = "macos")]
    {
        if input::InputObserver::check_permission() {
            println!("Input Monitoring: granted");
        } else {
            println!("Input Monitoring: unavailable");
        }
    }
    #[cfg(target_os = "windows")]
    {
        // Low-level hooks don't require user-granted permission on Windows.
        println!("Input Hooks: granted");
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        println!("Input Monitoring: unavailable");
    }

    // Active Window (Windows-only). The Python provider's
    // permission_status() parses this line ("active window" prefix) on
    // Windows to report the foreground-window capability. The C# helper
    // prints "Foreground Window:" but the Python parser expects
    // "active window" — we use "Active Window:" to match the parser.
    #[cfg(target_os = "windows")]
    {
        if foreground_window::try_get().is_some() {
            println!("Active Window: granted");
        } else {
            println!("Active Window: unavailable");
        }
    }

    0
}

// -- stop -----------------------------------------------------------------

fn run_stop(args: &[String]) -> i32 {
    let mut session: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--session" {
            i += 1;
            session = args.get(i).cloned();
        } else {
            eprintln!("cyberalfred-capture: unknown argument '{}'", args[i]);
        }
        i += 1;
    }
    let session = match session {
        Some(s) => s,
        None => {
            eprintln!("Usage: cyberalfred-capture stop --session SESSION_ID");
            return 1;
        }
    };

    if stop::signal_stop(&session) {
        println!("stop: signaled capture for session '{session}'");
        0
    } else {
        eprintln!("stop: no active capture for session '{session}'");
        2
    }
}

// -- helpers --------------------------------------------------------------

fn highest_existing_frame_number(frames_dir: &Path) -> u32 {
    let entries = match fs::read_dir(frames_dir) {
        Ok(entries) => entries,
        Err(_) => return 0,
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| {
            name.strip_suffix(".jpg")
                .or_else(|| name.strip_suffix(".png"))
                .map(str::to_string)
        })
        .filter_map(|stem| stem.parse::<u32>().ok())
        .max()
        .unwrap_or(0)
}

fn save_jpeg(path: &Path, frame: &CapturedFrame, quality: u8) -> Result<(), String> {
    let mut rgba = frame.bgra.clone();
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.swap(0, 2); // BGRA -> RGBA
    }
    let img = image::RgbaImage::from_raw(frame.width, frame.height, rgba)
        .ok_or("failed to create image buffer")?;
    // Convert to RGB for JPEG (no alpha channel in JPEG).
    let rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = std::io::BufWriter::new(file);
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, quality);
    encoder
        .encode(&rgb, frame.width, frame.height, image::ColorType::Rgb8)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_args_prints_usage_and_exits_1() {
        assert_eq!(run(&args(&["cyberalfred-capture"])), 1);
    }

    #[test]
    fn help_exits_0() {
        assert_eq!(run(&args(&["cyberalfred-capture", "help"])), 0);
        assert_eq!(run(&args(&["cyberalfred-capture", "--help"])), 0);
        assert_eq!(run(&args(&["cyberalfred-capture", "-h"])), 0);
    }

    #[test]
    fn unknown_command_exits_1() {
        assert_eq!(run(&args(&["cyberalfred-capture", "frobnicate"])), 1);
    }

    #[test]
    fn stop_without_session_exits_1() {
        assert_eq!(run(&args(&["cyberalfred-capture", "stop"])), 1);
    }

    #[test]
    fn stop_for_nonexistent_session_exits_2() {
        let session = format!(
            "nonexistent-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        assert_eq!(
            run(&args(&["cyberalfred-capture", "stop", "--session", &session])),
            2
        );
    }

    #[test]
    fn check_exits_0() {
        assert_eq!(run(&args(&["cyberalfred-capture", "check"])), 0);
    }

    #[test]
    fn start_without_args_exits_1() {
        assert_eq!(run(&args(&["cyberalfred-capture", "start"])), 1);
        assert_eq!(
            run(&args(&["cyberalfred-capture", "start", "--session", "x"])),
            1,
            "start with --session but no --output must exit 1"
        );
    }

    #[test]
    fn highest_frame_number_handles_jpg_and_png() {
        let dir = std::env::temp_dir().join(format!(
            "frame-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("000001.jpg"), b"").unwrap();
        fs::write(dir.join("000003.png"), b"").unwrap();
        fs::write(dir.join("000002.jpg"), b"").unwrap();
        assert_eq!(highest_existing_frame_number(&dir), 3);
        fs::remove_dir_all(&dir).ok();
    }
}
