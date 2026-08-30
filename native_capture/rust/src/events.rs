//! JSONL event schema emitted to stdout and appended to events.jsonl.
//!
//! One flat event struct with optional fields (snake_case keys), matching
//! the EXACT schema of native_mac's Models.swift — so the shared Python
//! translator `native_common._translate_event` consumes Rust events
//! unchanged. Every event is serialized as one JSON object per line,
//! written to stdout (so the Python provider reads it live) AND appended
//! to `<output>/<session>/events.jsonl` (so persisted evidence survives a
//! restart and can be replayed via `load_persisted_events`).
//!
//! Event types and their typical fields:
//! - `session_start` / `session_stop`: `type`, `timestamp`
//! - `app_change`: `type`, `timestamp`, `app_name`, `bundle_id`
//! - `window_change`: `type`, `timestamp`, `app_name`, `bundle_id`, `window_title`
//! - `text_observed`: `type`, `timestamp`, `app_name`, `bundle_id`, `window_title`,
//!   `text`, `text_source` ("accessibility" or "ocr")
//! - `screen_change`: `type`, `timestamp`, `app_name`, `bundle_id`, `window_title`,
//!   `frame_path`, `screen_difference` (0.0..=1.0)
//! - `mouse_click`: `type`, `timestamp`, `app_name`, `bundle_id`, `window_title`,
//!   `mouse_x`, `mouse_y`, `frame_path`
//! - `scroll`: `type`, `timestamp`, `app_name`, `bundle_id`, `window_title`, `key_count`
//! - `key_activity`: `type`, `timestamp`, `app_name`, `bundle_id`, `window_title`,
//!   `key_category`, `key_count`
//! - `error`: `type`, `timestamp`, `component`, `message`
//!
//! Optional fields are omitted entirely when `None` (skip_serializing_if), so a
//! `session_start` line is exactly `{"type":"session_start","timestamp":"..."}`.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde::Serialize;

// -- Event type constants (match EventType in Models.swift) --

pub const SESSION_START: &str = "session_start";
pub const SESSION_STOP: &str = "session_stop";
pub const APP_CHANGE: &str = "app_change";
pub const WINDOW_CHANGE: &str = "window_change";
pub const TEXT_OBSERVED: &str = "text_observed";
pub const SCREEN_CHANGE: &str = "screen_change";
pub const MOUSE_CLICK: &str = "mouse_click";
pub const SCROLL: &str = "scroll";
pub const KEY_ACTIVITY: &str = "key_activity";
pub const ERROR: &str = "error";

// -- Config constants (match CaptureConfig in Models.swift) --

/// Side length of the downsampled grayscale grid for change detection.
#[allow(dead_code)]
pub const DOWNSAMPLE_SIZE: usize = 32;
/// Mean pixel difference (0.0..=1.0) above which a frame is "meaningfully changed".
#[allow(dead_code)]
pub const CHANGE_THRESHOLD: f64 = 0.02;
/// Minimum time between two retained screenshots.
pub const MIN_RETAINED_INTERVAL_SECS: u64 = 2;
/// Force a retained screenshot at least this often even with no change.
pub const SAFETY_CHECKPOINT_SECS: u64 = 10;
/// How often aggregated key/scroll counters are flushed as events.
#[allow(dead_code)]
pub const KEY_AGGREGATION_WINDOW_SECS: u64 = 1;
/// Max chars of AX/OCR text retained per event.
pub const MAX_TEXT_LENGTH: usize = 10_000;
/// Bounds for the constrained Accessibility text walk.
pub const AX_MAX_DEPTH: usize = 5;
pub const AX_MAX_NODES: usize = 60;
/// JPEG compression quality for retained screenshots.
pub const JPEG_QUALITY: u8 = 60;

/// One flat event with optional fields, matching `CaptureEvent` in Models.swift.
/// All fields except `timestamp` and `event_type` are optional and omitted from
/// JSON when `None`.
#[derive(Serialize, Debug, PartialEq)]
pub struct CaptureEvent {
    pub timestamp: String,
    #[serde(rename = "type")]
    pub event_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mouse_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mouse_y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screen_difference: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl CaptureEvent {
    pub fn new(event_type: &'static str) -> Self {
        Self {
            timestamp: now_iso8601(),
            event_type,
            app_name: None,
            bundle_id: None,
            window_title: None,
            text: None,
            text_source: None,
            input_type: None,
            key_category: None,
            key_count: None,
            mouse_x: None,
            mouse_y: None,
            frame_path: None,
            screen_difference: None,
            component: None,
            message: None,
        }
    }

    pub fn session_start() -> Self {
        Self::new(SESSION_START)
    }

    pub fn session_stop() -> Self {
        Self::new(SESSION_STOP)
    }

    pub fn error(component: &str, message: &str) -> Self {
        Self::new(ERROR)
            .with_component(component)
            .with_message(message)
    }

    pub fn app_change(app_name: &str, bundle_id: Option<&str>) -> Self {
        let mut ev = Self::new(APP_CHANGE).with_app_name(app_name);
        if let Some(bid) = bundle_id {
            ev = ev.with_bundle_id(bid);
        }
        ev
    }

    pub fn window_change(app_name: Option<&str>, bundle_id: Option<&str>, window_title: &str) -> Self {
        let mut ev = Self::new(WINDOW_CHANGE).with_window_title(window_title);
        if let Some(app) = app_name {
            ev = ev.with_app_name(app);
        }
        if let Some(bid) = bundle_id {
            ev = ev.with_bundle_id(bid);
        }
        ev
    }

    pub fn text_observed(text: &str, text_source: &str) -> Self {
        Self::new(TEXT_OBSERVED)
            .with_text(text)
            .with_text_source(text_source)
    }

    pub fn screen_change(frame_path: &str, screen_difference: f64) -> Self {
        Self::new(SCREEN_CHANGE)
            .with_frame_path(frame_path)
            .with_screen_difference(screen_difference)
    }

    pub fn mouse_click(x: f64, y: f64, frame_path: Option<&str>) -> Self {
        let mut ev = Self::new(MOUSE_CLICK)
            .with_mouse_x(x)
            .with_mouse_y(y);
        if let Some(path) = frame_path {
            ev = ev.with_frame_path(path);
        }
        ev
    }

    pub fn scroll(count: u32) -> Self {
        Self::new(SCROLL).with_key_count(count)
    }

    pub fn key_activity(category: &str, count: u32) -> Self {
        Self::new(KEY_ACTIVITY)
            .with_key_category(category)
            .with_key_count(count)
    }

    // -- Builder helpers --

    pub fn with_app_name(mut self, app_name: &str) -> Self {
        self.app_name = Some(app_name.to_string());
        self
    }

    pub fn with_bundle_id(mut self, bundle_id: &str) -> Self {
        self.bundle_id = Some(bundle_id.to_string());
        self
    }

    pub fn with_window_title(mut self, window_title: &str) -> Self {
        self.window_title = Some(window_title.to_string());
        self
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text = Some(text.to_string());
        self
    }

    pub fn with_text_source(mut self, text_source: &str) -> Self {
        self.text_source = Some(text_source.to_string());
        self
    }

    pub fn with_frame_path(mut self, frame_path: &str) -> Self {
        self.frame_path = Some(frame_path.to_string());
        self
    }

    pub fn with_screen_difference(mut self, screen_difference: f64) -> Self {
        self.screen_difference = Some(screen_difference);
        self
    }

    pub fn with_mouse_x(mut self, x: f64) -> Self {
        self.mouse_x = Some(x);
        self
    }

    pub fn with_mouse_y(mut self, y: f64) -> Self {
        self.mouse_y = Some(y);
        self
    }

    pub fn with_key_category(mut self, category: &str) -> Self {
        self.key_category = Some(category.to_string());
        self
    }

    pub fn with_key_count(mut self, count: u32) -> Self {
        self.key_count = Some(count);
        self
    }

    pub fn with_component(mut self, component: &str) -> Self {
        self.component = Some(component.to_string());
        self
    }

    pub fn with_message(mut self, message: &str) -> Self {
        self.message = Some(message.to_string());
        self
    }
}

/// Writes events as newline-delimited JSON to stdout and, when a session
/// directory is provided, appends each line to `<session>/events.jsonl` so
/// persisted evidence can be replayed after the capture process exits.
pub struct EventWriter {
    file: Option<std::fs::File>,
}

impl EventWriter {
    /// Opens `<session_dir>/events.jsonl` for appending (created if absent).
    pub fn new(session_dir: &Path) -> Result<Self, String> {
        let path = session_dir.join("events.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| format!("could not open {}: {err}", path.display()))?;
        Ok(Self { file: Some(file) })
    }

    /// A writer that only writes to stdout (no events.jsonl). Used by `check`.
    #[allow(dead_code)]
    pub fn stdout_only() -> Self {
        Self { file: None }
    }

    /// Serializes `event` to one JSON line, prints it to stdout (flushed so a
    /// live consumer sees it immediately), and appends it to events.jsonl when
    /// a file is open. A file-write failure is logged to stderr but never
    /// panics -- losing the on-disk copy of one event must not kill a session.
    pub fn emit(&mut self, event: &CaptureEvent) {
        let line = match serde_json::to_string(event) {
            Ok(line) => line,
            Err(err) => {
                eprintln!("cyberalfred-capture: failed to serialize event: {err}");
                return;
            }
        };
        println!("{line}");
        let _ = std::io::stdout().flush();
        if let Some(file) = &mut self.file {
            if let Err(err) = writeln!(file, "{line}") {
                eprintln!("cyberalfred-capture: failed to write events.jsonl: {err}");
            }
            let _ = file.flush();
        }
    }
}

/// The result of comparing the current active app/window against the previous
/// sample. `app_change`/`window_change` are `Some` only when that dimension
/// changed AND the new value is known.
#[allow(dead_code)]
pub struct WindowDiff {
    pub app_change: Option<CaptureEvent>,
    pub window_change: Option<CaptureEvent>,
    pub app_changed: bool,
}

/// Diffs the current active app/window against the previous sample and
/// produces `app_change`/`window_change` events when they differ. The app
/// name and window title are diffed independently (mirroring native_mac).
/// A transient lookup failure (None) does not emit a change event and does
/// NOT reset the last-known value.
#[allow(dead_code)]
pub fn diff_window(
    last_app: &Option<String>,
    last_window: &Option<String>,
    current_app: &Option<String>,
    current_title: &Option<String>,
) -> WindowDiff {
    let app_changed = current_app.as_deref() != last_app.as_deref();
    let window_changed = current_title.as_deref() != last_window.as_deref();

    let app_change = if app_changed {
        current_app
            .as_ref()
            .map(|app| CaptureEvent::app_change(app, None))
    } else {
        None
    };
    let window_change = if window_changed {
        current_title
            .as_ref()
            .map(|title| CaptureEvent::window_change(current_app.as_deref(), None, title))
    } else {
        None
    };

    WindowDiff {
        app_change,
        window_change,
        app_changed,
    }
}

pub fn now_iso8601() -> String {
    // ISO-8601 UTC with millisecond precision and trailing "Z", matching
    // Models.swift's ISO8601DateFormatter with .withFractionalSeconds.
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn session_start_has_only_type_and_timestamp() {
        let ev = CaptureEvent::session_start();
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 2, "session_start must have exactly 2 fields: {json}");
        assert_eq!(obj["type"], "session_start");
        assert!(obj["timestamp"].is_string());
        assert!(!obj.contains_key("app_name"));
        assert!(!obj.contains_key("bundle_id"));
        assert!(!obj.contains_key("window_title"));
        assert!(!obj.contains_key("text"));
        assert!(!obj.contains_key("text_source"));
        assert!(!obj.contains_key("frame_path"));
        assert!(!obj.contains_key("screen_difference"));
        assert!(!obj.contains_key("component"));
        assert!(!obj.contains_key("message"));
    }

    #[test]
    fn session_stop_has_only_type_and_timestamp() {
        let ev = CaptureEvent::session_stop();
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 2);
        assert_eq!(obj["type"], "session_stop");
    }

    #[test]
    fn error_has_component_and_message() {
        let ev = CaptureEvent::error("input", "tap failed");
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 4);
        assert_eq!(obj["type"], "error");
        assert_eq!(obj["component"], "input");
        assert_eq!(obj["message"], "tap failed");
    }

    #[test]
    fn app_change_has_app_name_and_bundle_id() {
        let ev = CaptureEvent::app_change("Terminal", Some("com.apple.Terminal"));
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 4);
        assert_eq!(obj["type"], "app_change");
        assert_eq!(obj["app_name"], "Terminal");
        assert_eq!(obj["bundle_id"], "com.apple.Terminal");
    }

    #[test]
    fn app_change_omits_bundle_id_when_none() {
        let ev = CaptureEvent::app_change("Terminal", None);
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 3);
        assert_eq!(obj["type"], "app_change");
        assert_eq!(obj["app_name"], "Terminal");
        assert!(!obj.contains_key("bundle_id"));
    }

    #[test]
    fn window_change_has_all_context_fields() {
        let ev = CaptureEvent::window_change(Some("Terminal"), Some("com.apple.Terminal"), "zsh");
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 5);
        assert_eq!(obj["type"], "window_change");
        assert_eq!(obj["app_name"], "Terminal");
        assert_eq!(obj["bundle_id"], "com.apple.Terminal");
        assert_eq!(obj["window_title"], "zsh");
    }

    #[test]
    fn text_observed_has_text_and_text_source() {
        let ev = CaptureEvent::text_observed("hello world", "accessibility")
            .with_app_name("Safari")
            .with_bundle_id("com.apple.Safari")
            .with_window_title("Google");
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "text_observed");
        assert_eq!(obj["text"], "hello world");
        assert_eq!(obj["text_source"], "accessibility");
        assert_eq!(obj["app_name"], "Safari");
        assert_eq!(obj["window_title"], "Google");
    }

    #[test]
    fn screen_change_has_frame_path_and_screen_difference() {
        let ev = CaptureEvent::screen_change("frames/000001.jpg", 0.05)
            .with_app_name("Terminal");
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "screen_change");
        assert_eq!(obj["frame_path"], "frames/000001.jpg");
        assert_eq!(obj["screen_difference"], 0.05);
        assert_eq!(obj["app_name"], "Terminal");
        // Must NOT use old v1 fields
        assert!(!obj.contains_key("path"), "must use frame_path not path");
        assert!(!obj.contains_key("active_app"), "must use app_name not active_app");
    }

    #[test]
    fn mouse_click_has_coordinates_and_optional_frame_path() {
        let ev = CaptureEvent::mouse_click(100.0, 200.0, Some("frames/000002.jpg"));
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "mouse_click");
        assert_eq!(obj["mouse_x"], 100.0);
        assert_eq!(obj["mouse_y"], 200.0);
        assert_eq!(obj["frame_path"], "frames/000002.jpg");

        let ev2 = CaptureEvent::mouse_click(10.0, 20.0, None);
        let json2 = serde_json::to_string(&ev2).unwrap();
        let v2: serde_json::Value = serde_json::from_str(&json2).unwrap();
        assert!(!v2.as_object().unwrap().contains_key("frame_path"));
    }

    #[test]
    fn scroll_has_key_count() {
        let ev = CaptureEvent::scroll(5);
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "scroll");
        assert_eq!(obj["key_count"], 5);
    }

    #[test]
    fn key_activity_has_category_and_count() {
        let ev = CaptureEvent::key_activity("typing", 10);
        let json = serde_json::to_string(&ev).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "key_activity");
        assert_eq!(obj["key_category"], "typing");
        assert_eq!(obj["key_count"], 10);
    }

    #[test]
    fn timestamp_is_iso8601_utc_with_z() {
        let ev = CaptureEvent::session_start();
        assert!(ev.timestamp.ends_with('Z'));
        assert!(ev.timestamp.contains('T'));
        assert!(ev.timestamp.rsplit_once('.').is_some());
    }

    // -- diff_window tests --

    #[test]
    fn diff_first_sample_emits_both_when_known() {
        let d = diff_window(&None, &None, &Some("Terminal".into()), &Some("zsh".into()));
        assert!(d.app_change.is_some());
        assert!(d.window_change.is_some());
        assert!(d.app_changed);
    }

    #[test]
    fn diff_first_sample_none_emits_nothing() {
        let d = diff_window(&None, &None, &None, &None);
        assert!(d.app_change.is_none());
        assert!(d.window_change.is_none());
        assert!(!d.app_changed);
    }

    #[test]
    fn diff_no_change_emits_nothing() {
        let d = diff_window(
            &Some("Terminal".into()),
            &Some("zsh".into()),
            &Some("Terminal".into()),
            &Some("zsh".into()),
        );
        assert!(d.app_change.is_none());
        assert!(d.window_change.is_none());
        assert!(!d.app_changed);
    }

    #[test]
    fn diff_app_change_only() {
        let d = diff_window(
            &Some("Terminal".into()),
            &Some("zsh".into()),
            &Some("Safari".into()),
            &Some("zsh".into()),
        );
        assert!(d.app_change.is_some());
        assert!(d.window_change.is_none());
        assert!(d.app_changed);
    }

    // -- EventWriter tests --

    use std::sync::atomic::{AtomicU64, Ordering};

    // Monotonic counter so each temp_session_dir() call gets a unique directory
    // even when tests run in parallel and two calls land on the same nanosecond
    // (which caused event_writer_appends_to_* to share an events.jsonl and fail).
    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_session_dir() -> PathBuf {
        let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "cyberalfred-capture-test-{}-{}-{}",
            std::process::id(),
            id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn event_writer_appends_to_events_jsonl() {
        let dir = temp_session_dir();
        let mut writer = EventWriter::new(&dir).unwrap();
        writer.emit(&CaptureEvent::session_start());
        writer.emit(&CaptureEvent::screen_change("frames/000001.jpg", 0.05));
        writer.emit(&CaptureEvent::session_stop());
        drop(writer);

        let path = dir.join("events.jsonl");
        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3, "expected 3 lines:\n{content}");
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["type"], "session_start");
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["type"], "screen_change");
        assert_eq!(second["frame_path"], "frames/000001.jpg");
        let third: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(third["type"], "session_stop");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn event_writer_appends_to_existing_file() {
        let dir = temp_session_dir();
        let mut w1 = EventWriter::new(&dir).unwrap();
        w1.emit(&CaptureEvent::session_start());
        drop(w1);
        let mut w2 = EventWriter::new(&dir).unwrap();
        w2.emit(&CaptureEvent::session_stop());
        drop(w2);

        let content = fs::read_to_string(dir.join("events.jsonl")).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }
}
