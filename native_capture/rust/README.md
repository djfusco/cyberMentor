# cyberalfred-capture (Rust)

A Rust screen capture agent that is a **drop-in replacement** for the Swift
`mentor-capture` native_mac helper. It captures the same evidence at the same
quality bar: Accessibility text extraction (AX-first), Vision OCR fallback,
event-driven app-change detection, self-capture exclusion, and coarse input
capture (clicks/scroll/key categories) — all emitting the exact same event
schema as `native_capture/Sources/MentorCapture/Models.swift`.

The Python provider (`app/services/native_common.py`) consumes this binary's
event stream unchanged: `_translate_event` and `_synthesize_text` already
handle all native_mac event types, so the Rust binary is a true drop-in.

## What this program captures

- **Accessibility text** (AX-first): focused element value/title/selected text
  + focused window title + bounded walk of window children (depth 5, 60 nodes,
  10k char cap). Mirrors `AccessibilityObserver.swift`.
- **Vision OCR fallback**: `VNRecognizeTextRequest` (accurate, no language
  correction) on retained screenshots, ONLY when AX text is empty OR the
  frontmost app is a terminal. Mirrors `OCRService.swift`.
- **Screenshots**: one still frame per second via `scap` (ScreenCaptureKit),
  retained to disk (JPEG, 60% quality) when the screen meaningfully changes
  (32x32 grayscale diff, 2% threshold) or every 10s safety checkpoint, with a
  2s minimum between retained frames.
- **Event-driven app changes**: `NSWorkspaceDidActivateApplicationNotification`
  fires immediately on app switch (catches terminal switches a 1fps poll
  misses). Mirrors `CaptureManager.wireAppChanges`.
- **Self-capture exclusion**: the launcher window (where `start` was typed) and
  its spawned dialogs are excluded from text/OCR evidence by AXUIElement
  identity (CFEqual) + bounded ancestor walk. Mirrors
  `CaptureManager.isLauncherWindow`.
- **Input capture** (coarse, never raw characters):
  - Mouse clicks (left/right) with position + triggered screenshot
  - Scroll wheel (aggregated count, flushed every 1.5s)
  - Key-down categorized as: return/tab/escape/delete/arrow/command-shortcut/typing
    (aggregated count, flushed every 1.5s). NEVER decodes raw characters.
  Mirrors `InputObserver.swift`.
- A newline-delimited JSON (JSONL) event log on stdout AND on disk at
  `<output>/<session>/events.jsonl`.

## Required macOS permissions

This binary needs **three** macOS permissions (same as native_mac):

1. **Screen Recording** — System Settings → Privacy & Security → Screen
   Recording → enable for your terminal app. Required for screenshots and OCR.
2. **Accessibility** — System Settings → Privacy & Security → Accessibility →
   enable for your terminal app. Required for AX text extraction and window
   titles.
3. **Input Monitoring** — System Settings → Privacy & Security → Input
   Monitoring → enable for your terminal app. Required for mouse/keyboard
   activity capture.

macOS attributes these permissions to the "responsible" parent process when
launched from a terminal, so they are granted to your terminal app, not to
`cyberalfred-capture` itself, for an unsigned dev build. Missing permissions
are a **degraded-but-running** state, not a fatal error: the binary continues
with whatever capabilities are available (e.g. AX text works without Screen
Recording).

## Build

Requires a stable Rust toolchain. Tested with Rust 1.92.

```bash
cd native_capture/rust
cargo build
```

The binary is produced at `target/debug/cyberalfred-capture`.

Run the unit tests with:

```bash
cargo test
```

## CLI surface

```
cyberalfred-capture <start|stop|check> [options]
  start --session SESSION_ID --output OUTPUT_DIRECTORY
  stop  --session SESSION_ID
  check
```

### check

```bash
./target/debug/cyberalfred-capture check
```

Prints three permission lines to STDOUT and exits 0:

```text
Screen Recording: granted
Accessibility: granted
Input Monitoring: granted
```

or `unavailable` for any that are missing. The Python provider parses these
lines (lowercase prefix + "granted" substring) the same way `native_mac.py`
parses the Swift helper's output. `check` always exits 0 regardless of
permission status.

### start

```bash
./target/debug/cyberalfred-capture start --session SESSION_ID --output OUTPUT_DIRECTORY
```

Starts a capture session. Runs until stopped (see below), then shuts down
cleanly (stops the capturer, flushes remaining input counts, writes a final
`session_stop` event, closes `events.jsonl`, removes the pidfile/sentinel).

### stop

```bash
./target/debug/cyberalfred-capture stop --session SESSION_ID
```

Signals the running `start` process for that session to shut down gracefully.
Exits 0 if a capture was signaled, 2 if no capture is running. SIGINT (Ctrl+C)
also stops gracefully.

## Stopping a session

Three stop paths, all producing the same clean shutdown:

1. **`stop --session <id>`** — cross-process sentinel file signal.
2. **Ctrl+C (SIGINT)** — handled by the `ctrlc` crate.
3. **Normal process exit** — the `StopGuard` cleans up.

## Output layout

```text
OUTPUT_DIRECTORY/
└── SESSION_ID/
    ├── events.jsonl      # one JSON object per line, newline-delimited
    └── frames/
        ├── 000001.jpg    # retained screenshots (JPEG, 60% quality)
        └── ...
```

Every event is printed to STDOUT as newline-delimited JSON (the Python
provider reads this live) AND appended to `events.jsonl` on disk (persisted
evidence replayed via `load_persisted_events`). Human-readable diagnostics go
to STDERR only.

## Event schema

One flat JSON object per line, matching `Models.swift` exactly. All events
share `type` and `timestamp` (ISO-8601 UTC with milliseconds, trailing `Z`).
Optional fields are omitted entirely when absent (`skip_serializing_if`).

```json
{"type":"session_start","timestamp":"2026-08-28T18:55:42.123Z"}
{"type":"app_change","timestamp":"2026-08-28T18:55:43.456Z","app_name":"Terminal","bundle_id":"com.apple.Terminal"}
{"type":"window_change","timestamp":"2026-08-28T18:55:43.789Z","app_name":"Terminal","bundle_id":"com.apple.Terminal","window_title":"zsh"}
{"type":"text_observed","timestamp":"2026-08-28T18:55:44.012Z","app_name":"Terminal","bundle_id":"com.apple.Terminal","window_title":"zsh","text":"mkdir test_dir","text_source":"accessibility"}
{"type":"screen_change","timestamp":"2026-08-28T18:55:45.345Z","app_name":"Terminal","bundle_id":"com.apple.Terminal","window_title":"zsh","frame_path":"frames/000001.jpg","screen_difference":0.05}
{"type":"text_observed","timestamp":"2026-08-28T18:55:45.678Z","app_name":"Terminal","bundle_id":"com.apple.Terminal","window_title":"zsh","text":"mkdir test_dir cd test_dir","text_source":"ocr"}
{"type":"mouse_click","timestamp":"2026-08-28T18:55:46.901Z","app_name":"Safari","bundle_id":"com.apple.Safari","window_title":"Google","mouse_x":450.0,"mouse_y":320.0,"frame_path":"frames/000002.jpg"}
{"type":"scroll","timestamp":"2026-08-28T18:55:47.234Z","app_name":"Safari","bundle_id":"com.apple.Safari","window_title":"Google","key_count":5}
{"type":"key_activity","timestamp":"2026-08-28T18:55:47.567Z","app_name":"Terminal","bundle_id":"com.apple.Terminal","window_title":"zsh","key_category":"typing","key_count":12}
{"type":"error","timestamp":"2026-08-28T18:55:48.890Z","component":"input","message":"event tap could not be created (permission likely missing)"}
{"type":"session_stop","timestamp":"2026-08-28T18:55:49.123Z"}
```

Event types: `session_start`, `session_stop`, `app_change`, `window_change`,
`text_observed`, `screen_change`, `mouse_click`, `scroll`, `key_activity`,
`error`.

Fields: `timestamp`, `type`, `app_name`, `bundle_id`, `window_title`, `text`,
`text_source` (`"accessibility"` or `"ocr"`), `input_type`, `key_category`,
`key_count`, `mouse_x`, `mouse_y`, `frame_path`, `screen_difference`,
`component`, `message`.

## Crate dependencies (macOS)

- `scap` 0.0.8 — screen capture (ScreenCaptureKit)
- `objc2` 0.6 + `objc2-foundation` 0.3 + `objc2-app-kit` 0.3 — NSWorkspace,
  NSRunningApplication, NSNotificationCenter
- `objc2-vision` 0.3 — VNRecognizeTextRequest, VNImageRequestHandler
- `objc2-core-graphics` 0.3 — CGEvent (input tap)
- `objc2-core-foundation` 0.3 — CFRunLoop, CFMachPort
- `objc2-io-kit` 0.3 — IOHIDCheckAccess (Input Monitoring permission)
- `block2` 0.6 — Objective-C blocks for notification observer
- `serde`/`serde_json` — JSONL events
- `chrono` — ISO-8601 timestamps
- `ctrlc` — SIGINT handling
- `image` — JPEG encoding

AX text extraction uses raw C FFI to the ApplicationServices framework
(AXUIElement is a C API). CGEvent tap and IOKit IOHIDCheckAccess also use raw
FFI. All objc2 framework crates compile cleanly alongside `scap` 0.0.8 on
Rust 1.92.

## Design notes

- `src/accessibility.rs` — AX text extraction via raw C FFI. Owns AXUIElement
  refs with a manual Send+Sync wrapper, sets messaging timeouts, dedups cycles
  via pointer-address HashSet, bounded walk (depth 5, 60 nodes, 10k chars).
- `src/ocr.rs` — Vision OCR via objc2-vision. Creates a CGImage from raw BGRA
  bytes, runs VNRecognizeTextRequest (accurate, no language correction).
- `src/app_observer.rs` — NSWorkspace notification observer via objc2 block.
- `src/input.rs` — CGEventTap (listen-only) via raw FFI. Key categorization is
  pure and unit-testable.
- `src/self_capture.rs` — Pure decision logic for launcher window exclusion,
  generic over identity type T, fully unit-testable without a live AX tree.
- `src/events.rs` — Event schema matching Models.swift, EventWriter, diff_window.
- `src/capture.rs` — scap wrapper with degraded-mode support.
- `src/change_detection.rs` — 32x32 grayscale diff (unchanged from v1).
- `src/stop.rs` — Cross-process pidfile + sentinel stop mechanism.

## Privacy

Runs only when manually started for a session, and only for as long as it's
running. No LaunchAgent/daemon, no auto-start, no hidden execution. No
microphone/camera access, no audio, no continuous video. No network calls of
any kind. Keys are categorized only (return/tab/typing/etc.) — raw typed
characters are NEVER captured. Everything is written to local files you
control; nothing is uploaded anywhere.
