//! Cross-process "please stop" signal for the capture session.
//!
//! Mirrors the pattern in native_capture/windows/StopSignal.cs and the SIGINT
//! stop path in native_capture/Sources/MentorCapture: the `start` process
//! registers a per-session stop handle, and the `stop` subcommand signals it.
//! Ctrl+C (SIGINT) also stops gracefully on every platform via the ctrlc
//! handler.
//!
//! Implementation: a **pidfile** plus a **sentinel file**, both in the
//! platform temp directory and keyed by session id, so `stop --session <id>`
//! can find the running capture without needing `--output`. `start` writes
//! its PID and polls for the sentinel each capture tick; `stop` creates the
//! sentinel. This is dependency-free and works identically on macOS and
//! Windows, unlike POSIX signals (which can't be delivered to an arbitrary
//! Win32 process). On macOS, SIGINT (Ctrl+C) remains a supported stop path
//! via the ctrlc handler.
//!
//! Stop latency is at most one capture tick (~1s at the 1fps sample rate):
//! the capture loop checks `stop_requested` at the top of each iteration.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Returns the pidfile path for a session in the platform temp directory.
/// Both `start` and `stop` derive this from the session id alone (no
/// `--output` needed), so they agree on the location as long as they run as
/// the same user in the same login session (temp_dir is stable per user).
fn pidfile_path(session_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("cyberalfred-capture-{session_id}.pid"))
}

/// Returns the sentinel (stop-request) file path for a session. Its mere
/// existence is the stop signal -- the start loop polls for it each tick.
fn sentinel_path(session_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("cyberalfred-capture-{session_id}.stop"))
}

/// A guard that removes the pidfile and sentinel when dropped, so a clean
/// shutdown (Ctrl+C, `stop`, or normal exit) doesn't leave stale files that
/// would make a later `stop` think a capture is still running.
pub struct StopGuard {
    session_id: String,
}

impl Drop for StopGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(pidfile_path(&self.session_id));
        let _ = fs::remove_file(sentinel_path(&self.session_id));
    }
}

/// Called by `start`: clears any stale sentinel, writes the pidfile, and
/// installs the Ctrl+C handler that flips `running` to false. Returns a guard
/// that cleans up both files on drop. A pidfile write failure is fatal (the
/// `stop` subcommand couldn't find this session); a ctrlc-handler install
/// failure is not (the sentinel path still works).
pub fn install(session_id: &str, running: Arc<AtomicBool>) -> Result<StopGuard, String> {
    // Clear a stale sentinel left by a previous crashed run so the new
    // session doesn't immediately read it as a stop request.
    let _ = fs::remove_file(sentinel_path(session_id));

    let pidfile = pidfile_path(session_id);
    if let Err(err) = fs::write(&pidfile, std::process::id().to_string()) {
        return Err(format!(
            "could not write pidfile {}: {err}",
            pidfile.display()
        ));
    }

    let running_for_handler = running.clone();
    if let Err(err) = ctrlc::set_handler(move || {
        running_for_handler.store(false, Ordering::SeqCst);
    }) {
        // Non-fatal: Ctrl+C just won't flip the flag, but `stop` still works.
        eprintln!("cyberalfred-capture: could not install Ctrl-C handler: {err}");
    }

    Ok(StopGuard {
        session_id: session_id.to_string(),
    })
}

/// Called by the capture loop each tick: true if `stop` has created the
/// sentinel file, signaling the running capture to shut down gracefully.
pub fn stop_requested(session_id: &str) -> bool {
    sentinel_path(session_id).exists()
}

/// Called by the `stop` subcommand: creates the sentinel file to signal the
/// running `start` process to shut down. Returns true if a capture appears to
/// be running (the pidfile exists), false otherwise -- matching the Windows
/// helper's `stop` exit code semantics (0 = signaled, 2 = no active capture).
pub fn signal_stop(session_id: &str) -> bool {
    if !pidfile_path(session_id).exists() {
        return false;
    }
    // Touch the sentinel; the start loop polls for it each tick (~1s latency).
    // The sentinel is removed by the start process's StopGuard on shutdown.
    let _ = fs::write(sentinel_path(session_id), "");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique session id per test so concurrent tests don't collide on the
    /// same pidfile/sentinel in the shared temp directory.
    fn unique_session(label: &str) -> String {
        format!(
            "test-{}-{}-{}",
            label,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }

    #[test]
    fn signal_stop_false_when_no_pidfile() {
        let session = unique_session("no-pid");
        // No pidfile written -> signal_stop must report no capture running.
        assert!(!signal_stop(&session));
        assert!(!stop_requested(&session), "no sentinel should be created when no pidfile");
    }

    #[test]
    fn install_then_signal_stop_creates_sentinel() {
        let session = unique_session("roundtrip");
        let running = Arc::new(AtomicBool::new(true));
        let guard = install(&session, running.clone()).unwrap();

        // After install, the pidfile exists and no stop is requested yet.
        assert!(pidfile_path(&session).exists());
        assert!(!stop_requested(&session));

        // signal_stop reports a capture is running and creates the sentinel.
        assert!(signal_stop(&session));
        assert!(stop_requested(&session), "sentinel must exist after signal_stop");

        // The capture loop would see the sentinel and flip running to false.
        // (Simulated here; the real loop checks stop_requested each tick.)
        running.store(false, Ordering::SeqCst);
        assert!(!running.load(Ordering::SeqCst));

        // Dropping the guard (simulating clean shutdown) removes both files.
        drop(guard);
        assert!(!pidfile_path(&session).exists(), "guard must remove pidfile");
        assert!(!sentinel_path(&session).exists(), "guard must remove sentinel");
    }

    #[test]
    fn install_clears_stale_sentinel() {
        let session = unique_session("stale");
        // Plant a stale sentinel (as if a previous run crashed mid-stop).
        fs::write(sentinel_path(&session), "").unwrap();
        assert!(stop_requested(&session));

        let running = Arc::new(AtomicBool::new(true));
        let guard = install(&session, running).unwrap();

        // install must clear the stale sentinel so the new session isn't
        // immediately told to stop.
        assert!(!stop_requested(&session), "install must clear a stale sentinel");

        drop(guard);
        assert!(!pidfile_path(&session).exists());
        assert!(!sentinel_path(&session).exists());
    }

    #[test]
    fn pidfile_and_sentinel_paths_are_distinct_and_keyed_by_session() {
        let a = "session-a";
        let b = "session-b";
        assert_ne!(pidfile_path(a), pidfile_path(b));
        assert_ne!(sentinel_path(a), sentinel_path(b));
        assert_ne!(pidfile_path(a), sentinel_path(a));
    }
}
