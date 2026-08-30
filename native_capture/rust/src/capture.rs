//! Thin wrapper around the `scap` cross-platform screen capture crate.
//!
//! This is the only file that talks to `scap` directly -- everything else
//! works with the plain `CapturedFrame` struct below, so a future change to
//! the capture backend only touches this file.

use scap::capturer::{Capturer, Options, Resolution};
use scap::frame::{Frame, FrameType};

/// One captured video frame, decoupled from scap's own frame types.
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    /// Raw pixels in BGRA order, `width * height * 4` bytes.
    pub bgra: Vec<u8>,
}

pub struct ScreenCapture {
    capturer: Capturer,
    running: bool,
}

impl ScreenCapture {
    /// Checks platform support and Screen Recording permission (requesting
    /// it if needed), then starts capturing the primary display at `fps`.
    pub fn start(fps: u32) -> Result<Self, String> {
        if !scap::is_supported() {
            return Err("this platform is not supported by scap".to_string());
        }

        if !scap::has_permission() {
            eprintln!("cyberalfred-capture: requesting Screen Recording permission...");
            if !scap::request_permission() {
                return Err(
                    "Screen Recording permission was not granted. Grant it in System Settings \
                     -> Privacy & Security -> Screen Recording for the terminal app you ran \
                     this from, then run again."
                        .to_string(),
                );
            }
        }

        let options = Options {
            fps,
            target: None, // None captures the primary display.
            show_cursor: true,
            show_highlight: false,
            excluded_targets: None,
            output_type: FrameType::BGRAFrame,
            output_resolution: Resolution::Captured,
            ..Default::default()
        };

        let mut capturer = Capturer::build(options).map_err(|err| err.to_string())?;
        capturer.start_capture();
        Ok(Self {
            capturer,
            running: true,
        })
    }

    /// Returns true if the capturer is active (started successfully).
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Blocks until the next video frame is available.
    pub fn next_frame(&mut self) -> Result<CapturedFrame, String> {
        match self.capturer.get_next_frame() {
            Ok(Frame::BGRA(frame)) => Ok(CapturedFrame {
                width: frame.width.max(0) as u32,
                height: frame.height.max(0) as u32,
                bgra: frame.data,
            }),
            Ok(_) => Err("received an unexpected frame format".to_string()),
            Err(err) => Err(format!("{err}")),
        }
    }

    pub fn stop(&mut self) {
        if self.running {
            self.capturer.stop_capture();
            self.running = false;
        }
    }
}
