//! Simple, dependency-free screenshot change detection.
//!
//! No ML, no computer-vision libraries: downscale the frame to a small
//! grayscale grid and compare the mean absolute pixel difference against
//! the last *retained* frame's grid. This mirrors the approach already
//! used by the existing Swift native capture helper in the main cyberAlfred
//! repo (see native_capture/Sources/MentorCapture/ScreenObserver.swift),
//! reimplemented here from scratch so this proof of concept has no runtime
//! dependency on that code.

/// Side length of the downsampled grayscale grid used for comparison.
pub const GRID_SIZE: usize = 32;
/// Mean pixel difference (0.0..=1.0) above which a frame is considered
/// "meaningfully changed".
pub const CHANGE_THRESHOLD: f64 = 0.02;

/// Downscales a raw BGRA buffer to a `GRID_SIZE x GRID_SIZE` grayscale grid
/// using simple block averaging. No resizing/image library involved -- this
/// operates directly on the raw pixel bytes scap hands back.
pub fn downscale_grayscale_bgra(bgra: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    let mut grid = vec![0u8; GRID_SIZE * GRID_SIZE];
    if width == 0 || height == 0 || bgra.len() < width * height * 4 {
        return grid;
    }

    let block_w = (width / GRID_SIZE).max(1);
    let block_h = (height / GRID_SIZE).max(1);

    for gy in 0..GRID_SIZE {
        for gx in 0..GRID_SIZE {
            let x0 = (gx * block_w).min(width - 1);
            let y0 = (gy * block_h).min(height - 1);
            let x1 = (x0 + block_w).min(width).max(x0 + 1);
            let y1 = (y0 + block_h).min(height).max(y0 + 1);

            let mut sum: u64 = 0;
            let mut count: u64 = 0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * width + x) * 4;
                    if idx + 2 < bgra.len() {
                        let b = bgra[idx] as u64;
                        let g = bgra[idx + 1] as u64;
                        let r = bgra[idx + 2] as u64;
                        // Standard luma approximation (ITU-R BT.601).
                        sum += (r * 299 + g * 587 + b * 114) / 1000;
                        count += 1;
                    }
                }
            }
            grid[gy * GRID_SIZE + gx] = if count > 0 { (sum / count) as u8 } else { 0 };
        }
    }
    grid
}

/// Mean absolute difference between two grayscale grids, normalized to
/// 0.0..=1.0. Returns 1.0 (treat as fully different, i.e. always retain
/// the first frame) when there is no baseline to compare against.
pub fn difference(current: &[u8], previous: Option<&[u8]>) -> f64 {
    let previous = match previous {
        Some(previous) => previous,
        None => return 1.0,
    };
    if previous.len() != current.len() || current.is_empty() {
        return 1.0;
    }
    let total: u64 = current
        .iter()
        .zip(previous.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
        .sum();
    let max_possible = (current.len() as u64) * 255;
    if max_possible == 0 {
        0.0
    } else {
        total as f64 / max_possible as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_baseline_is_always_different() {
        let grid = vec![0u8; GRID_SIZE * GRID_SIZE];
        assert_eq!(difference(&grid, None), 1.0);
    }

    #[test]
    fn identical_grids_have_zero_difference() {
        let grid = vec![128u8; GRID_SIZE * GRID_SIZE];
        assert_eq!(difference(&grid, Some(&grid)), 0.0);
    }

    #[test]
    fn fully_different_grids_have_max_difference() {
        let black = vec![0u8; GRID_SIZE * GRID_SIZE];
        let white = vec![255u8; GRID_SIZE * GRID_SIZE];
        assert_eq!(difference(&white, Some(&black)), 1.0);
    }

    #[test]
    fn downscale_handles_undersized_buffer_gracefully() {
        let grid = downscale_grayscale_bgra(&[0, 0, 0, 0], 10, 10);
        assert_eq!(grid.len(), GRID_SIZE * GRID_SIZE);
    }
}
