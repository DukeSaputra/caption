use std::path::PathBuf;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use log::debug;

use crate::error::CaptionError;

const DEFAULT_FONT: &str = "inter-bold.ttf";

pub fn find_font(explicit_path: Option<&str>) -> Result<PathBuf, CaptionError> {
    if let Some(path) = explicit_path {
        let p = PathBuf::from(path);
        if p.exists() {
            debug!("Using explicit font path: {}", p.display());
            return Ok(p);
        }
        return Err(CaptionError::FontNotFound(path.to_string()));
    }

    let mut searched: Vec<PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        let resolved = std::fs::canonicalize(&exe).unwrap_or(exe);
        if let Some(dir) = resolved.parent() {
            let candidate = dir.join(DEFAULT_FONT);
            debug!("Checking exe dir: {}", candidate.display());
            if candidate.exists() {
                return Ok(candidate);
            }
            searched.push(candidate);
        }
    }

    #[cfg(target_os = "macos")]
    let font_dirs: &[&str] = &["/System/Library/Fonts", "/Library/Fonts"];

    #[cfg(target_os = "linux")]
    let font_dirs: &[&str] = &["/usr/share/fonts", "/usr/local/share/fonts"];

    #[cfg(target_os = "windows")]
    let font_dirs: &[&str] = &["C:\\Windows\\Fonts"];

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let font_dirs: &[&str] = &[];

    for dir in font_dirs {
        let candidate = PathBuf::from(dir).join(DEFAULT_FONT);
        debug!("Checking font dir: {}", candidate.display());
        if candidate.exists() {
            return Ok(candidate);
        }
        searched.push(candidate);
    }

    let searched_str = searched
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Err(CaptionError::FontNotFound(searched_str))
}

// Base size is 64pt at 1920px tall.
pub fn scaled_font_size(video_height: u32) -> f32 {
    120.0 * (video_height as f32 / 1920.0)
}

pub fn measure_text_width(
    font_data: &[u8],
    text: &str,
    font_size: f32,
) -> Result<f32, CaptionError> {
    let font = FontRef::try_from_slice(font_data)
        .map_err(|e| CaptionError::BurnFailed(format!("Failed to parse font: {e}")))?;

    let scale = PxScale::from(font_size);
    let scaled = font.as_scaled(scale);

    let mut width = 0.0f32;
    let mut prev_glyph_id = None;

    for ch in text.chars() {
        let glyph_id = scaled.glyph_id(ch);

        if let Some(prev) = prev_glyph_id {
            width += scaled.kern(prev, glyph_id);
        }

        width += scaled.h_advance(glyph_id);
        prev_glyph_id = Some(glyph_id);
    }

    Ok(width)
}

#[derive(Debug, Clone)]
pub struct PillDimensions {
    pub width: f32,
    pub height: f32,
    pub radius: f32,
    pub x: f32,
    pub y: f32,
}

// Pill is horizontally centered and placed at 70% from the top of the frame.
pub fn pill_dimensions(
    text_width: f32,
    font_size: f32,
    video_width: u32,
    video_height: u32,
) -> PillDimensions {
    let pad_scale = video_height as f32 / 1920.0;
    let pad_h = 95.0 * pad_scale;
    let pad_v = 45.0 * pad_scale;

    let pill_w = text_width + pad_h * 2.0;
    let pill_h = font_size + pad_v * 2.0;
    let radius = pill_h / 2.0;

    let x = (video_width as f32 - pill_w) / 2.0;
    let y = video_height as f32 * 0.75 - pill_h / 2.0;

    PillDimensions {
        width: pill_w,
        height: pill_h,
        radius,
        x,
        y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_font_size_at_1920() {
        assert_eq!(scaled_font_size(1920), 120.0);
    }

    #[test]
    fn scaled_font_size_at_1080() {
        let size = scaled_font_size(1080);
        let expected = 120.0 * (1080.0f32 / 1920.0);
        assert!(
            (size - expected).abs() < 1e-4,
            "expected {expected}, got {size}"
        );
    }

    #[test]
    fn scaled_font_size_at_2560() {
        let size = scaled_font_size(2560);
        let expected = 120.0 * (2560.0f32 / 1920.0);
        assert!(
            (size - expected).abs() < 1e-4,
            "expected {expected}, got {size}"
        );
    }

    #[test]
    fn pill_dimensions_centered_at_75_percent() {
        let text_width = 200.0f32;
        let font_size = 120.0f32;
        let video_width = 1080u32;
        let video_height = 1920u32;

        let dims = pill_dimensions(text_width, font_size, video_width, video_height);

        let expected_x = (video_width as f32 - dims.width) / 2.0;
        assert!(
            (dims.x - expected_x).abs() < 1e-4,
            "x not centered: got {}",
            dims.x
        );

        let expected_y = video_height as f32 * 0.75 - dims.height / 2.0;
        assert!(
            (dims.y - expected_y).abs() < 1e-4,
            "y not at 75%: got {}",
            dims.y
        );
    }

    #[test]
    fn pill_radius_is_half_height() {
        let dims = pill_dimensions(150.0, 36.0, 1080, 1080);
        assert!((dims.radius - dims.height / 2.0).abs() < 1e-4);
    }

    #[test]
    fn pill_width_includes_padding() {
        let text_width = 200.0f32;
        let video_height = 1920u32;
        let pad_scale = video_height as f32 / 1920.0;
        let pad_h = 95.0 * pad_scale;

        let dims = pill_dimensions(text_width, 120.0, 1080, video_height);

        let expected_width = text_width + pad_h * 2.0;
        assert!((dims.width - expected_width).abs() < 1e-4);

        assert!(dims.width > text_width + 2.0 * 60.0 - 1e-4);
    }

    #[test]
    fn find_font_explicit_nonexistent_returns_error() {
        let result = find_font(Some("/nonexistent/path/to/font.ttf"));
        assert!(matches!(result, Err(CaptionError::FontNotFound(_))));
        if let Err(CaptionError::FontNotFound(path)) = result {
            assert!(path.contains("/nonexistent/path/to/font.ttf"));
        }
    }
}
