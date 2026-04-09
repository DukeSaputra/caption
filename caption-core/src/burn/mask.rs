use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use tiny_skia::{
    Color, FillRule, GradientStop, LinearGradient, Paint, PathBuilder, Pixmap, SpreadMode, Stroke,
    Transform,
};

use crate::burn::font::PillDimensions;

// ~30% opacity
const TINT_ALPHA: u8 = 75;

const HIGHLIGHT_ALPHA: u8 = 40;

const BORDER_TOP_ALPHA: u8 = 60;

const BORDER_BOTTOM_ALPHA: u8 = 20;

const BORDER_WIDTH: f32 = 5.0;

// Kappa: cubic Bezier approximation of circular arcs
const KAPPA: f32 = 0.5522847498;

fn build_pill_path(pill: &PillDimensions) -> Option<tiny_skia::Path> {
    let x = pill.x;
    let y = pill.y;
    let w = pill.width;
    let h = pill.height;
    let r = pill.radius.min(w / 2.0).min(h / 2.0);
    let k = r * KAPPA;

    let mut pb = PathBuilder::new();

    pb.move_to(x + r, y);
    pb.line_to(x + w - r, y);
    pb.cubic_to(x + w - r + k, y, x + w, y + r - k, x + w, y + r);
    pb.line_to(x + w, y + h - r);
    pb.cubic_to(x + w, y + h - r + k, x + w - r + k, y + h, x + w - r, y + h);
    pb.line_to(x + r, y + h);
    pb.cubic_to(x + r - k, y + h, x, y + h - r + k, x, y + h - r);
    pb.line_to(x, y + r);
    pb.cubic_to(x, y + r - k, x + r - k, y, x + r, y);

    pb.close();
    pb.finish()
}

pub fn render_overlay_frame(
    video_width: u32,
    video_height: u32,
    pill: &PillDimensions,
    text: &str,
    font_data: &[u8],
    font_size: f32,
    opacity: f32,
) -> Vec<u8> {
    let double_height = video_height * 2;
    let mut pixmap =
        Pixmap::new(video_width, double_height).expect("video dimensions must be non-zero");

    let mask_alpha = (255.0 * opacity) as u8;

    if let Some(path) = build_pill_path(pill) {
        let mut paint = Paint::default();
        paint.set_color_rgba8(255, 255, 255, mask_alpha);
        paint.anti_alias = true;
        pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }

    let bottom_pill = PillDimensions {
        x: pill.x,
        y: pill.y + video_height as f32,
        width: pill.width,
        height: pill.height,
        radius: pill.radius,
    };

    if let Some(path) = build_pill_path(&bottom_pill) {
        let tint_a = (TINT_ALPHA as f32 * opacity) as u8;
        let mut paint = Paint::default();
        paint.set_color_rgba8(10, 10, 10, tint_a);
        paint.anti_alias = true;
        pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }

    if let Some(path) = build_pill_path(&bottom_pill) {
        let highlight_a = (HIGHLIGHT_ALPHA as f32 * opacity) as u8;
        let gradient = LinearGradient::new(
            tiny_skia::Point::from_xy(0.0, bottom_pill.y),
            tiny_skia::Point::from_xy(0.0, bottom_pill.y + bottom_pill.height * 0.7),
            vec![
                GradientStop::new(0.0, Color::from_rgba8(255, 255, 255, highlight_a)),
                GradientStop::new(0.4, Color::from_rgba8(255, 255, 255, highlight_a / 3)),
                GradientStop::new(1.0, Color::from_rgba8(255, 255, 255, 0)),
            ],
            SpreadMode::Pad,
            Transform::identity(),
        );
        if let Some(shader) = gradient {
            let mut paint = Paint::default();
            paint.shader = shader;
            paint.anti_alias = true;
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }

    if let Some(path) = build_pill_path(&bottom_pill) {
        let top_a = (BORDER_TOP_ALPHA as f32 * opacity) as u8;
        let bot_a = (BORDER_BOTTOM_ALPHA as f32 * opacity) as u8;
        let gradient = LinearGradient::new(
            tiny_skia::Point::from_xy(0.0, bottom_pill.y),
            tiny_skia::Point::from_xy(0.0, bottom_pill.y + bottom_pill.height),
            vec![
                GradientStop::new(0.0, Color::from_rgba8(255, 255, 255, top_a)),
                GradientStop::new(1.0, Color::from_rgba8(255, 255, 255, bot_a)),
            ],
            SpreadMode::Pad,
            Transform::identity(),
        );
        if let Some(shader) = gradient {
            let mut paint = Paint::default();
            paint.shader = shader;
            paint.anti_alias = true;
            let stroke = Stroke {
                width: BORDER_WIDTH,
                ..Stroke::default()
            };
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }

    draw_text(
        &mut pixmap,
        &bottom_pill,
        text,
        font_data,
        font_size,
        opacity,
    );

    pixmap.data().to_vec()
}

pub fn render_empty_frame(video_width: u32, video_height: u32) -> Vec<u8> {
    vec![0u8; video_width as usize * video_height as usize * 2 * 4]
}

fn draw_text(
    pixmap: &mut Pixmap,
    pill: &PillDimensions,
    text: &str,
    font_data: &[u8],
    font_size: f32,
    opacity: f32,
) {
    let font = match FontRef::try_from_slice(font_data) {
        Ok(f) => f,
        Err(_) => return,
    };

    let scale = PxScale::from(font_size);
    let scaled = font.as_scaled(scale);

    let mut text_width = 0.0_f32;
    let glyphs: Vec<_> = {
        let mut prev_id = None;
        text.chars()
            .map(|ch| {
                let glyph_id = scaled.glyph_id(ch);
                let kern = prev_id.map(|p| scaled.kern(p, glyph_id)).unwrap_or(0.0);
                let advance = scaled.h_advance(glyph_id);
                let x_offset = text_width + kern;
                text_width += advance + kern;
                prev_id = Some(glyph_id);
                (glyph_id, x_offset)
            })
            .collect()
    };

    let text_x = pill.x + (pill.width - text_width) / 2.0;
    let ascent = scaled.ascent();
    let descent = scaled.descent();
    let text_height = ascent - descent;
    let text_y = pill.y + (pill.height - text_height) / 2.0 + ascent;

    let pw = pixmap.width() as usize;
    let ph = pixmap.height() as usize;
    let stride = pw * 4;
    let data = pixmap.data_mut();

    for &(glyph_id, x_offset) in &glyphs {
        let glyph = glyph_id.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            let gx = (text_x + x_offset + bounds.min.x) as i32;
            let gy = (text_y + bounds.min.y) as i32;

            outlined.draw(|rx, ry, coverage| {
                let px = gx + rx as i32;
                let py = gy + ry as i32;
                if px < 0 || py < 0 || px as usize >= pw || py as usize >= ph {
                    return;
                }

                let alpha = (coverage * 255.0 * opacity) as u8;
                if alpha == 0 {
                    return;
                }

                let offset = py as usize * stride + px as usize * 4;
                let dst_r = data[offset];
                let dst_g = data[offset + 1];
                let dst_b = data[offset + 2];
                let dst_a = data[offset + 3];

                let src_a = alpha as u16;
                let inv_a = 255 - src_a;

                data[offset] = ((255 * src_a + dst_r as u16 * inv_a) / 255) as u8;
                data[offset + 1] = ((255 * src_a + dst_g as u16 * inv_a) / 255) as u8;
                data[offset + 2] = ((255 * src_a + dst_b as u16 * inv_a) / 255) as u8;
                data[offset + 3] = ((src_a + dst_a as u16 * inv_a / 255).min(255)) as u8;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pill() -> PillDimensions {
        PillDimensions {
            x: 50.0,
            y: 70.0,
            width: 200.0,
            height: 80.0,
            radius: 40.0,
        }
    }

    #[test]
    fn empty_frame_is_double_height() {
        let frame = render_empty_frame(100, 100);
        assert_eq!(frame.len(), 100 * 200 * 4);
        assert!(frame.iter().all(|&b| b == 0));
    }

    #[test]
    fn overlay_frame_is_double_height() {
        let frame = render_overlay_frame(300, 200, &test_pill(), "hi", &[], 32.0, 1.0);
        assert_eq!(frame.len(), 300 * 400 * 4);
    }

    #[test]
    fn top_half_has_full_alpha_inside_pill() {
        let frame = render_overlay_frame(300, 200, &test_pill(), "X", &[], 32.0, 1.0);
        let cx: usize = 150;
        let cy: usize = 110;
        let stride = 300 * 4;
        let alpha = frame[cy * stride + cx * 4 + 3];
        assert_eq!(
            alpha, 255,
            "top half pill center alpha should be 255, got {alpha}"
        );
    }

    #[test]
    fn bottom_half_has_tint_alpha_inside_pill() {
        let frame = render_overlay_frame(300, 200, &test_pill(), "X", &[], 32.0, 1.0);
        let cx: usize = 150;
        let cy: usize = 310;
        let stride = 300 * 4;
        let alpha = frame[cy * stride + cx * 4 + 3];
        assert!(
            alpha > 0,
            "bottom half pill alpha should be nonzero, got {alpha}"
        );
    }

    #[test]
    fn outside_pill_is_transparent() {
        let frame = render_overlay_frame(300, 200, &test_pill(), "X", &[], 32.0, 1.0);
        assert_eq!(frame[3], 0);
        let stride = 300 * 4;
        assert_eq!(frame[200 * stride + 3], 0);
    }

    #[test]
    fn pill_path_builds_successfully() {
        let pill = test_pill();
        assert!(build_pill_path(&pill).is_some());
    }

    // --- Edge case: pill at top-left corner (0,0) ---

    #[test]
    fn pill_at_origin() {
        let pill = PillDimensions {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
            radius: 25.0,
        };
        assert!(build_pill_path(&pill).is_some());
    }

    // --- Edge case: pill touching right/bottom video boundary ---

    #[test]
    fn pill_at_bottom_right_boundary() {
        let vw = 300u32;
        let vh = 200u32;
        // Pill positioned so its right and bottom edges touch the video boundary
        let pill = PillDimensions {
            x: (vw as f32) - 100.0,
            y: (vh as f32) - 50.0,
            width: 100.0,
            height: 50.0,
            radius: 25.0,
        };
        let frame = render_overlay_frame(vw, vh, &pill, "X", &[], 32.0, 1.0);
        assert_eq!(frame.len(), vw as usize * vh as usize * 2 * 4);
    }

    // --- Edge case: very small video dimensions ---

    #[test]
    fn very_small_video_dimensions() {
        let frame = render_overlay_frame(
            2,
            2,
            &PillDimensions {
                x: 0.0,
                y: 0.0,
                width: 2.0,
                height: 2.0,
                radius: 1.0,
            },
            "x",
            &[],
            8.0,
            1.0,
        );
        // 2 wide * 4 tall (double height) * 4 bytes RGBA
        assert_eq!(frame.len(), 2 * 4 * 4);
    }

    #[test]
    fn empty_frame_very_small() {
        let frame = render_empty_frame(1, 1);
        // 1 * 2 * 4 = 8 bytes (width=1, double height=2, RGBA=4)
        assert_eq!(frame.len(), 8);
        assert!(frame.iter().all(|&b| b == 0));
    }

    // --- Edge case: pill wider than video ---

    #[test]
    fn pill_wider_than_video() {
        let pill = PillDimensions {
            x: -50.0,
            y: 70.0,
            width: 400.0,
            height: 80.0,
            radius: 40.0,
        };
        // Should not panic even if pill extends beyond video bounds
        let frame = render_overlay_frame(300, 200, &pill, "X", &[], 32.0, 1.0);
        assert_eq!(frame.len(), 300 * 400 * 4);
    }

    // --- Edge case: radius larger than half the dimensions ---

    #[test]
    fn pill_path_clamps_radius_to_half_dimension() {
        let pill = PillDimensions {
            x: 10.0,
            y: 10.0,
            width: 40.0,
            height: 20.0,
            radius: 100.0, // way larger than w/2 and h/2
        };
        // build_pill_path clamps r to min(w/2, h/2), should still build
        assert!(build_pill_path(&pill).is_some());
    }

    // --- Edge case: zero-size pill ---

    #[test]
    fn pill_path_zero_dimensions() {
        let pill = PillDimensions {
            x: 50.0,
            y: 50.0,
            width: 0.0,
            height: 0.0,
            radius: 0.0,
        };
        // Degenerate path; may return None or a zero-area path
        let _ = build_pill_path(&pill);
    }

    // --- Edge case: zero opacity ---

    #[test]
    fn overlay_zero_opacity_is_transparent() {
        let frame = render_overlay_frame(300, 200, &test_pill(), "hi", &[], 32.0, 0.0);
        assert_eq!(frame.len(), 300 * 400 * 4);
        // With zero opacity, every pixel's alpha channel should be 0
        for i in (3..frame.len()).step_by(4) {
            assert_eq!(frame[i], 0, "non-zero alpha at byte offset {i}");
        }
    }

    // --- Edge case: pill at exact video center ---

    #[test]
    fn pill_centered_in_frame() {
        let vw = 400u32;
        let vh = 300u32;
        let pw = 100.0f32;
        let ph = 50.0f32;
        let pill = PillDimensions {
            x: (vw as f32 - pw) / 2.0,
            y: (vh as f32 - ph) / 2.0,
            width: pw,
            height: ph,
            radius: ph / 2.0,
        };
        let frame = render_overlay_frame(vw, vh, &pill, "Z", &[], 24.0, 1.0);
        assert_eq!(frame.len(), vw as usize * vh as usize * 2 * 4);
        // The center of the pill in the top half should have non-zero alpha
        let cx = vw as usize / 2;
        let cy = vh as usize / 2; // center of top half where pill is
        let stride = vw as usize * 4;
        let alpha = frame[cy * stride + cx * 4 + 3];
        assert!(alpha > 0, "expected non-zero alpha at pill center");
    }
}
