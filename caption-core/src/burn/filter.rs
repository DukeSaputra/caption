pub fn build_filter_complex(video_width: u32, video_height: u32) -> String {
    format!(
        "[1:v]crop={w}:{h}:0:0[blur_mask];\
         [1:v]crop={w}:{h}:0:{h}[content];\
         [0:v]split[orig][src];\
         [src]gblur=sigma=20,eq=saturation=1.8:brightness=0.05[blurred];\
         [blurred][blur_mask]alphamerge[masked_blur];\
         [orig][masked_blur]overlay=format=auto[with_blur];\
         [with_blur][content]overlay=format=auto[out]",
        w = video_width,
        h = video_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_ends_with_out_label() {
        let f = build_filter_complex(1080, 1920);
        assert!(f.ends_with("[out]"));
    }

    #[test]
    fn filter_crops_top_and_bottom() {
        let f = build_filter_complex(1080, 1920);
        assert!(f.contains("crop=1080:1920:0:0[blur_mask]"));
        assert!(f.contains("crop=1080:1920:0:1920[content]"));
    }

    #[test]
    fn filter_has_blur_pipeline() {
        let f = build_filter_complex(1080, 1920);
        assert!(f.contains("gblur=sigma=20"));
        assert!(f.contains("saturation=1.8"));
        assert!(f.contains("alphamerge"));
    }

    #[test]
    fn filter_has_two_overlays() {
        let f = build_filter_complex(1080, 1920);
        assert_eq!(f.matches("overlay=format=auto").count(), 2);
    }

    #[test]
    fn filter_has_no_drawtext() {
        let f = build_filter_complex(1080, 1920);
        assert!(!f.contains("drawtext"));
    }
}
