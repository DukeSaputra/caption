use std::path::Path;
use std::process::Command;

use crate::error::CaptionError;

#[derive(Debug, Clone, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

pub fn probe_video(input: &Path, ffmpeg_path: &Path) -> Result<VideoInfo, CaptionError> {
    let ffprobe_name = if cfg!(target_os = "windows") {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };

    let ffprobe_path = ffmpeg_path
        .parent()
        .map(|dir| dir.join(ffprobe_name))
        .unwrap_or_else(|| Path::new(ffprobe_name).to_path_buf());

    if ffprobe_path.exists() {
        let result = Command::new(&ffprobe_path)
            .args([
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0",
                &input.to_string_lossy(),
            ])
            .output();

        if let Ok(output) = result {
            if output.status.success() {
                let csv = String::from_utf8_lossy(&output.stdout);
                if let Ok(info) = parse_ffprobe_csv(csv.trim()) {
                    return Ok(info);
                }
            }
        }
    }

    let output = Command::new(ffmpeg_path)
        .args(["-i", &input.to_string_lossy()])
        .output()
        .map_err(|e| CaptionError::VideoProbeError(format!("Failed to run ffmpeg: {}", e)))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_ffmpeg_stderr(&stderr)
}

// Expected format: `width,height,r_frame_rate` e.g. `1080,1920,30/1`
pub fn parse_ffprobe_csv(csv: &str) -> Result<VideoInfo, CaptionError> {
    let line = csv.lines().find(|l| !l.trim().is_empty()).unwrap_or(csv);

    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 3 {
        return Err(CaptionError::VideoProbeError(format!(
            "ffprobe CSV has too few fields (expected 3, got {}): {:?}",
            parts.len(),
            csv
        )));
    }

    let width = parts[0].trim().parse::<u32>().map_err(|_| {
        CaptionError::VideoProbeError(format!("Invalid width in ffprobe CSV: {:?}", parts[0]))
    })?;

    let height = parts[1].trim().parse::<u32>().map_err(|_| {
        CaptionError::VideoProbeError(format!("Invalid height in ffprobe CSV: {:?}", parts[1]))
    })?;

    let fps = parse_frame_rate(parts[2].trim())?;

    Ok(VideoInfo { width, height, fps })
}

// Handles fractional notation (`30/1`, `30000/1001`) and decimal notation (`29.97`).
pub fn parse_frame_rate(s: &str) -> Result<f64, CaptionError> {
    if let Some((num_str, den_str)) = s.split_once('/') {
        let num = num_str.trim().parse::<f64>().map_err(|_| {
            CaptionError::VideoProbeError(format!("Invalid frame rate numerator: {:?}", num_str))
        })?;
        let den = den_str.trim().parse::<f64>().map_err(|_| {
            CaptionError::VideoProbeError(format!("Invalid frame rate denominator: {:?}", den_str))
        })?;
        if den == 0.0 {
            return Err(CaptionError::VideoProbeError(format!(
                "Frame rate denominator is zero: {:?}",
                s
            )));
        }
        Ok(num / den)
    } else {
        s.trim()
            .parse::<f64>()
            .map_err(|_| CaptionError::VideoProbeError(format!("Invalid frame rate: {:?}", s)))
    }
}

pub fn parse_ffmpeg_stderr(stderr: &str) -> Result<VideoInfo, CaptionError> {
    for line in stderr.lines() {
        if !line.contains("Video:") {
            continue;
        }

        let dims = line.split(',').find_map(|segment| {
            let segment = segment.trim();
            segment.split_whitespace().find_map(|token| {
                let token = token.trim_end_matches(|c: char| !c.is_ascii_digit());
                let (w_str, h_str) = token.split_once('x')?;
                let width = w_str.parse::<u32>().ok()?;
                let height = h_str.parse::<u32>().ok()?;
                if width == 0 || height == 0 {
                    return None;
                }
                Some((width, height))
            })
        });

        let (width, height) = match dims {
            Some(d) => d,
            None => continue,
        };

        let fps = line.split(',').find_map(|segment| {
            let segment = segment.trim();
            let tokens: Vec<&str> = segment.split_whitespace().collect();
            for i in 0..tokens.len().saturating_sub(1) {
                if tokens[i + 1] == "fps" {
                    return parse_frame_rate(tokens[i]).ok();
                }
            }
            None
        });

        let fps = match fps {
            Some(f) => f,
            None => continue,
        };

        return Ok(VideoInfo { width, height, fps });
    }

    Err(CaptionError::VideoProbeError(
        "No video stream found in ffmpeg output".to_string(),
    ))
}

// ---

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_ffprobe_csv ────────────────────────────────────────────────────

    #[test]
    fn parse_ffprobe_csv_standard() {
        let info = parse_ffprobe_csv("1080,1920,30/1").unwrap();
        assert_eq!(info.width, 1080);
        assert_eq!(info.height, 1920);
        assert_eq!(info.fps, 30.0);
    }

    #[test]
    fn parse_ffprobe_csv_ntsc() {
        let info = parse_ffprobe_csv("1920,1080,30000/1001").unwrap();
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
        assert!((info.fps - 29.97002997).abs() < 0.001);
    }

    #[test]
    fn parse_ffprobe_csv_too_few_fields() {
        let err = parse_ffprobe_csv("1080,1920").unwrap_err();
        assert!(matches!(err, CaptionError::VideoProbeError(_)));
        assert!(err.to_string().contains("too few fields"));
    }

    // ── parse_frame_rate ────────────────────────────────────────────────────

    #[test]
    fn parse_frame_rate_fraction() {
        assert_eq!(parse_frame_rate("30/1").unwrap(), 30.0);
        let fps = parse_frame_rate("24000/1001").unwrap();
        assert!((fps - 23.976).abs() < 0.001);
    }

    #[test]
    fn parse_frame_rate_decimal() {
        assert_eq!(parse_frame_rate("29.97").unwrap(), 29.97);
    }

    #[test]
    fn parse_frame_rate_zero_denominator() {
        let err = parse_frame_rate("30/0").unwrap_err();
        assert!(matches!(err, CaptionError::VideoProbeError(_)));
        assert!(err.to_string().contains("zero"));
    }

    // ── parse_ffmpeg_stderr ──────────────────────────────────────────────────

    #[test]
    fn parse_ffmpeg_stderr_finds_video_stream() {
        let stderr = r#"ffmpeg version 6.1.1
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:01:23.45, start: 0.000000, bitrate: 5000 kb/s
  Stream #0:0(und): Video: h264 (High), yuv420p, 1080x1920 [SAR 1:1 DAR 9:16], 4900 kb/s, 30 fps, 30 tbr, 90k tbn, 60 tbc
  Stream #0:1(und): Audio: aac, 44100 Hz, stereo, fltp, 128 kb/s
"#;
        let info = parse_ffmpeg_stderr(stderr).unwrap();
        assert_eq!(info.width, 1080);
        assert_eq!(info.height, 1920);
        assert_eq!(info.fps, 30.0);
    }

    #[test]
    fn parse_ffmpeg_stderr_no_video() {
        let stderr = r#"ffmpeg version 6.1.1
Input #0, mp3, from 'audio.mp3':
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 192 kb/s
"#;
        let err = parse_ffmpeg_stderr(stderr).unwrap_err();
        assert!(matches!(err, CaptionError::VideoProbeError(_)));
        assert!(err.to_string().contains("No video stream"));
    }

    #[test]
    fn parse_ffmpeg_stderr_decimal_fps() {
        let stderr = r#"ffmpeg version 6.1.1
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
  Stream #0:0(und): Video: h264, yuv420p, 1920x1080, 4900 kb/s, 29.97 fps, 29.97 tbr, 90k tbn
"#;
        let info = parse_ffmpeg_stderr(stderr).unwrap();
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
        assert_eq!(info.fps, 29.97);
    }
}
