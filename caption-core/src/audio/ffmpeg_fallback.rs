use std::path::{Path, PathBuf};
use std::process::Command;

use log::debug;

use crate::error::CaptionError;

fn find_ffmpeg() -> Result<PathBuf, CaptionError> {
    let binary_name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let bundled = exe_dir.join(binary_name);
            if bundled.is_file() {
                debug!("Found bundled FFmpeg at {}", bundled.display());
                return Ok(bundled);
            }
        }
    }

    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(binary_name);
            if candidate.is_file() {
                debug!("Found system FFmpeg at {}", candidate.display());
                return Ok(candidate);
            }
        }
    }

    Err(ffmpeg_not_found_error())
}

pub fn find_ffmpeg_public() -> Result<PathBuf, CaptionError> {
    find_ffmpeg()
}

fn ffmpeg_not_found_error() -> CaptionError {
    CaptionError::FfmpegNotFound(
        "FFmpeg is required for this file format. \
         Install FFmpeg or place the ffmpeg binary next to the caption executable."
            .to_string(),
    )
}

pub fn extract_with_ffmpeg(input_path: &Path) -> Result<Vec<f32>, CaptionError> {
    let ffmpeg = find_ffmpeg()?;

    let child = Command::new(&ffmpeg)
        .args([
            "-i",
            &input_path.display().to_string(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .spawn()
        .map_err(|e| {
            CaptionError::ExtractionFailed(format!("failed to spawn FFmpeg process: {e}"))
        })?;

    let output = child.wait_with_output().map_err(|e| {
        CaptionError::ExtractionFailed(format!("failed to read FFmpeg output: {e}"))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output
            .status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        return Err(CaptionError::ExtractionFailed(format!(
            "FFmpeg exited with code {code}: {stderr}"
        )));
    }

    let raw = output.stdout;
    if raw.len() % 4 != 0 {
        return Err(CaptionError::ExtractionFailed(format!(
            "FFmpeg produced {} bytes of PCM data, which is not a multiple of 4 (expected f32le)",
            raw.len()
        )));
    }

    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    debug!(
        "Decoded via FFmpeg fallback: {} samples ({:.1}s at 16 kHz)",
        samples.len(),
        samples.len() as f64 / 16000.0
    );

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg_available() -> bool {
        find_ffmpeg().is_ok()
    }

    fn create_test_wav(path: &Path, sample_rate: u32, duration_secs: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    // ---- Existing tests ----

    #[test]
    fn ffmpeg_not_found_error_message_is_descriptive() {
        let err = ffmpeg_not_found_error();
        let msg = err.to_string();
        assert!(msg.contains("FFmpeg is required"), "got: {msg}");
        assert!(msg.contains("Install FFmpeg"), "got: {msg}");
    }

    #[test]
    fn nonexistent_file_returns_extraction_failed() {
        let path = PathBuf::from("/tmp/this-file-absolutely-does-not-exist.webm");
        let result = extract_with_ffmpeg(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            CaptionError::ExtractionFailed(msg) => {
                assert!(
                    msg.contains("FFmpeg exited"),
                    "Expected FFmpeg exit error, got: {msg}"
                );
            }
            CaptionError::FfmpegNotFound(_) => {}
            other => panic!("Unexpected error variant: {other:?}"),
        }
    }

    // ---- New tests: FFmpeg not found ----

    #[test]
    #[ignore]
    fn find_ffmpeg_with_empty_path_returns_not_found_or_bundled() {
        let original = std::env::var("PATH").unwrap_or_default();
        std::env::set_var("PATH", "/nowhere-that-exists");

        let result = find_ffmpeg();

        std::env::set_var("PATH", &original);

        match result {
            Ok(path) => {
                assert!(
                    path.is_file(),
                    "find_ffmpeg returned a path that doesn't exist: {}",
                    path.display()
                );
            }
            Err(CaptionError::FfmpegNotFound(msg)) => {
                assert!(
                    msg.contains("FFmpeg is required"),
                    "Expected descriptive error, got: {msg}"
                );
            }
            Err(other) => panic!("Expected FfmpegNotFound, got: {other:?}"),
        }
    }

    #[test]
    #[ignore]
    fn extract_with_ffmpeg_empty_path_returns_error() {
        let original = std::env::var("PATH").unwrap_or_default();
        std::env::set_var("PATH", "/nowhere-that-exists");

        let path = PathBuf::from("/tmp/does-not-matter.webm");
        let result = extract_with_ffmpeg(&path);

        std::env::set_var("PATH", &original);

        assert!(result.is_err(), "Expected an error, got: {result:?}");
        match result.unwrap_err() {
            CaptionError::FfmpegNotFound(_) | CaptionError::ExtractionFailed(_) => {}
            other => panic!("Expected FfmpegNotFound or ExtractionFailed, got: {other:?}"),
        }
    }

    // ---- New test: FFmpeg failure on non-media file ----

    #[test]
    #[ignore] // Requires FFmpeg on PATH
    fn non_media_file_returns_extraction_failed_with_stderr() {
        if !ffmpeg_available() {
            eprintln!("Skipping: FFmpeg not found");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-audio.txt");
        std::fs::write(&path, "This is definitely not a media file.\n").unwrap();

        let result = extract_with_ffmpeg(&path);
        assert!(result.is_err(), "Expected error for non-media file");

        match result.unwrap_err() {
            CaptionError::ExtractionFailed(msg) => {
                assert!(
                    msg.contains("FFmpeg exited with code"),
                    "Expected 'FFmpeg exited with code' in error, got: {msg}"
                );
                let stderr_part = msg.splitn(2, ": ").nth(1).unwrap_or("");
                assert!(
                    !stderr_part.is_empty(),
                    "Expected FFmpeg stderr in error message, got: {msg}"
                );
            }
            other => panic!("Expected ExtractionFailed, got: {other:?}"),
        }
    }

    // ---- New test: Fallback transparency (Symphonia vs FFmpeg same duration) ----

    #[test]
    #[ignore] // Requires FFmpeg on PATH
    fn ffmpeg_and_symphonia_produce_same_duration_for_wav() {
        if !ffmpeg_available() {
            eprintln!("Skipping: FFmpeg not found");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("transparency-test.wav");
        let duration_secs = 2.0_f32;
        let sample_rate = 44100_u32;
        create_test_wav(&wav_path, sample_rate, duration_secs);

        let symphonia_audio =
            super::super::extract::extract_audio(&wav_path).expect("Symphonia should decode WAV");
        let symphonia_duration =
            symphonia_audio.samples.len() as f64 / symphonia_audio.sample_rate as f64;

        let ffmpeg_samples = extract_with_ffmpeg(&wav_path).expect("FFmpeg should decode WAV");
        let ffmpeg_duration = ffmpeg_samples.len() as f64 / 16000.0;

        let diff = (symphonia_duration - ffmpeg_duration).abs();
        let tolerance = symphonia_duration * 0.01;
        assert!(
            diff <= tolerance,
            "Duration mismatch: Symphonia={symphonia_duration:.4}s, \
             FFmpeg={ffmpeg_duration:.4}s, diff={diff:.4}s, tolerance={tolerance:.4}s"
        );

        assert!(
            (symphonia_duration - duration_secs as f64).abs() < 0.05,
            "Symphonia duration {symphonia_duration:.4}s far from expected {duration_secs}s"
        );
        assert!(
            (ffmpeg_duration - duration_secs as f64).abs() < 0.05,
            "FFmpeg duration {ffmpeg_duration:.4}s far from expected {duration_secs}s"
        );

        for &s in &ffmpeg_samples {
            assert!(s.is_finite(), "FFmpeg produced non-finite sample: {s}");
        }
    }

    // ---- New test: FFmpeg integration for WebM/Opus ----

    #[test]
    #[ignore] // Requires FFmpeg on PATH
    fn ffmpeg_decodes_webm_opus_to_16khz_mono() {
        if !ffmpeg_available() {
            eprintln!("Skipping: FFmpeg not found");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let webm_path = dir.path().join("silence.webm");

        let ffmpeg_bin = find_ffmpeg().unwrap();
        let gen_status = std::process::Command::new(&ffmpeg_bin)
            .args([
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-t",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
            ])
            .arg(webm_path.to_str().unwrap())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null())
            .status();

        let gen_status = match gen_status {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Skipping: could not run FFmpeg to generate test file: {e}");
                return;
            }
        };

        if !gen_status.success() {
            eprintln!("Skipping: FFmpeg failed to generate WebM/Opus test file (may lack libopus)");
            return;
        }

        assert!(
            webm_path.exists(),
            "FFmpeg should have created the test WebM file"
        );

        let samples =
            extract_with_ffmpeg(&webm_path).expect("extract_with_ffmpeg should handle WebM/Opus");

        assert!(
            samples.len() >= 14000 && samples.len() <= 18000,
            "Expected ~16000 samples for 1s at 16kHz, got {}",
            samples.len()
        );

        let max_abs = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_abs < 0.01,
            "Expected near-silence, but max absolute sample was {max_abs}"
        );
    }
}
