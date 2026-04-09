use std::fs::File;
use std::path::Path;

use log::debug;
use symphonia::core::audio::{AudioBufferRef, Signal};

use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::ffmpeg_fallback;
use crate::error::CaptionError;

#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn extract_audio(input_path: &Path) -> Result<AudioData, CaptionError> {
    if !input_path.exists() {
        return Err(CaptionError::FileNotFound(input_path.display().to_string()));
    }

    match extract_with_symphonia(input_path) {
        Ok(audio) => {
            debug!("Decoded via Symphonia");
            Ok(audio)
        }
        Err(CaptionError::UnsupportedFormat(ref fmt)) => {
            log::info!("Symphonia can't decode format '{fmt}', trying FFmpeg fallback");
            let samples = ffmpeg_fallback::extract_with_ffmpeg(input_path)?;
            Ok(AudioData {
                samples,
                sample_rate: 16000,
                channels: 1,
            })
        }
        Err(e) => Err(e),
    }
}

fn extract_with_symphonia(input_path: &Path) -> Result<AudioData, CaptionError> {
    let file = File::open(input_path)
        .map_err(|e| CaptionError::ExtractionFailed(format!("failed to open file: {e}")))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = input_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| match e {
            SymphoniaError::Unsupported(msg) => CaptionError::UnsupportedFormat(msg.to_string()),
            other => {
                CaptionError::ExtractionFailed(format!("failed to probe file format: {other}"))
            }
        })?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(CaptionError::NoAudioTrack)?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    debug!(
        "Found audio track: id={}, codec={:?}, sample_rate={:?}, channels={:?}",
        track_id, codec_params.codec, codec_params.sample_rate, codec_params.channels
    );

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| match e {
            SymphoniaError::Unsupported(_) => {
                let codec_name = format!("{:?}", codec_params.codec);
                CaptionError::UnsupportedFormat(codec_name)
            }
            other => CaptionError::ExtractionFailed(format!("failed to create decoder: {other}")),
        })?;

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate: Option<u32> = None;
    let mut channels: Option<u16> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(CaptionError::ExtractionFailed(format!(
                    "failed to read packet: {e}"
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(msg)) => {
                debug!("Skipping corrupt frame: {msg}");
                continue;
            }
            Err(e) => {
                return Err(CaptionError::ExtractionFailed(format!(
                    "decode failed: {e}"
                )));
            }
        };

        let spec = decoded.spec();
        if sample_rate.is_none() {
            sample_rate = Some(spec.rate);
            channels = Some(spec.channels.count() as u16);
            debug!(
                "Decoder reports: sample_rate={}, channels={}",
                spec.rate,
                spec.channels.count()
            );
        }

        append_samples(&decoded, &mut samples);
    }

    let sample_rate = sample_rate.ok_or(CaptionError::NoAudioTrack)?;
    let channels = channels.ok_or(CaptionError::NoAudioTrack)?;

    debug!(
        "Extraction complete: {} samples, {} Hz, {} ch",
        samples.len(),
        sample_rate,
        channels
    );

    Ok(AudioData {
        samples,
        sample_rate,
        channels,
    })
}

fn append_samples(buffer: &AudioBufferRef, output: &mut Vec<f32>) {
    match buffer {
        AudioBufferRef::S16(buf) => interleave_and_convert(buf, output, |s| s as f32 / 32768.0),
        AudioBufferRef::S32(buf) => {
            interleave_and_convert(buf, output, |s| s as f32 / 2_147_483_648.0);
        }
        AudioBufferRef::F32(buf) => interleave_and_convert(buf, output, |s| s),
        AudioBufferRef::F64(buf) => interleave_and_convert(buf, output, |s| s as f32),
        _ => {
            log::warn!("Encountered unsupported sample format, skipping buffer");
        }
    }
}

fn interleave_and_convert<T: symphonia::core::sample::Sample>(
    buf: &symphonia::core::audio::AudioBuffer<T>,
    output: &mut Vec<f32>,
    convert: impl Fn(T) -> f32,
) {
    let num_channels = buf.spec().channels.count();
    let num_frames = buf.frames();
    output.reserve(num_frames * num_channels);
    for frame in 0..num_frames {
        for ch in 0..num_channels {
            output.push(convert(buf.chan(ch)[frame]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn nonexistent_file_returns_file_not_found() {
        let path = PathBuf::from("/tmp/this-file-does-not-exist.wav");
        let result = extract_audio(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            CaptionError::FileNotFound(p) => {
                assert!(p.contains("this-file-does-not-exist"));
            }
            other => panic!("Expected FileNotFound, got: {other:?}"),
        }
    }

    #[test]
    fn empty_file_returns_extraction_or_no_audio_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        std::fs::write(&path, []).unwrap();

        let result = extract_audio(&path);
        assert!(
            matches!(
                result,
                Err(CaptionError::NoAudioTrack)
                    | Err(CaptionError::ExtractionFailed(_))
                    | Err(CaptionError::FfmpegNotFound(_))
            ),
            "Expected NoAudioTrack, ExtractionFailed, or FfmpegNotFound, got: {result:?}"
        );
    }

    #[test]
    fn valid_wav_extracts_successfully() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sine.wav");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..44100 {
            let t = i as f32 / 44100.0;
            let sample = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();

        let audio = extract_audio(&path).expect("extract_audio should succeed");
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        assert!(
            audio.samples.len() >= 44000 && audio.samples.len() <= 44200,
            "Expected ~44100 samples, got {}",
            audio.samples.len()
        );
        for &s in &audio.samples {
            assert!(
                (-1.0..=1.0).contains(&s),
                "Sample {s} out of normalized range"
            );
        }
    }

    #[test]
    fn stereo_wav_extracts_interleaved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stereo.wav");

        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..48000 {
            let t = i as f32 / 48000.0;
            let left = ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 32767.0) as i16;
            let right = ((t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 32767.0) as i16;
            writer.write_sample(left).unwrap();
            writer.write_sample(right).unwrap();
        }
        writer.finalize().unwrap();

        let audio = extract_audio(&path).expect("extract_audio should succeed");
        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels, 2);
        assert!(
            audio.samples.len() >= 95000 && audio.samples.len() <= 97000,
            "Expected ~96000 interleaved samples, got {}",
            audio.samples.len()
        );
    }

    // ---- FFmpeg fallback integration tests ----

    fn ffmpeg_available() -> bool {
        !matches!(
            ffmpeg_fallback::extract_with_ffmpeg(Path::new("/dev/null")),
            Err(CaptionError::FfmpegNotFound(_))
        )
    }

    #[test]
    #[ignore] // Requires FFmpeg on PATH
    fn extract_audio_falls_back_to_ffmpeg_for_webm_opus() {
        if !ffmpeg_available() {
            eprintln!("Skipping: FFmpeg not found");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let webm_path = dir.path().join("fallback-test.webm");

        let ffmpeg_bin = which_ffmpeg().expect("FFmpeg should be available");
        let status = std::process::Command::new(&ffmpeg_bin)
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

        match status {
            Ok(s) if s.success() => {}
            _ => {
                eprintln!("Skipping: FFmpeg could not generate WebM/Opus test file");
                return;
            }
        }

        let audio = extract_audio(&webm_path)
            .expect("extract_audio should handle WebM/Opus via FFmpeg fallback");

        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.channels, 1);

        assert!(
            audio.samples.len() >= 14000 && audio.samples.len() <= 18000,
            "Expected ~16000 samples for 1s at 16kHz, got {}",
            audio.samples.len()
        );
    }

    fn which_ffmpeg() -> Option<std::path::PathBuf> {
        let binary = if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        };
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                let candidate = dir.join(binary);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }
}
