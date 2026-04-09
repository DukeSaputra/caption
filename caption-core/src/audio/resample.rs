use log::debug;
use rubato::{FftFixedIn, Resampler};

use super::extract::AudioData;
use crate::error::CaptionError;

const TARGET_SAMPLE_RATE: u32 = 16_000;

pub fn prepare_for_stt(audio: AudioData) -> Result<Vec<f32>, CaptionError> {
    let channels = audio.channels as usize;
    let source_rate = audio.sample_rate;

    let mut mono = if channels == 1 {
        debug!("Audio is already mono");
        audio.samples
    } else {
        debug!("Converting {} channels to mono", channels);
        mix_to_mono(&audio.samples, channels)?
    };

    remove_dc_offset(&mut mono);
    peak_normalize(&mut mono);
    highpass_filter(&mut mono, 80.0, source_rate);

    if source_rate == TARGET_SAMPLE_RATE {
        debug!(
            "Audio is already at {} Hz, no resampling needed",
            TARGET_SAMPLE_RATE
        );
        return Ok(mono);
    }

    debug!(
        "Resampling from {} Hz to {} Hz ({} input samples)",
        source_rate,
        TARGET_SAMPLE_RATE,
        mono.len()
    );
    resample(&mono, source_rate, TARGET_SAMPLE_RATE)
}

fn mix_to_mono(interleaved: &[f32], channels: usize) -> Result<Vec<f32>, CaptionError> {
    if channels == 0 {
        return Err(CaptionError::ResamplingFailed(
            "channel count is zero".to_string(),
        ));
    }
    if !interleaved.len().is_multiple_of(channels) {
        return Err(CaptionError::ResamplingFailed(format!(
            "sample count {} is not divisible by channel count {}",
            interleaved.len(),
            channels
        )));
    }

    let num_frames = interleaved.len() / channels;
    let divisor = channels as f32;
    let mut mono = Vec::with_capacity(num_frames);

    for frame in 0..num_frames {
        let offset = frame * channels;
        let sum: f32 = interleaved[offset..offset + channels].iter().sum();
        mono.push(sum / divisor);
    }

    Ok(mono)
}

fn remove_dc_offset(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    if mean == 0.0 {
        return;
    }
    debug!("Removing DC offset (mean = {:.6})", mean);
    for s in samples.iter_mut() {
        *s -= mean;
    }
}

fn peak_normalize(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    let peak = samples
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);

    if peak == 0.0 {
        debug!("Audio is silent, skipping normalization");
        return;
    }

    // -1 dBFS = 10^(-1/20)
    let target = 10.0_f32.powf(-1.0 / 20.0);
    let gain = target / peak;

    if (gain - 1.0).abs() < 1e-6 {
        debug!("Audio peak is already at target, skipping normalization");
        return;
    }

    debug!(
        "Peak normalizing: peak = {:.4}, gain = {:.4}x ({:+.1} dB)",
        peak,
        gain,
        20.0 * gain.log10()
    );
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

fn highpass_filter(samples: &mut [f32], cutoff_hz: f32, sample_rate: u32) {
    if samples.is_empty() {
        return;
    }

    let alpha = 1.0 / (1.0 + 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate as f32);
    debug!(
        "Applying high-pass filter: cutoff = {} Hz, sample_rate = {} Hz, alpha = {:.6}",
        cutoff_hz, sample_rate, alpha
    );

    let mut prev_x = samples[0];
    let mut prev_y = samples[0];

    for i in 1..samples.len() {
        let x = samples[i];
        prev_y = alpha * (prev_y + x - prev_x);
        samples[i] = prev_y;
        prev_x = x;
    }
}

fn resample(input: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>, CaptionError> {
    let chunk_size = 1024;
    let num_channels = 1;

    let mut resampler = FftFixedIn::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        chunk_size,
        2, // sub_chunks for better quality
        num_channels,
    )
    .map_err(|e| CaptionError::ResamplingFailed(format!("failed to create resampler: {e}")))?;

    let expected_len =
        (input.len() as f64 * target_rate as f64 / source_rate as f64).round() as usize;
    let mut output = Vec::with_capacity(expected_len + chunk_size);

    let mut pos = 0;

    while pos + resampler.input_frames_next() <= input.len() {
        let frames_needed = resampler.input_frames_next();
        let chunk = vec![input[pos..pos + frames_needed].to_vec()];
        let resampled = resampler
            .process(&chunk, None)
            .map_err(|e| CaptionError::ResamplingFailed(format!("resampling failed: {e}")))?;
        output.extend_from_slice(&resampled[0]);
        pos += frames_needed;
    }

    if pos < input.len() {
        let frames_needed = resampler.input_frames_next();
        let remaining = input.len() - pos;
        let mut padded = vec![0.0f32; frames_needed];
        padded[..remaining].copy_from_slice(&input[pos..]);
        let chunk = vec![padded];
        let resampled = resampler.process(&chunk, None).map_err(|e| {
            CaptionError::ResamplingFailed(format!("resampling failed on final chunk: {e}"))
        })?;
        output.extend_from_slice(&resampled[0]);
    }

    output.truncate(expected_len);

    debug!(
        "Resampling complete: {} input samples -> {} output samples",
        input.len(),
        output.len()
    );

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_16khz_skips_resampling() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio = AudioData {
            samples: samples.clone(),
            sample_rate: 16_000,
            channels: 1,
        };

        let result = prepare_for_stt(audio).unwrap();
        assert_eq!(
            result.len(),
            samples.len(),
            "16 kHz mono should not change sample count"
        );
    }

    #[test]
    fn stereo_to_mono_averages_channels() {
        let interleaved = vec![0.4, 0.6, 0.2, 0.8, -0.5, 0.5];
        let mono = mix_to_mono(&interleaved, 2).unwrap();

        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < f32::EPSILON);
        assert!((mono[1] - 0.5).abs() < f32::EPSILON);
        assert!((mono[2] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn three_channel_to_mono_averages_equally() {
        let interleaved = vec![0.3, 0.6, 0.9, 0.0, 0.0, 0.0];
        let mono = mix_to_mono(&interleaved, 3).unwrap();

        assert_eq!(mono.len(), 2);
        assert!(
            (mono[0] - 0.6).abs() < 1e-6,
            "Expected 0.6, got {}",
            mono[0]
        );
        assert!((mono[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_channel_count_errors() {
        let result = mix_to_mono(&[0.1, 0.2, 0.3], 2);
        assert!(result.is_err(), "3 samples with 2 channels should error");
    }

    #[test]
    fn zero_channels_errors() {
        let result = mix_to_mono(&[0.1], 0);
        assert!(result.is_err(), "0 channels should error");
    }

    #[test]
    fn resampling_44100_to_16000_produces_correct_length() {
        let source_rate = 44100;
        let duration_secs = 1.0;
        let num_samples = (source_rate as f64 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();

        let audio = AudioData {
            samples,
            sample_rate: source_rate,
            channels: 1,
        };

        let result = prepare_for_stt(audio).unwrap();

        let expected = (duration_secs * 16000.0).round() as usize;
        let tolerance = 50;
        assert!(
            result.len().abs_diff(expected) <= tolerance,
            "Expected ~{expected} samples, got {}",
            result.len()
        );
    }

    #[test]
    fn stereo_44100_to_16000_mono() {
        let source_rate = 44100;
        let num_frames = 44100;
        let mut interleaved = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            let t = i as f32 / source_rate as f32;
            let sample = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            interleaved.push(sample);
            interleaved.push(sample);
        }

        let audio = AudioData {
            samples: interleaved,
            sample_rate: source_rate,
            channels: 2,
        };

        let result = prepare_for_stt(audio).unwrap();

        let expected = 16000;
        let tolerance = 50;
        assert!(
            result.len().abs_diff(expected) <= tolerance,
            "Expected ~{expected} samples, got {}",
            result.len()
        );
    }

    #[test]
    fn resampling_48000_to_16000() {
        let source_rate = 48000;
        let num_samples = 48000;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();

        let audio = AudioData {
            samples,
            sample_rate: source_rate,
            channels: 1,
        };

        let result = prepare_for_stt(audio).unwrap();

        let expected = 16000;
        let tolerance = 100;
        assert!(
            result.len().abs_diff(expected) <= tolerance,
            "Expected ~{expected} samples, got {}",
            result.len()
        );
    }

    // ── DC offset removal ────────────────────────────────────────────

    #[test]
    fn dc_offset_removal_centers_around_zero() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        remove_dc_offset(&mut samples);

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            mean.abs() < 1e-6,
            "Mean after DC removal should be ~0, got {mean}"
        );
    }

    #[test]
    fn dc_offset_removal_preserves_relative_differences() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original_diff = samples[1] - samples[0];
        remove_dc_offset(&mut samples);

        let new_diff = samples[1] - samples[0];
        assert!(
            (new_diff - original_diff).abs() < 1e-6,
            "Relative differences should be preserved"
        );
    }

    #[test]
    fn dc_offset_noop_when_already_centered() {
        let mut samples = vec![-1.0, 0.0, 1.0];
        let original = samples.clone();
        remove_dc_offset(&mut samples);

        assert_eq!(
            samples, original,
            "Already-centered audio should not change"
        );
    }

    #[test]
    fn dc_offset_empty_input() {
        let mut samples: Vec<f32> = vec![];
        remove_dc_offset(&mut samples);
        assert!(samples.is_empty());
    }

    // ── Peak normalization ───────────────────────────────────────────

    #[test]
    fn peak_normalize_scales_to_target() {
        let mut samples = vec![0.0, 0.25, -0.5, 0.1];
        peak_normalize(&mut samples);

        let peak = samples
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);
        let target = 10.0_f32.powf(-1.0 / 20.0); // ~0.891
        assert!(
            (peak - target).abs() < 1e-4,
            "Peak should be ~{target:.3}, got {peak:.4}"
        );
    }

    #[test]
    fn peak_normalize_silent_audio_unchanged() {
        let mut samples = vec![0.0, 0.0, 0.0];
        let original = samples.clone();
        peak_normalize(&mut samples);

        assert_eq!(
            samples, original,
            "Silent audio should not be modified by normalization"
        );
    }

    #[test]
    fn peak_normalize_already_at_target() {
        let target = 10.0_f32.powf(-1.0 / 20.0);
        let mut samples = vec![0.0, target, -target / 2.0];
        let original = samples.clone();
        peak_normalize(&mut samples);

        for (i, (a, b)) in samples.iter().zip(original.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "Sample {i} changed: {b} -> {a}");
        }
    }

    #[test]
    fn peak_normalize_preserves_polarity() {
        let mut samples = vec![0.1, -0.2, 0.3, -0.4];
        peak_normalize(&mut samples);

        assert!(samples[0] > 0.0, "Positive sample should stay positive");
        assert!(samples[1] < 0.0, "Negative sample should stay negative");
        assert!(samples[2] > 0.0, "Positive sample should stay positive");
        assert!(samples[3] < 0.0, "Negative sample should stay negative");
    }

    #[test]
    fn peak_normalize_quiet_recording_gets_boosted() {
        // -30 dBFS peak
        let quiet_peak = 10.0_f32.powf(-30.0 / 20.0); // ~0.0316
        let mut samples = vec![0.0, quiet_peak, -quiet_peak / 2.0, quiet_peak * 0.8];
        peak_normalize(&mut samples);

        let new_peak = samples
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);
        let target = 10.0_f32.powf(-1.0 / 20.0);
        assert!(
            (new_peak - target).abs() < 1e-4,
            "Quiet audio should be boosted to ~{target:.3}, got {new_peak:.4}"
        );
    }

    #[test]
    fn peak_normalize_empty_input() {
        let mut samples: Vec<f32> = vec![];
        peak_normalize(&mut samples);
        assert!(samples.is_empty());
    }

    // ── High-pass filter ─────────────────────────────────────────────

    #[test]
    fn highpass_attenuates_low_frequency() {
        let sample_rate = 44100_u32;
        let freq = 30.0_f32;
        let duration_samples = sample_rate as usize;
        let mut samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * freq * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();

        let original_rms = rms(&samples);
        highpass_filter(&mut samples, 80.0, sample_rate);
        let filtered_rms = rms(&samples);

        assert!(
            filtered_rms < original_rms * 0.6,
            "30 Hz signal should be significantly attenuated: original RMS = {original_rms:.4}, filtered RMS = {filtered_rms:.4}"
        );
    }

    #[test]
    fn highpass_preserves_speech_band() {
        let sample_rate = 44100_u32;
        let freq = 1000.0_f32;
        let duration_samples = sample_rate as usize;
        let mut samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * freq * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();

        let original_rms = rms(&samples);
        highpass_filter(&mut samples, 80.0, sample_rate);
        let filtered_rms = rms(&samples);

        let ratio = filtered_rms / original_rms;
        assert!(
            ratio > 0.95,
            "1000 Hz signal should be mostly preserved: ratio = {ratio:.4}"
        );
    }

    #[test]
    fn highpass_very_low_frequency_strongly_attenuated() {
        let sample_rate = 44100_u32;
        let freq = 10.0_f32;
        let duration_samples = sample_rate as usize;
        let mut samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * freq * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();

        let original_rms = rms(&samples);
        highpass_filter(&mut samples, 80.0, sample_rate);
        let filtered_rms = rms(&samples);

        assert!(
            filtered_rms < original_rms * 0.3,
            "10 Hz signal should be strongly attenuated: original = {original_rms:.4}, filtered = {filtered_rms:.4}"
        );
    }

    #[test]
    fn highpass_empty_input() {
        let mut samples: Vec<f32> = vec![];
        highpass_filter(&mut samples, 80.0, 44100);
        assert!(samples.is_empty());
    }

    #[test]
    fn highpass_single_sample() {
        let mut samples = vec![0.5];
        highpass_filter(&mut samples, 80.0, 44100);
        assert_eq!(samples.len(), 1);
    }

    // ── Integration: full pipeline output length ─────────────────────

    #[test]
    fn prepare_for_stt_with_conditioning_produces_correct_length() {
        let source_rate = 44100;
        let num_samples = 44100;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin()
            })
            .collect();

        let audio = AudioData {
            samples,
            sample_rate: source_rate,
            channels: 1,
        };

        let result = prepare_for_stt(audio).unwrap();

        let expected = 16000;
        let tolerance = 50;
        assert!(
            result.len().abs_diff(expected) <= tolerance,
            "Expected ~{expected} samples after full pipeline, got {}",
            result.len()
        );
    }

    #[test]
    fn prepare_for_stt_conditioning_order_matters() {
        let source_rate = 16_000;
        let num_samples = 16000;
        let dc_offset = 0.5;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.1 + dc_offset
            })
            .collect();

        let audio = AudioData {
            samples,
            sample_rate: source_rate,
            channels: 1,
        };

        let result = prepare_for_stt(audio).unwrap();

        let mean = result.iter().sum::<f32>() / result.len() as f32;
        assert!(
            mean.abs() < 0.05,
            "DC offset should be removed: mean = {mean:.4}"
        );
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }
}
