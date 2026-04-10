use std::path::Path;

use log::warn;
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;

use crate::error::CaptionError;

// ---------------------------------------------------------------------------
// Silero VAD v5 constants
// ---------------------------------------------------------------------------

const SAMPLE_RATE: u32 = 16_000;

// 512 samples = 32ms at 16 kHz
const FRAME_SIZE: usize = 512;

const SECONDS_PER_FRAME: f64 = FRAME_SIZE as f64 / SAMPLE_RATE as f64;

// ---------------------------------------------------------------------------
// Default integration parameters
// ---------------------------------------------------------------------------

const DEFAULT_THRESHOLD: f32 = 0.45;

const DEFAULT_MIN_SPEECH_SECONDS: f64 = 0.100;

const DEFAULT_MIN_SILENCE_SECONDS: f64 = 2.0;

const DEFAULT_SPEECH_PAD_SECONDS: f64 = 0.200;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_speech_seconds: f64,
    pub min_silence_seconds: f64,
    pub speech_pad_seconds: f64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            min_speech_seconds: DEFAULT_MIN_SPEECH_SECONDS,
            min_silence_seconds: DEFAULT_MIN_SILENCE_SECONDS,
            speech_pad_seconds: DEFAULT_SPEECH_PAD_SECONDS,
        }
    }
}

pub struct StandaloneVad {
    session: Session,
}

impl StandaloneVad {
    pub fn new(model_path: &Path) -> Result<Self, CaptionError> {
        let session = Session::builder()
            .map_err(|e| CaptionError::ModelLoadFailed(format!("ORT session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                CaptionError::ModelLoadFailed(format!(
                    "Failed to load VAD model from {}: {e}",
                    model_path.display()
                ))
            })?;

        Ok(Self { session })
    }

    pub fn detect_speech(&mut self, audio: &[f32]) -> Result<Vec<SpeechSegment>, CaptionError> {
        self.detect_speech_with_config(audio, &VadConfig::default())
    }

    pub fn detect_speech_with_config(
        &mut self,
        audio: &[f32],
        config: &VadConfig,
    ) -> Result<Vec<SpeechSegment>, CaptionError> {
        if audio.is_empty() {
            warn!("VAD received empty audio, returning no speech segments");
            return Ok(Vec::new());
        }

        let probabilities = self.compute_frame_probabilities(audio)?;
        let raw_segments = threshold_segments(&probabilities, config.threshold);
        let merged = merge_close_segments(raw_segments, config.min_silence_seconds);
        let filtered = filter_short_segments(merged, config.min_speech_seconds);
        let audio_duration = audio.len() as f64 / f64::from(SAMPLE_RATE);
        let padded = pad_segments(filtered, config.speech_pad_seconds, audio_duration);

        if padded.is_empty() {
            warn!("VAD detected zero speech segments in {audio_duration:.2}s of audio");
        }

        Ok(padded)
    }

    fn compute_frame_probabilities(&mut self, audio: &[f32]) -> Result<Vec<f32>, CaptionError> {
        let num_frames = audio.len() / FRAME_SIZE;
        if num_frames == 0 {
            return Ok(Vec::new());
        }

        let mut probabilities = Vec::with_capacity(num_frames);

        // Silero VAD v5 state tensor: shape [2, 1, 128] = 256 elements
        const STATE_DIM: usize = 128;
        const STATE_LEN: usize = 2 * STATE_DIM;
        let mut state_data = vec![0.0_f32; STATE_LEN];

        let sr_data = [i64::from(SAMPLE_RATE)];

        for frame_idx in 0..num_frames {
            let frame_start = frame_idx * FRAME_SIZE;
            let frame_end = frame_start + FRAME_SIZE;
            let frame_data = &audio[frame_start..frame_end];

            let input_tensor = TensorRef::from_array_view(([1_usize, FRAME_SIZE], frame_data))
                .map_err(|e| CaptionError::VadProcessingFailed(format!("input tensor: {e}")))?;
            let sr_tensor = TensorRef::from_array_view(([1_usize], sr_data.as_slice()))
                .map_err(|e| CaptionError::VadProcessingFailed(format!("sr tensor: {e}")))?;
            let state_tensor =
                TensorRef::from_array_view(([2_usize, 1, STATE_DIM], state_data.as_slice()))
                    .map_err(|e| CaptionError::VadProcessingFailed(format!("state tensor: {e}")))?;

            let outputs = self
                .session
                .run(inputs![
                    "input" => input_tensor,
                    "sr" => sr_tensor,
                    "state" => state_tensor,
                ])
                .map_err(|e| {
                    CaptionError::VadProcessingFailed(format!(
                        "Inference failed at frame {frame_idx}: {e}"
                    ))
                })?;

            let (_, prob_data) = outputs["output"].try_extract_tensor::<f32>().map_err(|e| {
                CaptionError::VadProcessingFailed(format!("Failed to extract output: {e}"))
            })?;
            let prob = prob_data.first().copied().ok_or_else(|| {
                CaptionError::VadProcessingFailed("Empty output tensor".to_string())
            })?;
            probabilities.push(prob);

            let (_, new_state) = outputs["stateN"].try_extract_tensor::<f32>().map_err(|e| {
                CaptionError::VadProcessingFailed(format!("Failed to extract stateN: {e}"))
            })?;
            state_data.copy_from_slice(new_state);
        }

        Ok(probabilities)
    }
}

// ---------------------------------------------------------------------------
// Post-processing: threshold, merge, filter, pad
// ---------------------------------------------------------------------------

fn threshold_segments(probabilities: &[f32], threshold: f32) -> Vec<SpeechSegment> {
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut segment_start_frame: usize = 0;

    for (i, &prob) in probabilities.iter().enumerate() {
        if prob >= threshold && !in_speech {
            in_speech = true;
            segment_start_frame = i;
        } else if prob < threshold && in_speech {
            in_speech = false;
            segments.push(SpeechSegment {
                start: segment_start_frame as f64 * SECONDS_PER_FRAME,
                end: i as f64 * SECONDS_PER_FRAME,
            });
        }
    }

    if in_speech {
        segments.push(SpeechSegment {
            start: segment_start_frame as f64 * SECONDS_PER_FRAME,
            end: probabilities.len() as f64 * SECONDS_PER_FRAME,
        });
    }

    segments
}

fn merge_close_segments(
    segments: Vec<SpeechSegment>,
    min_silence_seconds: f64,
) -> Vec<SpeechSegment> {
    if segments.is_empty() {
        return segments;
    }

    let mut merged: Vec<SpeechSegment> = Vec::with_capacity(segments.len());
    let mut current = segments[0].clone();

    for seg in segments.iter().skip(1) {
        let gap = seg.start - current.end;
        if gap <= min_silence_seconds {
            current.end = seg.end;
        } else {
            merged.push(current);
            current = seg.clone();
        }
    }
    merged.push(current);

    merged
}

fn filter_short_segments(
    segments: Vec<SpeechSegment>,
    min_speech_seconds: f64,
) -> Vec<SpeechSegment> {
    segments
        .into_iter()
        .filter(|seg| (seg.end - seg.start) >= min_speech_seconds)
        .collect()
}

fn pad_segments(
    segments: Vec<SpeechSegment>,
    pad_seconds: f64,
    audio_duration: f64,
) -> Vec<SpeechSegment> {
    if segments.is_empty() {
        return segments;
    }

    let padded: Vec<SpeechSegment> = segments
        .into_iter()
        .map(|seg| SpeechSegment {
            start: (seg.start - pad_seconds).max(0.0),
            end: (seg.end + pad_seconds).min(audio_duration),
        })
        .collect();

    merge_close_segments(padded, 0.0)
}

// ---------------------------------------------------------------------------
// Chunk extraction
// ---------------------------------------------------------------------------

pub fn extract_chunks(
    audio: &[f32],
    sample_rate: u32,
    segments: &[SpeechSegment],
    padding_seconds: f64,
    max_chunk_seconds: f64,
) -> Vec<(Vec<f32>, f64)> {
    if segments.is_empty() || audio.is_empty() {
        return Vec::new();
    }

    let sr = f64::from(sample_rate);
    let audio_duration = audio.len() as f64 / sr;

    let padded: Vec<SpeechSegment> = segments
        .iter()
        .map(|seg| SpeechSegment {
            start: (seg.start - padding_seconds).max(0.0),
            end: (seg.end + padding_seconds).min(audio_duration),
        })
        .collect();

    let merged = merge_close_segments(padded, 0.0);

    let final_segments = split_long_segments(merged, max_chunk_seconds);

    final_segments
        .into_iter()
        .map(|seg| {
            let start_sample = (seg.start * sr).round() as usize;
            let end_sample = ((seg.end * sr).round() as usize).min(audio.len());
            let chunk = audio[start_sample..end_sample].to_vec();
            (chunk, seg.start)
        })
        .collect()
}

fn split_long_segments(segments: Vec<SpeechSegment>, max_seconds: f64) -> Vec<SpeechSegment> {
    let mut result = Vec::with_capacity(segments.len());

    for seg in segments {
        let duration = seg.end - seg.start;
        if duration <= max_seconds {
            result.push(seg);
        } else {
            let num_pieces = (duration / max_seconds).ceil() as usize;
            let piece_duration = duration / num_pieces as f64;
            for i in 0..num_pieces {
                let start = seg.start + i as f64 * piece_duration;
                let end = if i == num_pieces - 1 {
                    seg.end
                } else {
                    start + piece_duration
                };
                result.push(SpeechSegment { start, end });
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SpeechSegment tests
    // -----------------------------------------------------------------------

    #[test]
    fn speech_segment_clone_and_debug() {
        let seg = SpeechSegment {
            start: 1.0,
            end: 2.5,
        };
        let cloned = seg.clone();
        assert_eq!(seg, cloned);
        let debug = format!("{seg:?}");
        assert!(debug.contains("SpeechSegment"));
    }

    // -----------------------------------------------------------------------
    // VadConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn vad_config_defaults() {
        let config = VadConfig::default();
        assert!((config.threshold - 0.45).abs() < f32::EPSILON);
        assert!((config.min_speech_seconds - 0.100).abs() < f64::EPSILON);
        assert!((config.min_silence_seconds - 2.0).abs() < f64::EPSILON);
        assert!((config.speech_pad_seconds - 0.200).abs() < f64::EPSILON);
    }

    #[test]
    fn vad_config_custom() {
        let config = VadConfig {
            threshold: 0.7,
            min_speech_seconds: 0.5,
            min_silence_seconds: 0.2,
            speech_pad_seconds: 0.05,
        };
        assert!((config.threshold - 0.7).abs() < f32::EPSILON);
        assert!((config.min_speech_seconds - 0.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // threshold_segments tests
    // -----------------------------------------------------------------------

    #[test]
    fn threshold_empty_probabilities() {
        let segments = threshold_segments(&[], 0.5);
        assert!(segments.is_empty());
    }

    #[test]
    fn threshold_all_below() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let segments = threshold_segments(&probs, 0.5);
        assert!(segments.is_empty());
    }

    #[test]
    fn threshold_all_above() {
        let probs = vec![0.6, 0.7, 0.8, 0.9];
        let segments = threshold_segments(&probs, 0.5);
        assert_eq!(segments.len(), 1);
        assert!((segments[0].start - 0.0).abs() < f64::EPSILON);
        assert!((segments[0].end - 4.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
    }

    #[test]
    fn threshold_single_speech_region() {
        let probs = vec![0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.1];
        let segments = threshold_segments(&probs, 0.5);
        assert_eq!(segments.len(), 1);
        assert!((segments[0].start - 2.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((segments[0].end - 5.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
    }

    #[test]
    fn threshold_two_speech_regions() {
        let probs = vec![0.1, 0.6, 0.7, 0.1, 0.1, 0.8, 0.9, 0.1];
        let segments = threshold_segments(&probs, 0.5);
        assert_eq!(segments.len(), 2);
        assert!((segments[0].start - 1.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((segments[0].end - 3.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((segments[1].start - 5.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((segments[1].end - 7.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
    }

    #[test]
    fn threshold_speech_at_end() {
        let probs = vec![0.1, 0.6, 0.7, 0.8];
        let segments = threshold_segments(&probs, 0.5);
        assert_eq!(segments.len(), 1);
        assert!((segments[0].end - 4.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
    }

    #[test]
    fn threshold_exact_boundary() {
        let probs = vec![0.5, 0.5, 0.5];
        let segments = threshold_segments(&probs, 0.5);
        assert_eq!(segments.len(), 1);
    }

    // -----------------------------------------------------------------------
    // merge_close_segments tests
    // -----------------------------------------------------------------------

    #[test]
    fn merge_empty() {
        let result = merge_close_segments(Vec::new(), 0.1);
        assert!(result.is_empty());
    }

    #[test]
    fn merge_single_segment() {
        let segments = vec![SpeechSegment {
            start: 1.0,
            end: 2.0,
        }];
        let result = merge_close_segments(segments, 0.1);
        assert_eq!(result.len(), 1);
        assert!((result[0].start - 1.0).abs() < f64::EPSILON);
        assert!((result[0].end - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_close_gap() {
        let segments = vec![
            SpeechSegment {
                start: 1.0,
                end: 2.0,
            },
            SpeechSegment {
                start: 2.05,
                end: 3.0,
            },
        ];
        let result = merge_close_segments(segments, 0.1);
        assert_eq!(result.len(), 1);
        assert!((result[0].start - 1.0).abs() < f64::EPSILON);
        assert!((result[0].end - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_wide_gap() {
        let segments = vec![
            SpeechSegment {
                start: 1.0,
                end: 2.0,
            },
            SpeechSegment {
                start: 3.0,
                end: 4.0,
            },
        ];
        let result = merge_close_segments(segments, 0.1);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_multiple_close() {
        let segments = vec![
            SpeechSegment {
                start: 0.0,
                end: 1.0,
            },
            SpeechSegment {
                start: 1.05,
                end: 2.0,
            },
            SpeechSegment {
                start: 2.08,
                end: 3.0,
            },
        ];
        let result = merge_close_segments(segments, 0.1);
        assert_eq!(result.len(), 1);
        assert!((result[0].start - 0.0).abs() < f64::EPSILON);
        assert!((result[0].end - 3.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // filter_short_segments tests
    // -----------------------------------------------------------------------

    #[test]
    fn filter_removes_short() {
        let segments = vec![
            SpeechSegment {
                start: 0.0,
                end: 0.1,
            },
            SpeechSegment {
                start: 1.0,
                end: 2.0,
            },
        ];
        let result = filter_short_segments(segments, 0.250);
        assert_eq!(result.len(), 1);
        assert!((result[0].start - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn filter_keeps_exact_minimum() {
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 0.250,
        }];
        let result = filter_short_segments(segments, 0.250);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn filter_all_short() {
        let segments = vec![
            SpeechSegment {
                start: 0.0,
                end: 0.1,
            },
            SpeechSegment {
                start: 1.0,
                end: 1.2,
            },
        ];
        let result = filter_short_segments(segments, 0.250);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // pad_segments tests
    // -----------------------------------------------------------------------

    #[test]
    fn pad_empty() {
        let result = pad_segments(Vec::new(), 0.1, 10.0);
        assert!(result.is_empty());
    }

    #[test]
    fn pad_clamps_to_zero() {
        let segments = vec![SpeechSegment {
            start: 0.05,
            end: 1.0,
        }];
        let result = pad_segments(segments, 0.1, 10.0);
        assert_eq!(result.len(), 1);
        assert!((result[0].start - 0.0).abs() < f64::EPSILON);
        assert!((result[0].end - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn pad_clamps_to_duration() {
        let segments = vec![SpeechSegment {
            start: 9.0,
            end: 9.95,
        }];
        let result = pad_segments(segments, 0.1, 10.0);
        assert_eq!(result.len(), 1);
        assert!((result[0].end - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pad_merges_overlapping() {
        let segments = vec![
            SpeechSegment {
                start: 1.0,
                end: 2.0,
            },
            SpeechSegment {
                start: 2.15,
                end: 3.0,
            },
        ];
        let result = pad_segments(segments, 0.1, 10.0);
        assert_eq!(result.len(), 1);
        assert!((result[0].start - 0.9).abs() < f64::EPSILON);
        assert!((result[0].end - 3.1).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // split_long_segments tests
    // -----------------------------------------------------------------------

    #[test]
    fn split_short_segment_unchanged() {
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 10.0,
        }];
        let result = split_long_segments(segments, 30.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn split_exact_max() {
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 30.0,
        }];
        let result = split_long_segments(segments, 30.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn split_just_over_max() {
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 31.0,
        }];
        let result = split_long_segments(segments, 30.0);
        assert_eq!(result.len(), 2);
        assert!((result[0].start - 0.0).abs() < f64::EPSILON);
        assert!((result[1].end - 31.0).abs() < f64::EPSILON);
        let dur0 = result[0].end - result[0].start;
        let dur1 = result[1].end - result[1].start;
        assert!((dur0 - dur1).abs() < 0.01);
    }

    #[test]
    fn split_three_pieces() {
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 90.0,
        }];
        let result = split_long_segments(segments, 30.0);
        assert_eq!(result.len(), 3);
        assert!((result[0].end - 30.0).abs() < f64::EPSILON);
        assert!((result[1].start - 30.0).abs() < f64::EPSILON);
        assert!((result[1].end - 60.0).abs() < f64::EPSILON);
        assert!((result[2].start - 60.0).abs() < f64::EPSILON);
        assert!((result[2].end - 90.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // extract_chunks tests
    // -----------------------------------------------------------------------

    #[test]
    fn extract_chunks_empty_segments() {
        let audio = vec![0.0_f32; 16_000];
        let chunks = extract_chunks(&audio, 16_000, &[], 0.1, 30.0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn extract_chunks_empty_audio() {
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 1.0,
        }];
        let chunks = extract_chunks(&[], 16_000, &segments, 0.1, 30.0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn extract_chunks_single_segment() {
        let audio = vec![0.5_f32; 48_000];
        let segments = vec![SpeechSegment {
            start: 1.0,
            end: 2.0,
        }];
        let chunks = extract_chunks(&audio, 16_000, &segments, 0.1, 30.0);
        assert_eq!(chunks.len(), 1);

        let (chunk, start) = &chunks[0];
        assert!((start - 0.9).abs() < 0.01);
        let expected_samples = ((2.1_f64 - 0.9) * 16000.0).round() as usize;
        assert!(
            (chunk.len() as i64 - expected_samples as i64).unsigned_abs() <= 1,
            "Expected ~{expected_samples} samples, got {}",
            chunk.len()
        );
    }

    #[test]
    fn extract_chunks_clamps_to_boundaries() {
        let audio = vec![0.5_f32; 32_000];
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 2.0,
        }];
        let chunks = extract_chunks(&audio, 16_000, &segments, 0.5, 30.0);
        assert_eq!(chunks.len(), 1);
        let (chunk, start) = &chunks[0];
        assert!((start - 0.0).abs() < f64::EPSILON);
        assert_eq!(chunk.len(), 32_000);
    }

    #[test]
    fn extract_chunks_splits_long() {
        let audio = vec![0.5_f32; 90 * 16_000];
        let segments = vec![SpeechSegment {
            start: 0.0,
            end: 90.0,
        }];
        let chunks = extract_chunks(&audio, 16_000, &segments, 0.0, 30.0);
        assert_eq!(chunks.len(), 3);

        for (chunk, _) in &chunks {
            let duration = chunk.len() as f64 / 16_000.0;
            assert!(
                (duration - 30.0).abs() < 0.1,
                "Expected ~30s chunk, got {duration:.2}s"
            );
        }
    }

    #[test]
    fn extract_chunks_merges_close_with_padding() {
        let audio = vec![0.5_f32; 48_000];
        let segments = vec![
            SpeechSegment {
                start: 0.5,
                end: 1.0,
            },
            SpeechSegment {
                start: 1.1,
                end: 1.5,
            },
        ];
        let chunks = extract_chunks(&audio, 16_000, &segments, 0.1, 30.0);
        assert_eq!(chunks.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Full pipeline integration test (no model needed)
    // -----------------------------------------------------------------------

    #[test]
    fn full_pipeline_post_processing() {
        let config = VadConfig::default();

        let mut probs = vec![0.1_f32; 60];
        for p in probs.iter_mut().skip(10).take(16) {
            *p = 0.8;
        }
        for p in probs.iter_mut().skip(29).take(22) {
            *p = 0.9;
        }

        let raw = threshold_segments(&probs, config.threshold);
        assert_eq!(raw.len(), 2, "Should detect 2 raw speech regions");

        let merged = merge_close_segments(raw, config.min_silence_seconds);
        assert_eq!(merged.len(), 1, "Regions should merge (gap < min_silence)");

        let filtered = filter_short_segments(merged, config.min_speech_seconds);
        assert_eq!(filtered.len(), 1, "Merged segment should pass min duration");

        let audio_duration = 60.0 * SECONDS_PER_FRAME;
        let padded = pad_segments(filtered, config.speech_pad_seconds, audio_duration);
        assert_eq!(padded.len(), 1);
        let expected_start = (10.0 * SECONDS_PER_FRAME - config.speech_pad_seconds).max(0.0);
        assert!(
            (padded[0].start - expected_start).abs() < f64::EPSILON,
            "Expected start {expected_start}, got {}",
            padded[0].start
        );
    }

    #[test]
    fn full_pipeline_separate_regions() {
        let config = VadConfig {
            min_silence_seconds: 0.100,
            ..VadConfig::default()
        };

        let mut probs = vec![0.1_f32; 100];
        for p in probs.iter_mut().skip(5).take(11) {
            *p = 0.8;
        }
        for p in probs.iter_mut().skip(31).take(15) {
            *p = 0.9;
        }

        let raw = threshold_segments(&probs, config.threshold);
        assert_eq!(raw.len(), 2);

        let merged = merge_close_segments(raw, config.min_silence_seconds);
        assert_eq!(merged.len(), 2, "Wide gap should keep segments separate");

        let filtered = filter_short_segments(merged, config.min_speech_seconds);
        assert_eq!(filtered.len(), 2, "Both segments exceed min duration");
    }

    #[test]
    fn zero_speech_returns_empty() {
        let probs = vec![0.1_f32; 100];
        let config = VadConfig::default();

        let raw = threshold_segments(&probs, config.threshold);
        assert!(raw.is_empty());

        let merged = merge_close_segments(raw, config.min_silence_seconds);
        assert!(merged.is_empty());

        let filtered = filter_short_segments(merged, config.min_speech_seconds);
        assert!(filtered.is_empty());

        let padded = pad_segments(filtered, config.speech_pad_seconds, 10.0);
        assert!(padded.is_empty());
    }

    #[test]
    fn only_short_speech_is_discarded() {
        let mut probs = vec![0.1_f32; 100];
        // 3 frames of speech = 96ms, below 100ms minimum
        probs[10] = 0.8;
        probs[11] = 0.8;
        probs[12] = 0.8;

        let config = VadConfig::default();
        let raw = threshold_segments(&probs, config.threshold);
        assert_eq!(raw.len(), 1);

        let merged = merge_close_segments(raw, config.min_silence_seconds);
        assert_eq!(merged.len(), 1);

        let filtered = filter_short_segments(merged, config.min_speech_seconds);
        assert!(filtered.is_empty(), "96ms segment should be discarded");
    }

    // -----------------------------------------------------------------------
    // Constants sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn constants_are_correct() {
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(FRAME_SIZE, 512);
        assert!((SECONDS_PER_FRAME - 0.032).abs() < 1e-6);
    }
}
