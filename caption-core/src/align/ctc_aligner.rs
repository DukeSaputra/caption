use std::path::Path;

use log::debug;
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;

use super::{
    build_char_to_idx, ctc_viterbi_align, group_into_words, words_to_char_sequence, ForcedAligner,
    WordAlignment, VOCAB_SIZE,
};
use crate::error::CaptionError;

pub const DEFAULT_ALIGNER_MODEL_FILENAME: &str = "wav2vec2-base-960h.onnx";

pub struct CtcAligner {
    session: Session,
    char_to_idx: Vec<Option<usize>>,
}

impl CtcAligner {
    pub fn new(model_path: &Path) -> Result<Self, CaptionError> {
        debug!("Loading CTC aligner model from: {}", model_path.display());

        let session = Session::builder()
            .map_err(|e| CaptionError::ModelLoadFailed(format!("ORT session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                CaptionError::ModelLoadFailed(format!(
                    "Failed to load aligner model from {}: {e}",
                    model_path.display()
                ))
            })?;

        let char_to_idx = build_char_to_idx();

        debug!("CTC aligner model loaded successfully");

        Ok(Self {
            session,
            char_to_idx,
        })
    }

    fn compute_log_probs(&mut self, audio: &[f32]) -> Result<(Vec<f32>, usize), CaptionError> {
        let num_samples = audio.len();

        let input_tensor =
            TensorRef::from_array_view(([1_usize, num_samples], audio)).map_err(|e| {
                CaptionError::AlignmentFailed(format!("Failed to create input tensor: {e}"))
            })?;

        let outputs = self
            .session
            .run(inputs!["input_values" => input_tensor])
            .map_err(|e| {
                CaptionError::AlignmentFailed(format!("wav2vec2 inference failed: {e}"))
            })?;

        // Output "logits" has shape [1, T, V].
        let (shape, raw_logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| CaptionError::AlignmentFailed(format!("Failed to extract logits: {e}")))?;

        let to_usize = |val: i64, name: &str| -> Result<usize, CaptionError> {
            usize::try_from(val).map_err(|_| {
                CaptionError::AlignmentFailed(format!("Invalid {name} dimension: {val}"))
            })
        };

        let num_frames = if shape.len() == 3 {
            to_usize(shape[1], "frames")?
        } else if shape.len() == 2 {
            to_usize(shape[0], "frames")?
        } else {
            return Err(CaptionError::AlignmentFailed(format!(
                "Unexpected logits shape: {shape:?}"
            )));
        };

        let actual_vocab = if shape.len() == 3 {
            to_usize(shape[2], "vocab")?
        } else {
            to_usize(shape[1], "vocab")?
        };

        if actual_vocab != VOCAB_SIZE {
            return Err(CaptionError::AlignmentFailed(format!(
                "Expected vocab size {VOCAB_SIZE}, got {actual_vocab}"
            )));
        }

        let logits_slice = if shape.len() == 3 {
            &raw_logits[..num_frames * VOCAB_SIZE]
        } else {
            &raw_logits[..num_frames * VOCAB_SIZE]
        };

        let log_probs = log_softmax(logits_slice, num_frames, VOCAB_SIZE);

        debug!(
            "wav2vec2 produced {} frames ({:.2}s at 50 fps)",
            num_frames,
            num_frames as f64 * 0.02
        );

        Ok((log_probs, num_frames))
    }
}

impl ForcedAligner for CtcAligner {
    fn align(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        words: &[String],
    ) -> Result<Vec<WordAlignment>, CaptionError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }

        if audio.is_empty() {
            return Err(CaptionError::AlignmentFailed(
                "Cannot align empty audio".to_string(),
            ));
        }

        if sample_rate != 16_000 {
            return Err(CaptionError::AlignmentFailed(format!(
                "Expected 16 kHz audio, got {sample_rate} Hz"
            )));
        }

        let (log_probs, num_frames) = self.compute_log_probs(audio)?;

        let char_sequence = words_to_char_sequence(words, &self.char_to_idx);

        if char_sequence.is_empty() {
            return Ok(words
                .iter()
                .map(|w| WordAlignment {
                    text: w.clone(),
                    start: 0.0,
                    end: 0.0,
                    confidence: 0.0,
                })
                .collect());
        }

        let char_alignments =
            ctc_viterbi_align(&log_probs, num_frames, VOCAB_SIZE, &char_sequence)?;

        let word_alignments = group_into_words(&char_alignments, &char_sequence, words);

        debug!(
            "Aligned {} words across {} frames",
            word_alignments.len(),
            num_frames
        );

        Ok(word_alignments)
    }
}

// ---------------------------------------------------------------------------
// Log-softmax helper
// ---------------------------------------------------------------------------

fn log_softmax(logits: &[f32], num_frames: usize, vocab_size: usize) -> Vec<f32> {
    let mut log_probs = vec![0.0_f32; num_frames * vocab_size];

    for t in 0..num_frames {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];

        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut log_sum_exp = 0.0_f32;
        for &val in row {
            log_sum_exp += (val - max_val).exp();
        }
        let log_sum_exp = max_val + log_sum_exp.ln();

        for v in 0..vocab_size {
            log_probs[offset + v] = logits[offset + v] - log_sum_exp;
        }
    }

    log_probs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // log_softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn log_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
        let lp = log_softmax(&logits, 2, 3);

        for t in 0..2 {
            let sum: f32 = (0..3).map(|v| lp[t * 3 + v].exp()).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Frame {t}: exp(log_softmax) sums to {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn log_softmax_preserves_order() {
        let logits = vec![1.0, 3.0, 2.0];
        let lp = log_softmax(&logits, 1, 3);
        assert!(lp[1] > lp[0]);
        assert!(lp[1] > lp[2]);
    }

    #[test]
    fn log_softmax_all_same() {
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let lp = log_softmax(&logits, 1, 4);
        let expected = (0.25_f32).ln();
        for &val in &lp {
            assert!(
                (val - expected).abs() < 1e-5,
                "Expected {expected}, got {val}"
            );
        }
    }

    #[test]
    fn log_softmax_large_values() {
        let logits = vec![1000.0, 1001.0, 999.0];
        let lp = log_softmax(&logits, 1, 3);
        let sum: f32 = lp.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-3, "Numerical stability: sum = {sum}");
    }

    #[test]
    fn log_softmax_negative_values() {
        let logits = vec![-10.0, -20.0, -5.0];
        let lp = log_softmax(&logits, 1, 3);
        let sum: f32 = lp.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "Negative inputs: sum = {sum}");
        assert!(lp[2] > lp[0]);
        assert!(lp[2] > lp[1]);
    }

    // -----------------------------------------------------------------------
    // CtcAligner construction (requires model file)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires wav2vec2-base-960h ONNX model file"]
    fn ctc_aligner_loads_model() {
        let model_path = std::path::Path::new("models/wav2vec2-base-960h.onnx");
        let result = CtcAligner::new(model_path);
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());
    }

    #[test]
    #[ignore = "requires ONNX Runtime dylib loaded (load-dynamic feature)"]
    fn ctc_aligner_nonexistent_model_errors() {
        let model_path = std::path::Path::new("/tmp/nonexistent-model.onnx");
        let result = CtcAligner::new(model_path);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires wav2vec2-base-960h ONNX model file"]
    fn ctc_aligner_aligns_simple_audio() {
        let model_path = std::path::Path::new("models/wav2vec2-base-960h.onnx");
        let mut aligner = CtcAligner::new(model_path).expect("model should load");

        let audio = vec![0.0_f32; 16_000];
        let words = vec!["hello".to_string(), "world".to_string()];

        let result = aligner.align(&audio, 16_000, &words);
        assert!(result.is_ok(), "Alignment failed: {result:?}");

        let alignments = result.unwrap();
        assert_eq!(alignments.len(), 2);
        assert_eq!(alignments[0].text, "hello");
        assert_eq!(alignments[1].text, "world");
    }

    #[test]
    #[ignore = "requires wav2vec2-base-960h ONNX model file"]
    fn ctc_aligner_empty_words_returns_empty() {
        let model_path = std::path::Path::new("models/wav2vec2-base-960h.onnx");
        let mut aligner = CtcAligner::new(model_path).expect("model should load");

        let audio = vec![0.0_f32; 16_000];
        let result = aligner.align(&audio, 16_000, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    #[ignore = "requires wav2vec2-base-960h ONNX model file"]
    fn ctc_aligner_rejects_wrong_sample_rate() {
        let model_path = std::path::Path::new("models/wav2vec2-base-960h.onnx");
        let mut aligner = CtcAligner::new(model_path).expect("model should load");

        let audio = vec![0.0_f32; 8_000];
        let words = vec!["hello".to_string()];
        let result = aligner.align(&audio, 8_000, &words);
        assert!(result.is_err());
    }
}
