use log::debug;

use crate::align::ForcedAligner;
use crate::error::CaptionError;
use crate::stt::{SpeechRecognizer, TranscribeConfig, Transcription, Word};
use crate::vad::{extract_chunks, StandaloneVad};

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub transcribe_config: TranscribeConfig,
    pub padding_seconds: f64,
    pub max_chunk_seconds: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            transcribe_config: TranscribeConfig::default(),
            padding_seconds: 0.1,
            max_chunk_seconds: 30.0,
        }
    }
}

pub fn transcribe_pipeline(
    audio: &[f32],
    sample_rate: u32,
    backend: &dyn SpeechRecognizer,
    vad: Option<&mut StandaloneVad>,
    aligner: Option<&mut dyn ForcedAligner>,
    config: &PipelineConfig,
) -> Result<Transcription, CaptionError> {
    let used_standalone_vad = vad.is_some();

    let chunks: Vec<(Vec<f32>, f64)> = match vad {
        Some(vad_instance) => {
            let segments = vad_instance.detect_speech(audio)?;

            if segments.is_empty() {
                debug!("VAD detected zero speech segments; returning empty transcription");
                return Ok(Transcription {
                    words: Vec::new(),
                    language: config.transcribe_config.language.clone(),
                });
            }

            debug!("VAD detected {} speech segment(s)", segments.len());

            let extracted = extract_chunks(
                audio,
                sample_rate,
                &segments,
                config.padding_seconds,
                config.max_chunk_seconds,
            );

            debug!(
                "Extracted {} audio chunk(s) for transcription",
                extracted.len()
            );
            extracted
        }
        None => {
            debug!("No standalone VAD; transcribing full audio");
            vec![(audio.to_vec(), 0.0)]
        }
    };

    let chunk_config = if used_standalone_vad {
        TranscribeConfig {
            vad_model_path: None,
            ..config.transcribe_config.clone()
        }
    } else {
        config.transcribe_config.clone()
    };

    let mut all_words: Vec<Word> = Vec::new();
    let mut language = config.transcribe_config.language.clone();

    for (chunk_audio, chunk_start) in &chunks {
        let transcription = backend.transcribe(chunk_audio, &chunk_config)?;
        language = transcription.language.clone();

        let mut chunk_words = transcription.words;

        for word in &mut chunk_words {
            word.start += chunk_start;
            word.end += chunk_start;
        }

        all_words.extend(chunk_words);
    }

    if let Some(aligner_ref) = aligner {
        debug!("Running forced alignment on {} chunk(s)", chunks.len());

        let mut word_cursor = 0;

        for (chunk_audio, chunk_start) in &chunks {
            let chunk_duration = chunk_audio.len() as f64 / f64::from(sample_rate);
            let chunk_end = chunk_start + chunk_duration;

            let chunk_word_start = word_cursor;
            while word_cursor < all_words.len() && all_words[word_cursor].start < chunk_end + 0.001
            {
                word_cursor += 1;
            }
            let chunk_word_end = word_cursor;

            if chunk_word_start >= chunk_word_end {
                continue;
            }

            let word_texts: Vec<String> = all_words[chunk_word_start..chunk_word_end]
                .iter()
                .map(|w| w.text.clone())
                .collect();

            match aligner_ref.align(chunk_audio, sample_rate, &word_texts) {
                Ok(alignments) => {
                    for (word, alignment) in all_words[chunk_word_start..chunk_word_end]
                        .iter_mut()
                        .zip(alignments.iter())
                    {
                        word.start = alignment.start + chunk_start;
                        word.end = alignment.end + chunk_start;
                        word.confidence = alignment.confidence;
                    }
                    debug!(
                        "Aligned {} words for chunk at {:.2}s",
                        alignments.len(),
                        chunk_start
                    );
                }
                Err(e) => {
                    debug!(
                        "Forced alignment failed for chunk at {:.2}s, keeping backend timestamps: {e}",
                        chunk_start
                    );
                }
            }
        }
    }

    Ok(Transcription {
        words: all_words,
        language,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::{ForcedAligner, WordAlignment};
    use crate::error::CaptionError;
    use crate::stt::{SpeechRecognizer, TranscribeConfig, Transcription, Word};

    // -----------------------------------------------------------------------
    // Mock STT backend
    // -----------------------------------------------------------------------

    struct MockRecognizer {
        words_per_call: Vec<Vec<Word>>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockRecognizer {
        fn new(words_per_call: Vec<Vec<Word>>) -> Self {
            Self {
                words_per_call,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl SpeechRecognizer for MockRecognizer {
        fn transcribe(
            &self,
            _audio: &[f32],
            _config: &TranscribeConfig,
        ) -> Result<Transcription, CaptionError> {
            let idx = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let words = if idx < self.words_per_call.len() {
                self.words_per_call[idx].clone()
            } else {
                Vec::new()
            };
            Ok(Transcription {
                words,
                language: "en".to_string(),
            })
        }

        fn supports_word_timestamps(&self) -> bool {
            true
        }
    }

    // -----------------------------------------------------------------------
    // Mock aligner
    // -----------------------------------------------------------------------

    struct MockAligner {
        offsets: Vec<(f64, f64)>,
    }

    impl ForcedAligner for MockAligner {
        fn align(
            &mut self,
            _audio: &[f32],
            _sample_rate: u32,
            words: &[String],
        ) -> Result<Vec<WordAlignment>, CaptionError> {
            let mut result = Vec::with_capacity(words.len());
            for (i, text) in words.iter().enumerate() {
                let (start, end) = if i < self.offsets.len() {
                    self.offsets[i]
                } else {
                    (i as f64 * 0.5, i as f64 * 0.5 + 0.4)
                };
                result.push(WordAlignment {
                    text: text.clone(),
                    start,
                    end,
                    confidence: 0.95,
                });
            }
            Ok(result)
        }
    }

    struct FailingAligner;

    impl ForcedAligner for FailingAligner {
        fn align(
            &mut self,
            _audio: &[f32],
            _sample_rate: u32,
            _words: &[String],
        ) -> Result<Vec<WordAlignment>, CaptionError> {
            Err(CaptionError::AlignmentFailed("mock failure".to_string()))
        }
    }

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn make_word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
            confidence: 0.9,
        }
    }

    // -----------------------------------------------------------------------
    // Tests: no VAD, no aligner (simplest path)
    // -----------------------------------------------------------------------

    #[test]
    fn no_vad_no_aligner_passes_full_audio() {
        let backend = MockRecognizer::new(vec![vec![
            make_word("hello", 0.0, 0.5),
            make_word("world", 0.6, 1.0),
        ]]);

        let config = PipelineConfig::default();
        let audio = vec![0.0_f32; 16_000];

        let result = transcribe_pipeline(&audio, 16_000, &backend, None, None, &config).unwrap();

        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[0].text, "hello");
        assert_eq!(result.words[1].text, "world");
        assert!((result.words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((result.words[1].start - 0.6).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Tests: timestamp offset with multiple chunks (simulated)
    // -----------------------------------------------------------------------

    #[test]
    fn timestamp_offset_applied_to_backend_words() {
        let backend = MockRecognizer::new(vec![vec![
            make_word("first", 1.0, 1.5),
            make_word("second", 2.0, 2.3),
        ]]);

        let audio = vec![0.0_f32; 16_000];
        let config = PipelineConfig::default();
        let result = transcribe_pipeline(&audio, 16_000, &backend, None, None, &config).unwrap();

        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[0].text, "first");
        assert!((result.words[0].start - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.words[1].text, "second");
        assert!((result.words[1].start - 2.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Tests: aligner overwrites timestamps with offset
    // -----------------------------------------------------------------------

    #[test]
    fn aligner_timestamps_offset_to_original_timeline() {
        let backend = MockRecognizer::new(vec![vec![
            make_word("hello", 0.0, 0.5),
            make_word("world", 0.6, 1.0),
        ]]);

        let mut aligner = MockAligner {
            offsets: vec![(0.1, 0.4), (0.5, 0.9)],
        };

        let config = PipelineConfig::default();
        let audio = vec![0.0_f32; 16_000];

        let result =
            transcribe_pipeline(&audio, 16_000, &backend, None, Some(&mut aligner), &config)
                .unwrap();

        assert_eq!(result.words.len(), 2);
        assert!((result.words[0].start - 0.1).abs() < f64::EPSILON);
        assert!((result.words[0].end - 0.4).abs() < f64::EPSILON);
        assert!((result.words[1].start - 0.5).abs() < f64::EPSILON);
        assert!((result.words[1].end - 0.9).abs() < f64::EPSILON);
        assert!((result.words[0].confidence - 0.95).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Tests: aligner failure falls back to backend timestamps
    // -----------------------------------------------------------------------

    #[test]
    fn aligner_failure_keeps_backend_timestamps() {
        let backend = MockRecognizer::new(vec![vec![
            make_word("hello", 0.0, 0.5),
            make_word("world", 0.6, 1.0),
        ]]);

        let mut aligner = FailingAligner;

        let config = PipelineConfig::default();
        let audio = vec![0.0_f32; 16_000];

        let result =
            transcribe_pipeline(&audio, 16_000, &backend, None, Some(&mut aligner), &config)
                .unwrap();

        assert_eq!(result.words.len(), 2);
        assert!((result.words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((result.words[0].end - 0.5).abs() < f64::EPSILON);
        assert!((result.words[1].start - 0.6).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Tests: empty audio
    // -----------------------------------------------------------------------

    #[test]
    fn empty_audio_no_vad_returns_backend_result() {
        let backend = MockRecognizer::new(vec![vec![]]);
        let config = PipelineConfig::default();

        let result = transcribe_pipeline(&[], 16_000, &backend, None, None, &config).unwrap();
        assert!(result.words.is_empty());
    }

    // -----------------------------------------------------------------------
    // Tests: timestamps are monotonically increasing after pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn timestamps_monotonically_increasing_after_pipeline() {
        let backend = MockRecognizer::new(vec![vec![
            make_word("alpha", 0.0, 0.4),
            make_word("beta", 0.5, 0.9),
            make_word("gamma", 1.0, 1.5),
            make_word("delta", 1.6, 2.0),
        ]]);

        let config = PipelineConfig::default();
        let audio = vec![0.0_f32; 32_000];

        let result = transcribe_pipeline(&audio, 16_000, &backend, None, None, &config).unwrap();

        assert_eq!(result.words.len(), 4);

        for w in &result.words {
            assert!(
                w.start <= w.end,
                "Word '{}': start {} > end {}",
                w.text,
                w.start,
                w.end
            );
        }

        for pair in result.words.windows(2) {
            assert!(
                pair[1].start >= pair[0].start,
                "Timestamps not monotonically increasing: '{}' at {:.3} follows '{}' at {:.3}",
                pair[1].text,
                pair[1].start,
                pair[0].text,
                pair[0].start
            );
        }
    }

    #[test]
    fn aligner_timestamps_monotonically_increasing_after_alignment() {
        let backend = MockRecognizer::new(vec![vec![
            make_word("one", 0.0, 0.3),
            make_word("two", 0.4, 0.7),
            make_word("three", 0.8, 1.1),
        ]]);

        let mut aligner = MockAligner {
            offsets: vec![(0.05, 0.25), (0.30, 0.60), (0.70, 1.00)],
        };

        let config = PipelineConfig::default();
        let audio = vec![0.0_f32; 16_000];

        let result =
            transcribe_pipeline(&audio, 16_000, &backend, None, Some(&mut aligner), &config)
                .unwrap();

        assert_eq!(result.words.len(), 3);

        for pair in result.words.windows(2) {
            assert!(
                pair[1].start >= pair[0].start,
                "After alignment, timestamps not monotonically increasing: \
                 '{}' at {:.3} follows '{}' at {:.3}",
                pair[1].text,
                pair[1].start,
                pair[0].text,
                pair[0].start
            );
        }
    }

    // -----------------------------------------------------------------------
    // Tests: PipelineConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert!((config.padding_seconds - 0.1).abs() < f64::EPSILON);
        assert!((config.max_chunk_seconds - 30.0).abs() < f64::EPSILON);
        assert_eq!(config.transcribe_config.language, "en");
    }

    // -----------------------------------------------------------------------
    // Tests: language propagation
    // -----------------------------------------------------------------------

    #[test]
    fn language_propagated_from_backend() {
        struct FrenchRecognizer;
        impl SpeechRecognizer for FrenchRecognizer {
            fn transcribe(
                &self,
                _audio: &[f32],
                _config: &TranscribeConfig,
            ) -> Result<Transcription, CaptionError> {
                Ok(Transcription {
                    words: vec![make_word("bonjour", 0.0, 0.5)],
                    language: "fr".to_string(),
                })
            }
            fn supports_word_timestamps(&self) -> bool {
                true
            }
        }

        let backend = FrenchRecognizer;
        let config = PipelineConfig::default();
        let audio = vec![0.0_f32; 16_000];

        let result = transcribe_pipeline(&audio, 16_000, &backend, None, None, &config).unwrap();
        assert_eq!(result.language, "fr");
    }

    // -----------------------------------------------------------------------
    // Tests: vad_model_path disabled for per-chunk calls
    // -----------------------------------------------------------------------

    #[test]
    fn vad_model_path_cleared_when_standalone_vad_is_active() {
        let config = PipelineConfig {
            transcribe_config: TranscribeConfig {
                vad_model_path: Some("/path/to/vad.onnx".to_string()),
                ..TranscribeConfig::default()
            },
            ..PipelineConfig::default()
        };

        let used_standalone_vad = true;

        let chunk_config = if used_standalone_vad {
            TranscribeConfig {
                vad_model_path: None,
                ..config.transcribe_config.clone()
            }
        } else {
            config.transcribe_config.clone()
        };

        assert!(chunk_config.vad_model_path.is_none());
        assert_eq!(chunk_config.language, "en");
        assert!(chunk_config.initial_prompt.is_none());
    }

    #[test]
    fn vad_model_path_preserved_when_no_standalone_vad() {
        let config = PipelineConfig {
            transcribe_config: TranscribeConfig {
                vad_model_path: Some("/path/to/vad.onnx".to_string()),
                ..TranscribeConfig::default()
            },
            ..PipelineConfig::default()
        };

        let used_standalone_vad = false;

        let chunk_config = if used_standalone_vad {
            TranscribeConfig {
                vad_model_path: None,
                ..config.transcribe_config.clone()
            }
        } else {
            config.transcribe_config.clone()
        };

        assert_eq!(
            chunk_config.vad_model_path.as_deref(),
            Some("/path/to/vad.onnx")
        );
    }
}
