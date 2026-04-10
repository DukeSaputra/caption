use std::path::Path;

use log::{debug, warn};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadParams,
};

use super::{SpeechRecognizer, TranscribeConfig, Transcription, Word};
use crate::error::CaptionError;

pub struct WhisperBackend {
    ctx: WhisperContext,
}

impl WhisperBackend {
    pub fn new(model_path: &Path) -> Result<Self, CaptionError> {
        if !model_path.is_file() {
            return Err(CaptionError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        debug!("Loading Whisper model from: {}", model_path.display());

        let path_str = model_path.to_str().ok_or_else(|| {
            CaptionError::ModelLoadFailed(format!(
                "model path '{}' contains invalid UTF-8",
                model_path.display()
            ))
        })?;

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(path_str, params).map_err(|e| {
            CaptionError::ModelLoadFailed(format!("failed to load '{}': {e}", model_path.display()))
        })?;

        Ok(Self { ctx })
    }
}

impl SpeechRecognizer for WhisperBackend {
    fn transcribe(
        &self,
        audio: &[f32],
        config: &TranscribeConfig,
    ) -> Result<Transcription, CaptionError> {
        let mut state = self.ctx.create_state().map_err(|e| {
            CaptionError::TranscriptionFailed(format!("failed to create whisper state: {e}"))
        })?;

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0, // default patience heuristic
        });

        params.set_language(Some(&config.language));
        params.set_token_timestamps(true);

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        if let Some(ref prompt) = config.initial_prompt {
            params.set_initial_prompt(prompt);
        }

        params.set_no_context(true);

        params.set_temperature(config.temperature);
        params.set_temperature_inc(0.2);

        params.set_entropy_thold(2.4);

        if let Some(ref vad_path) = config.vad_model_path {
            debug!("Enabling VAD with model: {vad_path}");
            params.set_vad_model_path(Some(vad_path));
            let mut vad_params = WhisperVadParams::new();
            vad_params.set_threshold(0.45);
            params.set_vad_params(vad_params);
            params.enable_vad(true);
        }

        debug!(
            "Running Whisper transcription on {} samples ({:.2}s of audio)",
            audio.len(),
            audio.len() as f64 / 16_000.0
        );

        state.full(params, audio).map_err(|e| {
            CaptionError::TranscriptionFailed(format!("whisper inference failed: {e}"))
        })?;

        let num_segments = state.full_n_segments();

        debug!("Whisper produced {} segments", num_segments);

        if num_segments == 0 {
            warn!("No speech detected in input file");
            return Ok(Transcription {
                words: Vec::new(),
                language: config.language.clone(),
            });
        }

        let raw_tokens = collect_tokens_from_state(&state, num_segments, NO_SPEECH_PROB_THRESHOLD)?;
        let words = join_tokens_into_words(&raw_tokens);

        debug!("Extracted {} words from token data", words.len());

        Ok(Transcription {
            words,
            language: config.language.clone(),
        })
    }

    fn supports_word_timestamps(&self) -> bool {
        true
    }
}

const NO_SPEECH_PROB_THRESHOLD: f32 = 0.4;

fn should_skip_segment(no_speech_prob: f32, threshold: f32) -> bool {
    no_speech_prob > threshold
}

#[derive(Debug, Clone)]
pub(crate) struct RawToken {
    pub text: String,
    pub t0: i64,
    pub t1: i64,
    pub p: f32,
}

fn collect_tokens_from_state(
    state: &whisper_rs::WhisperState,
    num_segments: i32,
    no_speech_threshold: f32,
) -> Result<Vec<RawToken>, CaptionError> {
    let mut tokens = Vec::new();

    for seg_idx in 0..num_segments {
        let segment = state.get_segment(seg_idx).ok_or_else(|| {
            CaptionError::TranscriptionFailed(format!("segment {seg_idx} out of bounds"))
        })?;

        let no_speech_prob = segment.no_speech_probability();
        if should_skip_segment(no_speech_prob, no_speech_threshold) {
            debug!(
                "Skipping segment {seg_idx}: no_speech_prob={no_speech_prob:.3} > threshold={no_speech_threshold}"
            );
            continue;
        }

        let n_tokens = segment.n_tokens();

        for tok_idx in 0..n_tokens {
            let token = segment.get_token(tok_idx).ok_or_else(|| {
                CaptionError::TranscriptionFailed(format!(
                    "token {tok_idx} out of bounds in segment {seg_idx}"
                ))
            })?;

            let token_id = token.token_id();

            // 50257 = special token boundary in whisper.cpp
            if token_id >= 50257 {
                continue;
            }

            let token_text = token.to_str_lossy().map_err(|e| {
                CaptionError::TranscriptionFailed(format!(
                    "failed to get token text for segment {seg_idx}, token {tok_idx}: {e}"
                ))
            })?;

            if token_text.trim().is_empty() {
                continue;
            }

            let token_data = token.token_data();

            tokens.push(RawToken {
                text: token_text.into_owned(),
                t0: token_data.t0,
                t1: token_data.t1,
                p: token_data.p,
            });
        }
    }

    Ok(tokens)
}

pub(crate) fn join_tokens_into_words(tokens: &[RawToken]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current_word = String::new();
    let mut word_start: f64 = 0.0;
    let mut word_end: f64 = 0.0;
    let mut word_confidence: f32 = 0.0;
    let mut token_count: usize = 0;

    for token in tokens {
        if token.text.starts_with(' ') && !current_word.is_empty() {
            words.push(Word {
                text: current_word.trim().to_string(),
                start: word_start,
                end: word_end,
                confidence: if token_count > 0 {
                    word_confidence / token_count as f32
                } else {
                    0.0
                },
            });
            current_word.clear();
            token_count = 0;
            word_confidence = 0.0;
        }

        if current_word.is_empty() {
            word_start = token.t0 as f64 / 100.0;
        }
        word_end = token.t1 as f64 / 100.0;
        current_word.push_str(token.text.trim());
        word_confidence += token.p;
        token_count += 1;
    }

    if !current_word.trim().is_empty() {
        words.push(Word {
            text: current_word.trim().to_string(),
            start: word_start,
            end: word_end,
            confidence: if token_count > 0 {
                word_confidence / token_count as f32
            } else {
                0.0
            },
        });
    }

    words.retain(|w| w.text.chars().any(|c| c.is_alphanumeric()));

    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_model_returns_error() {
        let result = WhisperBackend::new(Path::new("/tmp/no-such-model.bin"));
        match result {
            Err(CaptionError::ModelNotFound(_)) => {}
            Err(other) => panic!("Expected ModelNotFound, got: {other:?}"),
            Ok(_) => panic!("Expected ModelNotFound error, got Ok"),
        }
    }

    #[test]
    fn whisper_backend_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WhisperBackend>();
    }

    // --- Sub-word token joining tests ---

    #[test]
    fn simple_words_with_leading_spaces() {
        let tokens = vec![
            RawToken {
                text: " Hello".to_string(),
                t0: 0,
                t1: 50,
                p: 0.9,
            },
            RawToken {
                text: " world".to_string(),
                t0: 50,
                t1: 100,
                p: 0.8,
            },
        ];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((words[0].end - 0.5).abs() < f64::EPSILON);
        assert!((words[1].start - 0.5).abs() < f64::EPSILON);
        assert!((words[1].end - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn subword_tokens_join_into_single_word() {
        let tokens = vec![
            RawToken {
                text: " un".to_string(),
                t0: 100,
                t1: 120,
                p: 0.7,
            },
            RawToken {
                text: "believ".to_string(),
                t0: 120,
                t1: 160,
                p: 0.8,
            },
            RawToken {
                text: "able".to_string(),
                t0: 160,
                t1: 200,
                p: 0.9,
            },
        ];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "unbelievable");
        assert!((words[0].start - 1.0).abs() < f64::EPSILON);
        assert!((words[0].end - 2.0).abs() < f64::EPSILON);
        assert!((words[0].confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn single_token_word() {
        let tokens = vec![RawToken {
            text: " I".to_string(),
            t0: 200,
            t1: 230,
            p: 0.95,
        }];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "I");
        assert!((words[0].start - 2.0).abs() < f64::EPSILON);
        assert!((words[0].end - 2.3).abs() < f64::EPSILON);
        assert!((words[0].confidence - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_tokens_produce_no_words() {
        let tokens: Vec<RawToken> = vec![];
        let words = join_tokens_into_words(&tokens);
        assert!(words.is_empty());
    }

    #[test]
    fn standalone_punctuation_token_is_dropped() {
        let tokens = vec![
            RawToken {
                text: " -".to_string(),
                t0: 1,
                t1: 18,
                p: 0.5,
            },
            RawToken {
                text: " Audio".to_string(),
                t0: 18,
                t1: 90,
                p: 0.9,
            },
        ];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "Audio");
    }

    #[test]
    fn punctuation_attached_to_word_is_kept() {
        let tokens = vec![RawToken {
            text: " wow!".to_string(),
            t0: 0,
            t1: 30,
            p: 0.9,
        }];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "wow!");
    }

    #[test]
    fn all_punctuation_tokens_produce_empty_list() {
        let tokens = vec![
            RawToken {
                text: " -".to_string(),
                t0: 0,
                t1: 10,
                p: 0.5,
            },
            RawToken {
                text: " ...".to_string(),
                t0: 10,
                t1: 20,
                p: 0.5,
            },
        ];
        let words = join_tokens_into_words(&tokens);
        assert!(words.is_empty());
    }

    #[test]
    fn mixed_simple_and_subword_tokens() {
        let tokens = vec![
            RawToken {
                text: " Hello".to_string(),
                t0: 0,
                t1: 50,
                p: 0.9,
            },
            RawToken {
                text: " un".to_string(),
                t0: 50,
                t1: 70,
                p: 0.7,
            },
            RawToken {
                text: "believ".to_string(),
                t0: 70,
                t1: 100,
                p: 0.8,
            },
            RawToken {
                text: "able".to_string(),
                t0: 100,
                t1: 130,
                p: 0.9,
            },
            RawToken {
                text: " world".to_string(),
                t0: 130,
                t1: 180,
                p: 0.85,
            },
        ];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "unbelievable");
        assert_eq!(words[2].text, "world");
    }

    #[test]
    fn centiseconds_to_seconds_conversion() {
        let tokens = vec![RawToken {
            text: " test".to_string(),
            t0: 150,
            t1: 250,
            p: 0.5,
        }];
        let words = join_tokens_into_words(&tokens);
        assert_eq!(words.len(), 1);
        assert!((words[0].start - 1.5).abs() < f64::EPSILON);
        assert!((words[0].end - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn no_speech_prob_threshold_is_reasonable() {
        assert!(NO_SPEECH_PROB_THRESHOLD > 0.0);
        assert!(NO_SPEECH_PROB_THRESHOLD < 1.0);
        assert!((NO_SPEECH_PROB_THRESHOLD - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn zero_segments_via_empty_token_join_produces_empty_words() {
        let empty_tokens: Vec<RawToken> = vec![];
        let words = join_tokens_into_words(&empty_tokens);
        assert!(words.is_empty(), "Expected empty words from empty tokens");
    }

    // --- should_skip_segment tests ---

    #[test]
    fn skip_segment_when_prob_exceeds_threshold() {
        assert!(should_skip_segment(0.7, 0.6));
        assert!(should_skip_segment(1.0, 0.6));
    }

    #[test]
    fn keep_segment_when_prob_below_threshold() {
        assert!(!should_skip_segment(0.3, 0.6));
        assert!(!should_skip_segment(0.0, 0.6));
    }

    #[test]
    fn keep_segment_when_prob_equals_threshold() {
        assert!(!should_skip_segment(0.6, 0.6));
    }

    #[test]
    #[ignore] // Requires a model file on disk
    fn integration_silence_produces_empty_transcription() {
        use crate::stt::model::find_model;

        let model_path = match find_model(None) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping integration test: no model file found");
                return;
            }
        };

        let backend = WhisperBackend::new(&model_path).expect("failed to load model");
        let silence = vec![0.0f32; 16_000 * 2];
        let config = TranscribeConfig::default();
        let result = backend
            .transcribe(&silence, &config)
            .expect("transcription should succeed");

        assert!(
            result.words.len() <= 2,
            "Expected 0-2 words from silence, got {}",
            result.words.len()
        );
    }
}
