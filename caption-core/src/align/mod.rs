pub mod ctc_aligner;

use crate::error::CaptionError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WordAlignment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
}

pub trait ForcedAligner: Send {
    fn align(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        words: &[String],
    ) -> Result<Vec<WordAlignment>, CaptionError>;
}

// ---------------------------------------------------------------------------
// wav2vec2-base-960h vocabulary
// ---------------------------------------------------------------------------

// Index 0 = <pad> (CTC blank), 4 = | (word boundary / space).
// Uppercase A-Z at indices 5-31 (frequency-ordered), apostrophe at 27.
const WAV2VEC2_VOCAB: &[(&str, usize)] = &[
    ("<pad>", 0),
    ("<s>", 1),
    ("</s>", 2),
    ("<unk>", 3),
    ("|", 4),
    ("E", 5),
    ("T", 6),
    ("A", 7),
    ("O", 8),
    ("N", 9),
    ("I", 10),
    ("H", 11),
    ("S", 12),
    ("R", 13),
    ("D", 14),
    ("L", 15),
    ("U", 16),
    ("M", 17),
    ("W", 18),
    ("C", 19),
    ("F", 20),
    ("G", 21),
    ("Y", 22),
    ("P", 23),
    ("B", 24),
    ("V", 25),
    ("K", 26),
    ("'", 27),
    ("X", 28),
    ("J", 29),
    ("Q", 30),
    ("Z", 31),
];

// CTC blank token index
pub const BLANK_IDX: usize = 0;

// Word boundary / space token index
pub const SPACE_IDX: usize = 4;

pub const VOCAB_SIZE: usize = 32;

// 1 frame per 20ms at 16 kHz
pub const FRAMES_PER_SECOND: f64 = 50.0;

// 0.02s = 20ms
pub const SECONDS_PER_FRAME: f64 = 1.0 / FRAMES_PER_SECOND;

// ---------------------------------------------------------------------------
// Character-to-index mapping
// ---------------------------------------------------------------------------

pub fn build_char_to_idx() -> Vec<Option<usize>> {
    let mut map = vec![None; 128];
    for &(token, idx) in WAV2VEC2_VOCAB {
        if token.len() == 1 {
            let ch = token.chars().next().expect("single-char token");
            if ch.is_ascii_alphabetic() {
                map[ch.to_ascii_uppercase() as usize] = Some(idx);
                map[ch.to_ascii_lowercase() as usize] = Some(idx);
            } else if ch == '\'' || ch == '|' {
                map[ch as usize] = Some(idx);
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Digit-to-alpha expansion for alignment
// ---------------------------------------------------------------------------

fn digit_to_alpha(d: char) -> &'static str {
    match d {
        '0' => "ZERO",
        '1' => "ONE",
        '2' => "TWO",
        '3' => "THREE",
        '4' => "FOUR",
        '5' => "FIVE",
        '6' => "SIX",
        '7' => "SEVEN",
        '8' => "EIGHT",
        '9' => "NINE",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// Stage 2: Character sequence from transcript words
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignChar {
    pub vocab_idx: usize,
    pub word_idx: usize,
    pub is_space: bool,
}

pub fn words_to_char_sequence(words: &[String], char_to_idx: &[Option<usize>]) -> Vec<AlignChar> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut sequence = Vec::new();

    for (word_idx, word) in words.iter().enumerate() {
        if word_idx > 0 {
            sequence.push(AlignChar {
                vocab_idx: SPACE_IDX,
                word_idx: word_idx - 1,
                is_space: true,
            });
        }

        for ch in word.chars() {
            if ch.is_ascii_digit() {
                for alpha_ch in digit_to_alpha(ch).chars() {
                    if let Some(idx) = char_to_idx[alpha_ch as usize] {
                        sequence.push(AlignChar {
                            vocab_idx: idx,
                            word_idx,
                            is_space: false,
                        });
                    }
                }
                continue;
            }

            if !ch.is_ascii() || ch as usize >= 128 {
                continue;
            }
            if let Some(idx) = char_to_idx[ch as usize] {
                if idx == SPACE_IDX {
                    continue;
                }
                sequence.push(AlignChar {
                    vocab_idx: idx,
                    word_idx,
                    is_space: false,
                });
            }
        }
    }

    sequence
}

// ---------------------------------------------------------------------------
// Stage 3: CTC Viterbi forced alignment
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CharAlignment {
    pub char_idx: usize,
    pub frame: usize,
    pub log_prob: f32,
}

pub fn ctc_viterbi_align(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    char_sequence: &[AlignChar],
) -> Result<Vec<CharAlignment>, CaptionError> {
    let num_chars = char_sequence.len();

    if num_chars == 0 {
        return Ok(Vec::new());
    }

    // Target: [blank, char0, blank, char1, blank, ..., charN-1, blank]
    let target_len = 2 * num_chars + 1;

    if num_frames < num_chars {
        return Err(CaptionError::AlignmentFailed(format!(
            "Not enough frames ({num_frames}) for {num_chars} characters"
        )));
    }

    let mut targets = Vec::with_capacity(target_len);
    for (i, ac) in char_sequence.iter().enumerate() {
        if i == 0 {
            targets.push(BLANK_IDX);
        }
        targets.push(ac.vocab_idx);
        targets.push(BLANK_IDX);
    }
    debug_assert_eq!(targets.len(), target_len);

    let neg_inf: f32 = f32::NEG_INFINITY;

    let mut prev = vec![neg_inf; target_len];
    let mut curr = vec![neg_inf; target_len];

    let mut backptr = vec![0_usize; num_frames * target_len];

    prev[0] = log_probs[targets[0]];
    if target_len > 1 {
        prev[1] = log_probs[targets[1]];
    }

    for t in 1..num_frames {
        let frame_offset = t * vocab_size;

        for s in 0..target_len {
            let emit_log_prob = log_probs[frame_offset + targets[s]];

            let mut best_score = prev[s];
            let mut best_prev_s = s;

            if s >= 1 && prev[s - 1] > best_score {
                best_score = prev[s - 1];
                best_prev_s = s - 1;
            }

            if s >= 2 && targets[s] != targets[s - 2] && prev[s - 2] > best_score {
                best_score = prev[s - 2];
                best_prev_s = s - 2;
            }

            curr[s] = best_score + emit_log_prob;
            backptr[t * target_len + s] = best_prev_s;
        }

        std::mem::swap(&mut prev, &mut curr);
        curr.fill(neg_inf);
    }

    let final_scores = &prev;
    let mut best_s = target_len - 1;
    if target_len >= 2 && final_scores[target_len - 2] > final_scores[target_len - 1] {
        best_s = target_len - 2;
    }

    let mut path = vec![0_usize; num_frames];
    path[num_frames - 1] = best_s;
    for t in (1..num_frames).rev() {
        path[t - 1] = backptr[t * target_len + path[t]];
    }

    let mut result = Vec::with_capacity(num_chars);

    for (char_idx, ac) in char_sequence.iter().enumerate() {
        let target_s = 2 * char_idx + 1;

        let mut best_frame = None;
        let mut best_log_prob = neg_inf;

        for (t, &s) in path.iter().enumerate() {
            if s == target_s {
                let lp = log_probs[t * vocab_size + ac.vocab_idx];
                if best_frame.is_none() || lp > best_log_prob {
                    best_frame = Some(t);
                    best_log_prob = lp;
                }
            }
        }

        let frame = best_frame.unwrap_or_else(|| {
            for (t, &s) in path.iter().enumerate() {
                if s.abs_diff(target_s) <= 1 {
                    return t;
                }
            }
            0
        });

        result.push(CharAlignment {
            char_idx,
            frame,
            log_prob: best_log_prob,
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Stage 4: Group character alignments into word boundaries
// ---------------------------------------------------------------------------

pub fn group_into_words(
    char_alignments: &[CharAlignment],
    char_sequence: &[AlignChar],
    words: &[String],
) -> Vec<WordAlignment> {
    if words.is_empty() {
        return Vec::new();
    }

    if char_alignments.is_empty() {
        return words
            .iter()
            .map(|w| WordAlignment {
                text: w.clone(),
                start: 0.0,
                end: 0.0,
                confidence: 0.0,
            })
            .collect();
    }

    let num_words = words.len();
    let mut word_starts = vec![usize::MAX; num_words];
    let mut word_ends = vec![0_usize; num_words];
    let mut word_log_probs: Vec<Vec<f32>> = vec![Vec::new(); num_words];

    for (ca, ac) in char_alignments.iter().zip(char_sequence.iter()) {
        if ac.is_space {
            continue;
        }

        let wi = ac.word_idx;
        if wi >= num_words {
            continue;
        }

        if ca.frame < word_starts[wi] {
            word_starts[wi] = ca.frame;
        }
        if ca.frame > word_ends[wi] {
            word_ends[wi] = ca.frame;
        }
        if ca.log_prob.is_finite() {
            word_log_probs[wi].push(ca.log_prob);
        }
    }

    let mut result = Vec::with_capacity(num_words);

    for (wi, word) in words.iter().enumerate() {
        let start_frame = word_starts[wi];
        let end_frame = word_ends[wi];

        if start_frame == usize::MAX {
            result.push(WordAlignment {
                text: word.clone(),
                start: 0.0,
                end: 0.0,
                confidence: 0.0,
            });
            continue;
        }

        let start_secs = start_frame as f64 * SECONDS_PER_FRAME;
        let end_secs = (end_frame + 1) as f64 * SECONDS_PER_FRAME;

        let confidence = if word_log_probs[wi].is_empty() {
            0.0
        } else {
            let sum: f32 = word_log_probs[wi].iter().sum();
            let mean_log_prob = sum / word_log_probs[wi].len() as f32;
            mean_log_prob.exp().clamp(0.0, 1.0)
        };

        result.push(WordAlignment {
            text: word.clone(),
            start: start_secs,
            end: end_secs,
            confidence,
        });
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
    // build_char_to_idx tests
    // -----------------------------------------------------------------------

    #[test]
    fn char_to_idx_maps_uppercase_letters() {
        let map = build_char_to_idx();
        assert_eq!(map['A' as usize], Some(7));
        assert_eq!(map['E' as usize], Some(5));
        assert_eq!(map['Z' as usize], Some(31));
    }

    #[test]
    fn char_to_idx_maps_lowercase_letters() {
        let map = build_char_to_idx();
        assert_eq!(map['a' as usize], Some(7));
        assert_eq!(map['e' as usize], Some(5));
        assert_eq!(map['z' as usize], Some(31));
    }

    #[test]
    fn char_to_idx_maps_apostrophe() {
        let map = build_char_to_idx();
        assert_eq!(map['\'' as usize], Some(27));
    }

    #[test]
    fn char_to_idx_maps_pipe_as_space() {
        let map = build_char_to_idx();
        assert_eq!(map['|' as usize], Some(SPACE_IDX));
    }

    #[test]
    fn char_to_idx_returns_none_for_digits() {
        let map = build_char_to_idx();
        assert_eq!(map['0' as usize], None);
        assert_eq!(map['9' as usize], None);
    }

    #[test]
    fn char_to_idx_returns_none_for_punctuation() {
        let map = build_char_to_idx();
        assert_eq!(map['.' as usize], None);
        assert_eq!(map[',' as usize], None);
        assert_eq!(map['!' as usize], None);
    }

    // -----------------------------------------------------------------------
    // words_to_char_sequence tests
    // -----------------------------------------------------------------------

    #[test]
    fn char_sequence_empty_words() {
        let map = build_char_to_idx();
        let seq = words_to_char_sequence(&[], &map);
        assert!(seq.is_empty());
    }

    #[test]
    fn char_sequence_single_word() {
        let map = build_char_to_idx();
        let words = vec!["hello".to_string()];
        let seq = words_to_char_sequence(&words, &map);

        assert_eq!(seq.len(), 5);
        assert_eq!(seq[0].vocab_idx, 11); // H
        assert_eq!(seq[1].vocab_idx, 5); // E
        assert_eq!(seq[2].vocab_idx, 15); // L
        assert_eq!(seq[3].vocab_idx, 15); // L
        assert_eq!(seq[4].vocab_idx, 8); // O

        for ac in &seq {
            assert_eq!(ac.word_idx, 0);
            assert!(!ac.is_space);
        }
    }

    #[test]
    fn char_sequence_two_words() {
        let map = build_char_to_idx();
        let words = vec!["hi".to_string(), "ok".to_string()];
        let seq = words_to_char_sequence(&words, &map);

        assert_eq!(seq.len(), 5);
        assert_eq!(seq[0].vocab_idx, 11); // H
        assert_eq!(seq[0].word_idx, 0);
        assert_eq!(seq[1].vocab_idx, 10); // I
        assert_eq!(seq[1].word_idx, 0);
        assert_eq!(seq[2].vocab_idx, SPACE_IDX); // |
        assert_eq!(seq[2].word_idx, 0);
        assert!(seq[2].is_space);
        assert_eq!(seq[3].vocab_idx, 8); // O
        assert_eq!(seq[3].word_idx, 1);
        assert_eq!(seq[4].vocab_idx, 26); // K
        assert_eq!(seq[4].word_idx, 1);
    }

    #[test]
    fn char_sequence_skips_unmapped_characters() {
        let map = build_char_to_idx();
        let words = vec!["don't".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 5);
        assert_eq!(seq[3].vocab_idx, 27); // apostrophe
    }

    #[test]
    fn char_sequence_strips_punctuation() {
        let map = build_char_to_idx();
        let words = vec!["hello,".to_string(), "world.".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 11);
    }

    #[test]
    fn char_sequence_handles_mixed_case() {
        let map = build_char_to_idx();
        let words = vec!["Hello".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq[0].vocab_idx, 11);
    }

    // -----------------------------------------------------------------------
    // digit_to_alpha tests
    // -----------------------------------------------------------------------

    #[test]
    fn digit_to_alpha_covers_all_digits() {
        for d in '0'..='9' {
            let alpha = digit_to_alpha(d);
            assert!(!alpha.is_empty(), "digit '{d}' should have an expansion");
            assert!(
                alpha.chars().all(|c| c.is_ascii_uppercase()),
                "expansion for '{d}' should be all uppercase: {alpha}"
            );
        }
    }

    #[test]
    fn digit_to_alpha_non_digit_returns_empty() {
        assert_eq!(digit_to_alpha('a'), "");
        assert_eq!(digit_to_alpha('.'), "");
    }

    // -----------------------------------------------------------------------
    // char_sequence digit expansion tests
    // -----------------------------------------------------------------------

    #[test]
    fn char_sequence_expands_pure_digits() {
        let map = build_char_to_idx();
        let words = vec!["584".to_string()];
        let seq = words_to_char_sequence(&words, &map);

        // "5" -> FIVE(4), "8" -> EIGHT(5), "4" -> FOUR(4) = 13 chars
        assert_eq!(seq.len(), 13);
        for ac in &seq {
            assert_eq!(ac.word_idx, 0);
            assert!(!ac.is_space);
        }
    }

    #[test]
    fn char_sequence_expands_mixed_digits_and_letters() {
        let map = build_char_to_idx();
        let words = vec!["73kg".to_string()];
        let seq = words_to_char_sequence(&words, &map);

        // "7" -> SEVEN(5), "3" -> THREE(5), "k" -> K(1), "g" -> G(1) = 12 chars
        assert_eq!(seq.len(), 12);
        for ac in &seq {
            assert_eq!(ac.word_idx, 0);
        }
    }

    #[test]
    fn char_sequence_digit_word_gets_nonzero_alignment() {
        let map = build_char_to_idx();
        let words = vec!["42".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert!(!seq.is_empty(), "digit word should produce alignment chars");

        let num_frames = 30;
        let assignments: Vec<(usize, usize)> = seq
            .iter()
            .enumerate()
            .map(|(i, ac)| (i * 2 + 1, ac.vocab_idx))
            .collect();
        let lp = build_test_log_probs(num_frames, VOCAB_SIZE, &assignments);

        let char_aligns =
            ctc_viterbi_align(&lp, num_frames, VOCAB_SIZE, &seq).expect("should succeed");
        let word_aligns = group_into_words(&char_aligns, &seq, &words);

        assert_eq!(word_aligns.len(), 1);
        assert_eq!(word_aligns[0].text, "42");
        assert!(
            word_aligns[0].end > word_aligns[0].start,
            "should have nonzero duration"
        );
        assert!(word_aligns[0].confidence > 0.0);
    }

    // -----------------------------------------------------------------------
    // CTC Viterbi alignment tests
    // -----------------------------------------------------------------------

    fn build_test_log_probs(
        num_frames: usize,
        vocab_size: usize,
        assignments: &[(usize, usize)],
    ) -> Vec<f32> {
        let low = -10.0_f32;
        let high = -0.1_f32;
        let blank_base = -5.0_f32;

        let mut lp = vec![low; num_frames * vocab_size];

        for t in 0..num_frames {
            lp[t * vocab_size + BLANK_IDX] = blank_base;
        }

        for &(frame, vidx) in assignments {
            if frame < num_frames && vidx < vocab_size {
                lp[frame * vocab_size + vidx] = high;
            }
        }

        lp
    }

    #[test]
    fn viterbi_empty_sequence() {
        let lp = vec![-1.0_f32; 10 * VOCAB_SIZE];
        let result = ctc_viterbi_align(&lp, 10, VOCAB_SIZE, &[]).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn viterbi_single_character() {
        let map = build_char_to_idx();
        let words = vec!["a".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 1);

        let a_idx = seq[0].vocab_idx;

        let lp = build_test_log_probs(10, VOCAB_SIZE, &[(5, a_idx)]);

        let result = ctc_viterbi_align(&lp, 10, VOCAB_SIZE, &seq).expect("should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].frame, 5);
    }

    #[test]
    fn viterbi_two_characters_same_word() {
        let map = build_char_to_idx();
        let words = vec!["hi".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 2);

        let h_idx = seq[0].vocab_idx;
        let i_idx = seq[1].vocab_idx;

        let lp = build_test_log_probs(10, VOCAB_SIZE, &[(3, h_idx), (7, i_idx)]);

        let result = ctc_viterbi_align(&lp, 10, VOCAB_SIZE, &seq).expect("should succeed");
        assert_eq!(result.len(), 2);
        assert!(
            result[0].frame <= result[1].frame,
            "characters should be in order"
        );
    }

    #[test]
    fn viterbi_two_words() {
        let map = build_char_to_idx();
        let words = vec!["hi".to_string(), "ok".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 5);

        let assignments: Vec<(usize, usize)> = vec![
            (2, seq[0].vocab_idx),
            (4, seq[1].vocab_idx),
            (6, SPACE_IDX),
            (8, seq[3].vocab_idx),
            (10, seq[4].vocab_idx),
        ];
        let lp = build_test_log_probs(15, VOCAB_SIZE, &assignments);

        let result = ctc_viterbi_align(&lp, 15, VOCAB_SIZE, &seq).expect("should succeed");
        assert_eq!(result.len(), 5);

        for i in 1..result.len() {
            assert!(
                result[i].frame >= result[i - 1].frame,
                "Frame order violated at char {i}: {} < {}",
                result[i].frame,
                result[i - 1].frame
            );
        }
    }

    #[test]
    fn viterbi_not_enough_frames_errors() {
        let map = build_char_to_idx();
        let words = vec!["hello".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        let lp = vec![-1.0_f32; 3 * VOCAB_SIZE];
        let result = ctc_viterbi_align(&lp, 3, VOCAB_SIZE, &seq);
        assert!(result.is_err());
    }

    #[test]
    fn viterbi_minimum_frames() {
        let map = build_char_to_idx();
        let words = vec!["ab".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 2);

        let a_idx = seq[0].vocab_idx;
        let b_idx = seq[1].vocab_idx;
        let lp = build_test_log_probs(2, VOCAB_SIZE, &[(0, a_idx), (1, b_idx)]);

        let result = ctc_viterbi_align(&lp, 2, VOCAB_SIZE, &seq).expect("should succeed");
        assert_eq!(result.len(), 2);
    }

    // -----------------------------------------------------------------------
    // group_into_words tests
    // -----------------------------------------------------------------------

    #[test]
    fn group_empty() {
        let result = group_into_words(&[], &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn group_single_word() {
        let char_seq = vec![
            AlignChar {
                vocab_idx: 11,
                word_idx: 0,
                is_space: false,
            },
            AlignChar {
                vocab_idx: 10,
                word_idx: 0,
                is_space: false,
            },
        ];
        let char_aligns = vec![
            CharAlignment {
                char_idx: 0,
                frame: 5,
                log_prob: -0.1,
            },
            CharAlignment {
                char_idx: 1,
                frame: 8,
                log_prob: -0.2,
            },
        ];
        let words = vec!["hi".to_string()];

        let result = group_into_words(&char_aligns, &char_seq, &words);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hi");
        assert!((result[0].start - 5.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((result[0].end - 9.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!(result[0].confidence > 0.0);
        assert!(result[0].confidence <= 1.0);
    }

    #[test]
    fn group_two_words_with_space() {
        let char_seq = vec![
            AlignChar {
                vocab_idx: 11,
                word_idx: 0,
                is_space: false,
            },
            AlignChar {
                vocab_idx: 10,
                word_idx: 0,
                is_space: false,
            },
            AlignChar {
                vocab_idx: SPACE_IDX,
                word_idx: 0,
                is_space: true,
            },
            AlignChar {
                vocab_idx: 8,
                word_idx: 1,
                is_space: false,
            },
            AlignChar {
                vocab_idx: 26,
                word_idx: 1,
                is_space: false,
            },
        ];
        let char_aligns = vec![
            CharAlignment {
                char_idx: 0,
                frame: 2,
                log_prob: -0.1,
            },
            CharAlignment {
                char_idx: 1,
                frame: 4,
                log_prob: -0.2,
            },
            CharAlignment {
                char_idx: 2,
                frame: 6,
                log_prob: -0.5,
            },
            CharAlignment {
                char_idx: 3,
                frame: 8,
                log_prob: -0.1,
            },
            CharAlignment {
                char_idx: 4,
                frame: 10,
                log_prob: -0.3,
            },
        ];
        let words = vec!["hi".to_string(), "ok".to_string()];

        let result = group_into_words(&char_aligns, &char_seq, &words);
        assert_eq!(result.len(), 2);

        assert_eq!(result[0].text, "hi");
        assert!((result[0].start - 2.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((result[0].end - 5.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);

        assert_eq!(result[1].text, "ok");
        assert!((result[1].start - 8.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
        assert!((result[1].end - 11.0 * SECONDS_PER_FRAME).abs() < f64::EPSILON);
    }

    #[test]
    fn group_confidence_from_log_probs() {
        let char_seq = vec![AlignChar {
            vocab_idx: 7,
            word_idx: 0,
            is_space: false,
        }];
        let char_aligns = vec![CharAlignment {
            char_idx: 0,
            frame: 5,
            log_prob: 0.0,
        }];
        let words = vec!["a".to_string()];

        let result = group_into_words(&char_aligns, &char_seq, &words);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn group_confidence_clamped_to_one() {
        let char_seq = vec![AlignChar {
            vocab_idx: 7,
            word_idx: 0,
            is_space: false,
        }];
        let char_aligns = vec![CharAlignment {
            char_idx: 0,
            frame: 5,
            log_prob: 1.0,
        }];
        let words = vec!["a".to_string()];

        let result = group_into_words(&char_aligns, &char_seq, &words);
        assert!((result[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn group_word_with_no_chars_gets_zero_duration() {
        let char_seq = vec![];
        let char_aligns = vec![];
        let words = vec!["★★★".to_string()];

        let result = group_into_words(&char_aligns, &char_seq, &words);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "★★★");
        assert!((result[0].start - 0.0).abs() < f64::EPSILON);
        assert!((result[0].end - 0.0).abs() < f64::EPSILON);
        assert!((result[0].confidence - 0.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // End-to-end algorithm test (no model)
    // -----------------------------------------------------------------------

    #[test]
    fn end_to_end_algorithm() {
        let map = build_char_to_idx();
        let words = vec!["hi".to_string(), "ok".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 5);

        let assignments = vec![
            (2, seq[0].vocab_idx),
            (5, seq[1].vocab_idx),
            (8, SPACE_IDX),
            (11, seq[3].vocab_idx),
            (14, seq[4].vocab_idx),
        ];
        let lp = build_test_log_probs(20, VOCAB_SIZE, &assignments);

        let char_aligns =
            ctc_viterbi_align(&lp, 20, VOCAB_SIZE, &seq).expect("alignment should succeed");
        assert_eq!(char_aligns.len(), 5);

        let word_aligns = group_into_words(&char_aligns, &seq, &words);
        assert_eq!(word_aligns.len(), 2);
        assert_eq!(word_aligns[0].text, "hi");
        assert_eq!(word_aligns[1].text, "ok");

        assert!(word_aligns[0].start < word_aligns[0].end);
        assert!(word_aligns[1].start > word_aligns[0].end);
        assert!(word_aligns[0].confidence > 0.0);
        assert!(word_aligns[1].confidence > 0.0);
    }

    #[test]
    fn end_to_end_with_apostrophe() {
        let map = build_char_to_idx();
        let words = vec!["don't".to_string()];
        let seq = words_to_char_sequence(&words, &map);
        assert_eq!(seq.len(), 5);

        let assignments: Vec<(usize, usize)> = seq
            .iter()
            .enumerate()
            .map(|(i, ac)| (i * 3, ac.vocab_idx))
            .collect();
        let num_frames = 20;
        let lp = build_test_log_probs(num_frames, VOCAB_SIZE, &assignments);

        let char_aligns =
            ctc_viterbi_align(&lp, num_frames, VOCAB_SIZE, &seq).expect("should succeed");
        let word_aligns = group_into_words(&char_aligns, &seq, &words);
        assert_eq!(word_aligns.len(), 1);
        assert_eq!(word_aligns[0].text, "don't");
    }

    // -----------------------------------------------------------------------
    // ForcedAligner trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn forced_aligner_trait_is_object_safe() {
        struct MockAligner;
        impl ForcedAligner for MockAligner {
            fn align(
                &mut self,
                _audio: &[f32],
                _sample_rate: u32,
                _words: &[String],
            ) -> Result<Vec<WordAlignment>, CaptionError> {
                Ok(vec![])
            }
        }

        let mut aligner: Box<dyn ForcedAligner> = Box::new(MockAligner);
        let result = aligner.align(&[], 16000, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn forced_aligner_is_send() {
        fn assert_send<T: Send>() {}
        struct MockAligner;
        impl ForcedAligner for MockAligner {
            fn align(
                &mut self,
                _audio: &[f32],
                _sample_rate: u32,
                _words: &[String],
            ) -> Result<Vec<WordAlignment>, CaptionError> {
                Ok(vec![])
            }
        }
        assert_send::<MockAligner>();
    }

    // -----------------------------------------------------------------------
    // Constants sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn constants_are_correct() {
        assert_eq!(BLANK_IDX, 0);
        assert_eq!(SPACE_IDX, 4);
        assert_eq!(VOCAB_SIZE, 32);
        assert!((FRAMES_PER_SECOND - 50.0).abs() < f64::EPSILON);
        assert!((SECONDS_PER_FRAME - 0.02).abs() < 1e-10);
    }

    #[test]
    fn vocab_has_expected_size() {
        assert_eq!(WAV2VEC2_VOCAB.len(), VOCAB_SIZE);
    }

    #[test]
    fn vocab_indices_are_unique_and_contiguous() {
        let mut indices: Vec<usize> = WAV2VEC2_VOCAB.iter().map(|&(_, idx)| idx).collect();
        indices.sort_unstable();
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(
                i, idx,
                "Vocab indices should be contiguous 0..31, but index {i} != {idx}"
            );
        }
    }
}
