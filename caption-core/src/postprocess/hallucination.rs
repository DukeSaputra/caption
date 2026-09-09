use crate::postprocess::PostProcessor;
use crate::stt::Word;

#[derive(Debug, Clone, Copy, Default)]
pub enum HallucinationMode {
    Off,
    #[default]
    Moderate,
    Aggressive,
}

pub struct HallucinationFilter {
    pub mode: HallucinationMode,
    pub allowed_phrases: Vec<String>,
}

impl PostProcessor for HallucinationFilter {
    fn process(&self, words: &mut Vec<Word>) {
        match self.mode {
            HallucinationMode::Off => {}
            HallucinationMode::Moderate => {
                remove_trailing_phrases(words, &self.allowed_phrases);
                deduplicate_consecutive_repeats(words);
                remove_ngram_repetitions(words);
            }
            HallucinationMode::Aggressive => {
                remove_trailing_phrases(words, &self.allowed_phrases);
                deduplicate_consecutive_repeats(words);
                remove_ngram_repetitions(words);
                remove_low_confidence_words(words);
                prune_repetitive_tail(words);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Common hallucination phrases
// ---------------------------------------------------------------------------

const HALLUCINATION_PHRASES: &[&str] = &[
    "thank you",
    "thank you for watching",
    "thank you for listening",
    "thanks for watching",
    "thanks for listening",
    "thank you so much for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "please like and subscribe",
    "please like comment and subscribe",
    "don't forget to subscribe",
    "don't forget to like and subscribe",
    "hit the bell",
    "leave a comment",
    "comment below",
    "link in the description",
    "see you next time",
    "see you in the next video",
    "see you in the next one",
    "see you later",
    "i'll see you in the next video",
    "stay tuned",
    "bye bye",
    "goodbye",
    "subtitles by the amara.org community",
    "subtitles by",
    "captions by",
    "transcribed by",
    "translated by",
    "copyright",
    "all rights reserved",
    "amen",
];

// ---------------------------------------------------------------------------
// 4a: Phrase blocklist -- trailing words only
// ---------------------------------------------------------------------------

fn remove_trailing_phrases(words: &mut Vec<Word>, allowed: &[String]) {
    // Max trailing words to inspect per pass
    const TRAILING_WINDOW: usize = 10;

    'outer: loop {
        if words.is_empty() {
            break;
        }

        let start = words.len().saturating_sub(TRAILING_WINDOW);
        let trailing_lower: Vec<String> = words[start..]
            .iter()
            .map(|w| w.text.to_lowercase())
            .collect();

        for phrase in HALLUCINATION_PHRASES {
            if allowed.iter().any(|a| a == phrase) {
                continue;
            }

            let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
            let phrase_len = phrase_words.len();

            let window_len = trailing_lower.len();
            if phrase_len > window_len {
                continue;
            }

            let tail_offset = window_len.saturating_sub(phrase_len);
            if tail_offset + phrase_len != window_len {
                continue;
            }
            let matches = phrase_words
                .iter()
                .enumerate()
                .all(|(i, pw)| trailing_lower[tail_offset + i] == *pw);

            if matches {
                let remove_from = start + tail_offset;
                words.truncate(remove_from);
                continue 'outer;
            }
        }

        break;
    }
}

// ---------------------------------------------------------------------------
// 4b: Consecutive-repeat detection
// ---------------------------------------------------------------------------

/// Trailing punctuation Whisper attaches to a word.
const COMPARE_TRIM: &[char] = &['.', ',', ';', ':', '?', '!'];

/// Key used to decide whether two words are "the same word".
///
/// `PunctuationStripper` runs *after* these filters, so words still carry their
/// punctuation here: "Ganyat?" and "Ganyat" compare as different strings and the
/// repeat escapes. Normalise for the comparison only -- the word's own text is
/// never modified, so the stripper stays responsible for the final output.
fn compare_key(text: &str) -> String {
    text.trim_end_matches(COMPARE_TRIM).to_lowercase()
}

fn deduplicate_consecutive_repeats(words: &mut Vec<Word>) {
    if words.len() < 2 {
        return;
    }

    let mut i = 0;
    while i < words.len() {
        let max_seq = (words.len() - i) / 2;
        let mut removed = false;

        'seq: for seq_len in 1..=max_seq {
            let mut reps = 1usize;
            loop {
                let next_start = i + seq_len * reps;
                let next_end = next_start + seq_len;
                if next_end > words.len() {
                    break;
                }
                let matches = words[i..i + seq_len]
                    .iter()
                    .zip(words[next_start..next_end].iter())
                    .all(|(a, b)| compare_key(&a.text) == compare_key(&b.text));
                if matches {
                    reps += 1;
                } else {
                    break;
                }
            }

            if reps > 1 {
                let remove_start = i + seq_len;
                let remove_end = i + seq_len * reps;
                words.drain(remove_start..remove_end);
                removed = true;
                break 'seq;
            }
        }

        if !removed {
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// 4b1/2: 5-gram duplicate detection (Moderate + Aggressive)
// ---------------------------------------------------------------------------

const NGRAM_SIZE: usize = 5;

const NGRAM_MAX_OCCURRENCES: usize = 2;

/// Repeats only count as a Whisper loop when they cluster inside this span.
///
/// Whisper hallucinates by looping a phrase within a single decode window, so a
/// real loop always fits inside one chunk (`PipelineConfig::max_chunk_seconds`
/// is 30s). Ordinary speech reuses stock phrasing minutes apart -- an 87-minute
/// meeting says "questions that you want to" many times without hallucinating.
/// Without this window the filter cut long transcripts down to their first few
/// hundred words.
const NGRAM_LOCALITY_SECONDS: f64 = 30.0;

fn remove_ngram_repetitions(words: &mut Vec<Word>) {
    if words.len() < NGRAM_SIZE {
        return;
    }

    let mut ngram_positions: std::collections::HashMap<Vec<String>, Vec<usize>> =
        std::collections::HashMap::new();

    for i in 0..=words.len() - NGRAM_SIZE {
        let gram: Vec<String> = words[i..i + NGRAM_SIZE]
            .iter()
            .map(|w| compare_key(&w.text))
            .collect();
        ngram_positions.entry(gram).or_default().push(i);
    }

    // Mark the looping copies for removal. We delete only the repeats
    // themselves, never the rest of the transcript: content after a loop is
    // ordinary speech that the old truncate-to-end behaviour threw away.
    let mut doomed = vec![false; words.len()];
    let mut any_marked = false;

    for positions in ngram_positions.values() {
        if positions.len() <= NGRAM_MAX_OCCURRENCES {
            continue;
        }

        let mut burst_start = 0;
        while burst_start < positions.len() {
            let burst_time = words[positions[burst_start]].start;
            let mut burst_end = burst_start;
            while burst_end < positions.len()
                && words[positions[burst_end]].start - burst_time <= NGRAM_LOCALITY_SECONDS
            {
                burst_end += 1;
            }

            // A burst only looks like a loop once it exceeds the allowance; the
            // first NGRAM_MAX_OCCURRENCES copies in each burst are always kept.
            let removal_start = (burst_start + NGRAM_MAX_OCCURRENCES).min(burst_end);
            for &position in &positions[removal_start..burst_end] {
                let span_end = (position + NGRAM_SIZE).min(doomed.len());
                for slot in &mut doomed[position..span_end] {
                    *slot = true;
                }
                any_marked = true;
            }

            burst_start = burst_end;
        }
    }

    if !any_marked {
        return;
    }

    let mut index = 0;
    words.retain(|_| {
        let keep = !doomed[index];
        index += 1;
        keep
    });
}

// ---------------------------------------------------------------------------
// 4c: Confidence filtering (Aggressive only)
// ---------------------------------------------------------------------------

const LOW_CONFIDENCE_THRESHOLD: f32 = 0.1;

fn remove_low_confidence_words(words: &mut Vec<Word>) {
    words.retain(|w| w.confidence >= LOW_CONFIDENCE_THRESHOLD);
}

// ---------------------------------------------------------------------------
// 4d: Repetition-ratio pruning (Aggressive only)
// ---------------------------------------------------------------------------

const REPETITION_RATIO_THRESHOLD: f32 = 0.3;

const MIN_WORDS_FOR_REPETITION_CHECK: usize = 5;

/// How many trailing words the ratio is measured over.
///
/// The check must be local. Measured across a whole transcript the ratio is
/// meaningless: natural speech reuses common words, so a real 15,779-word
/// meeting scores 0.15 -- under the threshold -- and the old whole-transcript
/// version truncated it at the first repeated word, keeping 9 words.
const REPETITION_TAIL_WINDOW: usize = 50;

fn prune_repetitive_tail(words: &mut Vec<Word>) {
    if words.len() < MIN_WORDS_FOR_REPETITION_CHECK {
        return;
    }

    let window_start = words.len().saturating_sub(REPETITION_TAIL_WINDOW);
    let window = &words[window_start..];

    let unique: std::collections::HashSet<String> =
        window.iter().map(|w| w.text.to_lowercase()).collect();
    let ratio = unique.len() as f32 / window.len() as f32;

    if ratio >= REPETITION_RATIO_THRESHOLD {
        return;
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (offset, word) in window.iter().enumerate() {
        let lower = word.text.to_lowercase();
        if !seen.insert(lower) {
            words.truncate(window_start + offset);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::Word;

    fn word(text: &str) -> Word {
        Word {
            text: text.to_string(),
            start: 0.0,
            end: 0.5,
            confidence: 0.99,
        }
    }

    fn word_conf(text: &str, confidence: f32) -> Word {
        Word {
            text: text.to_string(),
            start: 0.0,
            end: 0.5,
            confidence,
        }
    }

    fn texts(words: &[Word]) -> Vec<&str> {
        words.iter().map(|w| w.text.as_str()).collect()
    }

    // -----------------------------------------------------------------------
    // Off mode
    // -----------------------------------------------------------------------

    #[test]
    fn off_mode_removes_nothing() {
        let mut words = vec![word("thank"), word("you"), word("for"), word("watching")];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Off,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["thank", "you", "for", "watching"]);
    }

    #[test]
    fn off_mode_empty_list_stays_empty() {
        let mut words: Vec<Word> = vec![];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Off,
        };
        filter.process(&mut words);
        assert!(words.is_empty());
    }

    // -----------------------------------------------------------------------
    // Moderate -- phrase blocklist
    // -----------------------------------------------------------------------

    #[test]
    fn moderate_removes_trailing_thank_you_for_watching() {
        let mut words = vec![
            word("Hello"),
            word("world"),
            word("thank"),
            word("you"),
            word("for"),
            word("watching"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Hello", "world"]);
    }

    #[test]
    fn moderate_does_not_remove_thank_you_in_mid_speech() {
        let mut words = vec![
            word("I"),
            word("want"),
            word("to"),
            word("thank"),
            word("you"),
            word("for"),
            word("coming"),
            word("today"),
            word("it"),
            word("was"),
            word("great"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(
            texts(&words),
            &["I", "want", "to", "thank", "you", "for", "coming", "today", "it", "was", "great"]
        );
    }

    #[test]
    fn moderate_removes_trailing_thanks_for_watching() {
        let mut words = vec![
            word("Great"),
            word("video"),
            word("thanks"),
            word("for"),
            word("watching"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Great", "video"]);
    }

    #[test]
    fn moderate_removes_trailing_please_subscribe() {
        let mut words = vec![
            word("That"),
            word("was"),
            word("great"),
            word("please"),
            word("subscribe"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["That", "was", "great"]);
    }

    #[test]
    fn moderate_removes_trailing_like_and_subscribe() {
        let mut words = vec![word("Bye"), word("like"), word("and"), word("subscribe")];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Bye"]);
    }

    #[test]
    fn moderate_removes_trailing_amen() {
        let mut words = vec![word("The"), word("end"), word("Amen")];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["The", "end"]);
    }

    #[test]
    fn moderate_phrase_match_is_case_insensitive() {
        let mut words = vec![
            word("Great"),
            word("THANK"),
            word("YOU"),
            word("FOR"),
            word("WATCHING"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Great"]);
    }

    #[test]
    fn moderate_only_real_content_no_phrase_unchanged() {
        let mut words = vec![word("Hello"), word("world"), word("today")];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Hello", "world", "today"]);
    }

    #[test]
    fn moderate_empty_list_stays_empty() {
        let mut words: Vec<Word> = vec![];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert!(words.is_empty());
    }

    // -----------------------------------------------------------------------
    // Moderate -- consecutive repeat detection
    // -----------------------------------------------------------------------

    #[test]
    fn moderate_removes_consecutive_single_word_repeats() {
        let mut words = vec![
            word("Thank"),
            word("you"),
            word("Thank"),
            word("you"),
            word("Thank"),
            word("you"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        let mut words2 = vec![
            word("Thank"),
            word("you"),
            word("Thank"),
            word("you"),
            word("Thank"),
            word("you"),
        ];
        deduplicate_consecutive_repeats(&mut words2);
        assert_eq!(texts(&words2), &["Thank", "you"]);
    }

    #[test]
    fn dedup_single_word_run() {
        let mut words = vec![word("Hello"), word("Hello"), word("Hello"), word("world")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["Hello", "world"]);
    }

    #[test]
    fn dedup_two_word_sequence_repeated() {
        let mut words = vec![
            word("the"),
            word("end"),
            word("the"),
            word("end"),
            word("the"),
            word("end"),
        ];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["the", "end"]);
    }

    #[test]
    fn dedup_non_consecutive_identical_words_kept() {
        let mut words = vec![word("Hello"), word("world"), word("Hello")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["Hello", "world", "Hello"]);
    }

    #[test]
    fn dedup_case_insensitive_repeats() {
        let mut words = vec![word("Thank"), word("you"), word("THANK"), word("YOU")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["Thank", "you"]);
    }

    #[test]
    fn dedup_empty_list_unchanged() {
        let mut words: Vec<Word> = vec![];
        deduplicate_consecutive_repeats(&mut words);
        assert!(words.is_empty());
    }

    #[test]
    fn dedup_single_word_unchanged() {
        let mut words = vec![word("hello")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(words.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Moderate + Aggressive -- 5-gram duplicate detection
    // -----------------------------------------------------------------------

    #[test]
    fn ngram_third_local_occurrence_removed_and_rest_kept() {
        let mut words = vec![
            word("a"),
            word("b"),
            word("c"),
            word("d"),
            word("e"),
            word("f"),
            word("g"),
            word("a"),
            word("b"),
            word("c"),
            word("d"),
            word("e"),
            word("h"),
            word("i"),
            word("a"),
            word("b"),
            word("c"),
            word("d"),
            word("e"),
            word("j"),
            word("k"),
        ];
        remove_ngram_repetitions(&mut words);
        // The third copy of "a b c d e" is dropped, but the words that follow it
        // ("j", "k") are ordinary content and survive. Previously the whole tail
        // was truncated, which is what destroyed long transcripts.
        assert_eq!(
            texts(&words),
            &["a", "b", "c", "d", "e", "f", "g", "a", "b", "c", "d", "e", "h", "i", "j", "k"]
        );
    }

    #[test]
    fn ngram_two_occurrences_left_alone() {
        let mut words = vec![
            word("a"),
            word("b"),
            word("c"),
            word("d"),
            word("e"),
            word("f"),
            word("g"),
            word("a"),
            word("b"),
            word("c"),
            word("d"),
            word("e"),
            word("h"),
            word("i"),
        ];
        remove_ngram_repetitions(&mut words);
        assert_eq!(
            texts(&words),
            &["a", "b", "c", "d", "e", "f", "g", "a", "b", "c", "d", "e", "h", "i"]
        );
    }

    #[test]
    fn ngram_short_transcript_unchanged() {
        let mut words = vec![word("hello"), word("world")];
        remove_ngram_repetitions(&mut words);
        assert_eq!(texts(&words), &["hello", "world"]);
    }

    #[test]
    fn ngram_empty_list_unchanged() {
        let mut words: Vec<Word> = vec![];
        remove_ngram_repetitions(&mut words);
        assert!(words.is_empty());
    }

    #[test]
    fn ngram_normal_varied_text_unchanged() {
        let mut words = vec![
            word("the"),
            word("quick"),
            word("brown"),
            word("fox"),
            word("jumps"),
            word("over"),
            word("the"),
            word("lazy"),
            word("dog"),
            word("today"),
        ];
        remove_ngram_repetitions(&mut words);
        assert_eq!(
            texts(&words),
            &["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"]
        );
    }

    // -----------------------------------------------------------------------
    // Tests: n-gram locality (long-form transcripts)
    // -----------------------------------------------------------------------

    /// Build a run of words starting at `start_time`, one word every 0.4s.
    fn words_at(texts: &[&str], start_time: f64) -> Vec<Word> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let start = start_time + i as f64 * 0.4;
                Word {
                    text: (*text).to_string(),
                    start,
                    end: start + 0.35,
                    confidence: 0.99,
                }
            })
            .collect()
    }

    const PHRASE: [&str; 5] = ["questions", "that", "you", "want", "to"];

    #[test]
    fn ngram_repeats_far_apart_in_time_are_kept() {
        // A real 87-minute meeting says "questions that you want to" three times,
        // minutes apart. That is ordinary speech, not a Whisper loop.
        let mut words = Vec::new();
        words.extend(words_at(
            &["welcome", "to", "this", "morning", "session"],
            0.0,
        ));
        words.extend(words_at(&PHRASE, 10.0));
        words.extend(words_at(&["please", "pop", "them", "in", "chat"], 300.0));
        words.extend(words_at(&PHRASE, 320.0));
        words.extend(words_at(&["we", "will", "cover", "them", "shortly"], 900.0));
        words.extend(words_at(&PHRASE, 920.0));

        let before = words.len();
        remove_ngram_repetitions(&mut words);

        assert_eq!(
            words.len(),
            before,
            "far-apart repeats must not truncate the transcript"
        );
    }

    #[test]
    fn ngram_content_after_repetition_burst_survives() {
        // A genuine loop early on must not cost us the rest of the recording.
        let mut words = Vec::new();
        words.extend(words_at(&PHRASE, 10.0));
        words.extend(words_at(&PHRASE, 12.0));
        words.extend(words_at(&PHRASE, 14.0));
        words.extend(words_at(&PHRASE, 16.0));
        words.extend(words_at(
            &["now", "lets", "talk", "about", "endorsement"],
            600.0,
        ));

        remove_ngram_repetitions(&mut words);

        let kept = texts(&words);
        assert!(
            kept.ends_with(&["now", "lets", "talk", "about", "endorsement"]),
            "content after the burst must survive, got: {kept:?}"
        );
    }

    #[test]
    fn ngram_tight_repeats_are_still_removed() {
        // Whisper looping the same phrase within a few seconds is a hallucination.
        let mut words = Vec::new();
        words.extend(words_at(&PHRASE, 10.0));
        words.extend(words_at(&PHRASE, 12.0));
        words.extend(words_at(&PHRASE, 14.0));
        words.extend(words_at(&PHRASE, 16.0));

        let before = words.len();
        remove_ngram_repetitions(&mut words);

        assert!(
            words.len() < before,
            "a tight repetition loop must still be trimmed"
        );
    }

    // -----------------------------------------------------------------------
    // Tests: punctuation must not hide a repeat
    //
    // Whisper attaches punctuation to the word, and PunctuationStripper runs
    // *after* these filters, so "Ganyat?" and "Ganyat" arrive here as different
    // strings. Comparisons have to look past it.
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_ignores_trailing_punctuation() {
        let mut words = vec![word("Ganyat?"), word("Ganyat")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["Ganyat?"]);
    }

    #[test]
    fn dedup_ignores_trailing_period() {
        let mut words = vec![word("Great."), word("great")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["Great."]);
    }

    #[test]
    fn dedup_keeps_genuinely_different_words() {
        let mut words = vec![word("great."), word("greater")];
        deduplicate_consecutive_repeats(&mut words);
        assert_eq!(texts(&words), &["great.", "greater"]);
    }

    #[test]
    fn ngram_matches_across_trailing_punctuation() {
        // Three copies of one phrase inside a few seconds, but the middle copy
        // carries a comma. Without normalisation the loop stays hidden.
        let mut words = Vec::new();
        words.extend(words_at(
            &["thank", "you", "very", "much", "everyone"],
            10.0,
        ));
        words.extend(words_at(
            &["thank,", "you", "very", "much", "everyone"],
            12.0,
        ));
        words.extend(words_at(
            &["thank", "you", "very", "much", "everyone"],
            14.0,
        ));

        let before = words.len();
        remove_ngram_repetitions(&mut words);

        assert!(
            words.len() < before,
            "a punctuated copy must not hide the loop"
        );
    }

    // -----------------------------------------------------------------------
    // Tests: repetition-ratio pruning over long transcripts
    // -----------------------------------------------------------------------

    #[test]
    fn prune_tail_keeps_long_varied_speech() {
        // Natural speech reuses common words constantly; across 15k words the
        // global unique/total ratio sits near 0.15, well under the 0.3 threshold.
        let vocabulary = [
            "we",
            "are",
            "going",
            "to",
            "look",
            "at",
            "the",
            "questions",
            "you",
            "have",
            "asked",
            "and",
            "answer",
            "them",
            "in",
            "themes",
            "today",
        ];
        let mut words: Vec<Word> = Vec::new();
        for i in 0..600 {
            let text = vocabulary[i % vocabulary.len()];
            let start = i as f64 * 0.4;
            words.push(Word {
                text: text.to_string(),
                start,
                end: start + 0.35,
                confidence: 0.99,
            });
        }

        let before = words.len();
        prune_repetitive_tail(&mut words);

        assert_eq!(
            words.len(),
            before,
            "ordinary long-form speech must not be pruned"
        );
    }

    #[test]
    fn prune_tail_still_trims_degenerate_ending() {
        let mut words = words_at(
            &[
                "so",
                "that",
                "is",
                "everything",
                "we",
                "wanted",
                "to",
                "cover",
                "with",
                "you",
            ],
            0.0,
        );
        for i in 0..60 {
            let start = 100.0 + i as f64 * 0.4;
            words.push(Word {
                text: "okay".to_string(),
                start,
                end: start + 0.35,
                confidence: 0.99,
            });
        }

        let before = words.len();
        prune_repetitive_tail(&mut words);

        assert!(
            words.len() < before,
            "a degenerate repeated tail must still be pruned"
        );
    }

    #[test]
    fn ngram_case_insensitive() {
        let mut words = vec![
            word("A"),
            word("B"),
            word("C"),
            word("D"),
            word("E"),
            word("f"),
            word("a"),
            word("b"),
            word("c"),
            word("d"),
            word("e"),
            word("g"),
            word("A"),
            word("B"),
            word("C"),
            word("D"),
            word("E"),
        ];
        remove_ngram_repetitions(&mut words);
        assert_eq!(
            texts(&words),
            &["A", "B", "C", "D", "E", "f", "a", "b", "c", "d", "e", "g"]
        );
    }

    #[test]
    fn ngram_exactly_five_words_no_repeat() {
        let mut words = vec![
            word("one"),
            word("two"),
            word("three"),
            word("four"),
            word("five"),
        ];
        remove_ngram_repetitions(&mut words);
        assert_eq!(texts(&words), &["one", "two", "three", "four", "five"]);
    }

    #[test]
    fn moderate_runs_ngram_detection() {
        let mut words = vec![
            word("real"),
            word("content"),
            word("here"),
            word("now"),
            word("ok"),
            word("x"),
            word("real"),
            word("content"),
            word("here"),
            word("now"),
            word("ok"),
            word("y"),
            word("real"),
            word("content"),
            word("here"),
            word("now"),
            word("ok"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(
            texts(&words),
            &[
                "real", "content", "here", "now", "ok", "x", "real", "content", "here", "now",
                "ok", "y"
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Aggressive -- low confidence filtering
    // -----------------------------------------------------------------------

    #[test]
    fn aggressive_removes_very_low_confidence_words() {
        let mut words = vec![
            word_conf("hello", 0.95),
            word_conf("um", 0.05),
            word_conf("world", 0.88),
            word_conf("xyz", 0.09),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Aggressive,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["hello", "world"]);
    }

    #[test]
    fn aggressive_keeps_words_at_or_above_threshold() {
        let mut words = vec![word_conf("hello", 0.10), word_conf("world", 0.11)];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Aggressive,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["hello", "world"]);
    }

    #[test]
    fn aggressive_empty_list_stays_empty() {
        let mut words: Vec<Word> = vec![];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Aggressive,
        };
        filter.process(&mut words);
        assert!(words.is_empty());
    }

    // -----------------------------------------------------------------------
    // Aggressive -- repetition-ratio pruning
    // -----------------------------------------------------------------------

    #[test]
    fn aggressive_prunes_highly_repetitive_tail() {
        let mut words: Vec<Word> = (0..10).map(|_| word("a")).collect();
        prune_repetitive_tail(&mut words);
        assert_eq!(texts(&words), &["a"]);
    }

    #[test]
    fn aggressive_does_not_prune_low_repetition_content() {
        let mut words = vec![
            word("hello"),
            word("beautiful"),
            word("wonderful"),
            word("amazing"),
            word("world"),
        ];
        prune_repetitive_tail(&mut words);
        assert_eq!(words.len(), 5);
    }

    #[test]
    fn aggressive_repetition_check_requires_minimum_words() {
        let mut words = vec![word("a"), word("a"), word("a"), word("a")];
        prune_repetitive_tail(&mut words);
        assert_eq!(words.len(), 4);
    }

    #[test]
    fn moderate_mode_does_not_run_confidence_filter() {
        let mut words = vec![word_conf("hello", 0.01), word_conf("world", 0.02)];
        let filter = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(words.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Allow-list
    // -----------------------------------------------------------------------

    #[test]
    fn allowed_phrase_is_not_removed() {
        let mut words = vec![
            word("Great"),
            word("video"),
            word("thanks"),
            word("for"),
            word("watching"),
        ];
        let filter = HallucinationFilter {
            allowed_phrases: vec!["thanks for watching".to_string()],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(
            texts(&words),
            &["Great", "video", "thanks", "for", "watching"]
        );
    }

    #[test]
    fn allowed_phrase_is_case_insensitive() {
        let mut words = vec![word("Hello"), word("Thank"), word("You")];
        let filter = HallucinationFilter {
            allowed_phrases: vec!["thank you".to_string()],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Hello", "Thank", "You"]);
    }

    #[test]
    fn non_allowed_phrase_still_removed() {
        let mut words = vec![word("Hello"), word("please"), word("subscribe")];
        let filter = HallucinationFilter {
            allowed_phrases: vec!["thanks for watching".to_string()],
            mode: HallucinationMode::Moderate,
        };
        filter.process(&mut words);
        assert_eq!(texts(&words), &["Hello"]);
    }

    // -----------------------------------------------------------------------
    // Default mode
    // -----------------------------------------------------------------------

    #[test]
    fn default_mode_is_moderate() {
        let mode = HallucinationMode::default();
        assert!(matches!(mode, HallucinationMode::Moderate));
    }
}
