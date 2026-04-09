// WER = (substitutions + insertions + deletions) / reference_word_count

fn normalize(text: &str) -> String {
    let lowered = text.to_lowercase();
    let stripped: String = lowered
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn calculate_wer(reference: &str, hypothesis: &str) -> f64 {
    let ref_norm = normalize(reference);
    let hyp_norm = normalize(hypothesis);

    let ref_words: Vec<&str> = if ref_norm.is_empty() {
        Vec::new()
    } else {
        ref_norm.split(' ').collect()
    };

    let hyp_words: Vec<&str> = if hyp_norm.is_empty() {
        Vec::new()
    } else {
        hyp_norm.split(' ').collect()
    };

    let n = ref_words.len();
    let m = hyp_words.len();

    if n == 0 && m == 0 {
        return 0.0;
    }

    if n == 0 {
        return m as f64;
    }

    // dp[i][j] = edit distance between ref_words[..i] and hyp_words[..j]
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate() {
        *val = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if ref_words[i - 1] == hyp_words[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    let edit_distance = dp[n][m];
    edit_distance as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_empty_returns_zero() {
        assert!((calculate_wer("", "") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn identical_strings_returns_zero() {
        assert!((calculate_wer("hello world", "hello world") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn one_substitution_out_of_two() {
        let wer = calculate_wer("hello world", "hello word");
        assert!((wer - 0.5).abs() < f64::EPSILON, "Expected 0.5, got {wer}");
    }

    #[test]
    fn one_deletion_out_of_three() {
        let wer = calculate_wer("the cat sat", "the cat");
        assert!(
            (wer - 1.0 / 3.0).abs() < 1e-10,
            "Expected 0.333..., got {wer}"
        );
    }

    #[test]
    fn one_insertion_out_of_two() {
        let wer = calculate_wer("the cat", "the big cat");
        assert!((wer - 0.5).abs() < f64::EPSILON, "Expected 0.5, got {wer}");
    }

    #[test]
    fn case_and_punctuation_normalization() {
        let wer = calculate_wer("Hello, World!", "hello world");
        assert!((wer - 0.0).abs() < f64::EPSILON, "Expected 0.0, got {wer}");
    }

    #[test]
    fn empty_reference_nonempty_hypothesis() {
        let wer = calculate_wer("", "hello world");
        assert!((wer - 2.0).abs() < f64::EPSILON, "Expected 2.0, got {wer}");
    }

    #[test]
    fn nonempty_reference_empty_hypothesis() {
        let wer = calculate_wer("hello world", "");
        assert!((wer - 1.0).abs() < f64::EPSILON, "Expected 1.0, got {wer}");
    }

    #[test]
    fn completely_different_words() {
        let wer = calculate_wer("the cat sat", "a dog ran");
        assert!((wer - 1.0).abs() < f64::EPSILON, "Expected 1.0, got {wer}");
    }

    #[test]
    fn extra_whitespace_is_collapsed() {
        let wer = calculate_wer("  hello   world  ", "hello world");
        assert!((wer - 0.0).abs() < f64::EPSILON, "Expected 0.0, got {wer}");
    }

    #[test]
    fn mixed_punctuation_stripped() {
        let wer = calculate_wer("it's a test!", "its a test");
        assert!((wer - 0.0).abs() < f64::EPSILON, "Expected 0.0, got {wer}");
    }

    #[test]
    fn normalize_lowercases() {
        assert_eq!(normalize("HELLO World"), "hello world");
    }

    #[test]
    fn normalize_strips_punctuation() {
        assert_eq!(normalize("Hello, world!"), "hello world");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize("  hello   world  "), "hello world");
    }

    #[test]
    fn normalize_empty_string() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn normalize_only_punctuation() {
        assert_eq!(normalize("...!?"), "");
    }
}
