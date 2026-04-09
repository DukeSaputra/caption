pub mod wer;

use std::time::Duration;

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub clip_name: String,
    pub reference_word_count: usize,
    pub wer: f64,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct BenchSummary {
    pub results: Vec<BenchResult>,
    pub mean_wer: f64,
    pub median_wer: f64,
    pub worst_wer: f64,
    pub mean_duration: Duration,
}

pub fn summarize(results: Vec<BenchResult>) -> BenchSummary {
    assert!(!results.is_empty(), "cannot summarize zero results");

    let count = results.len();

    let total_wer: f64 = results.iter().map(|r| r.wer).sum();
    let mean_wer = total_wer / count as f64;

    let mut wers: Vec<f64> = results.iter().map(|r| r.wer).collect();
    wers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_wer = if count % 2 == 1 {
        wers[count / 2]
    } else {
        (wers[count / 2 - 1] + wers[count / 2]) / 2.0
    };

    let worst_wer = wers.last().copied().unwrap_or(0.0);

    let total_duration: Duration = results.iter().map(|r| r.duration).sum();
    let mean_duration = total_duration / count as u32;

    BenchSummary {
        results,
        mean_wer,
        median_wer,
        worst_wer,
        mean_duration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(name: &str, wer: f64, secs: u64) -> BenchResult {
        BenchResult {
            clip_name: name.to_string(),
            reference_word_count: 10,
            wer,
            duration: Duration::from_secs(secs),
        }
    }

    #[test]
    fn single_result_summary() {
        let summary = summarize(vec![result("a.wav", 0.05, 3)]);
        assert_eq!(summary.results.len(), 1);
        assert!((summary.mean_wer - 0.05).abs() < f64::EPSILON);
        assert!((summary.median_wer - 0.05).abs() < f64::EPSILON);
        assert!((summary.worst_wer - 0.05).abs() < f64::EPSILON);
        assert_eq!(summary.mean_duration, Duration::from_secs(3));
    }

    #[test]
    fn multiple_results_odd_count() {
        let summary = summarize(vec![
            result("a.wav", 0.02, 2),
            result("b.wav", 0.05, 3),
            result("c.wav", 0.10, 4),
        ]);
        assert_eq!(summary.results.len(), 3);

        let expected_mean = (0.02 + 0.05 + 0.10) / 3.0;
        assert!(
            (summary.mean_wer - expected_mean).abs() < 1e-10,
            "Expected mean {expected_mean}, got {}",
            summary.mean_wer
        );

        assert!(
            (summary.median_wer - 0.05).abs() < f64::EPSILON,
            "Expected median 0.05, got {}",
            summary.median_wer
        );

        assert!(
            (summary.worst_wer - 0.10).abs() < f64::EPSILON,
            "Expected worst 0.10, got {}",
            summary.worst_wer
        );

        assert_eq!(summary.mean_duration, Duration::from_secs(3));
    }

    #[test]
    fn multiple_results_even_count() {
        let summary = summarize(vec![
            result("a.wav", 0.02, 2),
            result("b.wav", 0.04, 3),
            result("c.wav", 0.06, 4),
            result("d.wav", 0.08, 5),
        ]);

        assert!(
            (summary.median_wer - 0.05).abs() < 1e-10,
            "Expected median 0.05, got {}",
            summary.median_wer
        );

        assert!(
            (summary.worst_wer - 0.08).abs() < f64::EPSILON,
            "Expected worst 0.08, got {}",
            summary.worst_wer
        );
    }

    #[test]
    fn all_perfect_results() {
        let summary = summarize(vec![result("a.wav", 0.0, 1), result("b.wav", 0.0, 2)]);
        assert!((summary.mean_wer - 0.0).abs() < f64::EPSILON);
        assert!((summary.median_wer - 0.0).abs() < f64::EPSILON);
        assert!((summary.worst_wer - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unsorted_input_still_correct_median() {
        let summary = summarize(vec![
            result("c.wav", 0.10, 4),
            result("a.wav", 0.02, 2),
            result("b.wav", 0.05, 3),
        ]);
        assert!(
            (summary.median_wer - 0.05).abs() < f64::EPSILON,
            "Expected median 0.05, got {}",
            summary.median_wer
        );
    }

    #[test]
    #[should_panic(expected = "cannot summarize zero results")]
    fn empty_results_panics() {
        summarize(vec![]);
    }
}
