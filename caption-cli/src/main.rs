mod download;

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::debug;

use caption_core::align::ctc_aligner::CtcAligner;
use caption_core::audio::ffmpeg_fallback::find_ffmpeg_public;
use caption_core::burn::encode::{run_burn, BurnConfig};
use caption_core::burn::font::find_font;
use caption_core::burn::probe::probe_video;

#[cfg(unix)]
fn suppress_stderr() -> Option<i32> {
    use std::os::unix::io::AsRawFd;
    unsafe {
        let devnull = libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
        if devnull < 0 {
            return None;
        }
        let saved = libc::dup(std::io::stderr().as_raw_fd());
        if saved < 0 {
            libc::close(devnull);
            return None;
        }
        libc::dup2(devnull, std::io::stderr().as_raw_fd());
        libc::close(devnull);
        Some(saved)
    }
}

#[cfg(not(unix))]
fn suppress_stderr() -> Option<i32> {
    None
}

#[cfg(unix)]
fn restore_stderr(saved: Option<i32>) {
    use std::os::unix::io::AsRawFd;
    if let Some(fd) = saved {
        unsafe {
            libc::dup2(fd, std::io::stderr().as_raw_fd());
            libc::close(fd);
        }
    }
}

#[cfg(not(unix))]
fn restore_stderr(_saved: Option<i32>) {}
use caption_core::align::ForcedAligner;
use caption_core::audio::extract::extract_audio;
use caption_core::audio::resample::prepare_for_stt;
use caption_core::bench::wer::calculate_wer;
use caption_core::bench::{summarize, BenchResult};
use caption_core::error::CaptionError;
use caption_core::format::ass::AssFormatter;
use caption_core::format::parse::parse_srt;
use caption_core::format::srt::SrtFormatter;
use caption_core::format::text::TextFormatter;
use caption_core::format::vtt::VttFormatter;
use caption_core::format::{FormatConfig, SubtitleFormatter};
use caption_core::pipeline::{transcribe_pipeline, PipelineConfig};
use caption_core::postprocess::fillers::{FillerMode, FillerRemover};
use caption_core::postprocess::hallucination::{HallucinationFilter, HallucinationMode};
use caption_core::postprocess::profanity::{ProfanityFilter, ProfanityMode};
use caption_core::postprocess::punctuation::PunctuationStripper;
use caption_core::postprocess::substitution::{parse_hot_words, SubstitutionMap};
use caption_core::postprocess::{run_pipeline as run_postprocessing, PostProcessor};
#[cfg(feature = "parakeet")]
use caption_core::stt::model::find_parakeet_model;
use caption_core::stt::model::{
    find_aligner_model, find_hot_words_file, find_model, find_vad_model,
};
#[cfg(feature = "parakeet")]
use caption_core::stt::parakeet_backend::ParakeetBackend;
use caption_core::stt::whisper_backend::WhisperBackend;
use caption_core::stt::{SpeechRecognizer, TranscribeConfig};
use caption_core::vad::StandaloneVad;

fn ort_not_found_message(feature: &str) -> String {
    if cfg!(target_os = "macos") {
        format!(
            "ONNX Runtime not found; {feature} unavailable. \
             Run `caption setup` to install, or: brew install onnxruntime"
        )
    } else {
        format!(
            "ONNX Runtime not found; {feature} unavailable. \
             Run `caption setup` to install it."
        )
    }
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Generate RSVP subtitles from video files using local speech-to-text"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(help = "Path to the input video or audio file")]
    input: Option<PathBuf>,

    #[arg(short, long, help = "Path for the output subtitle file")]
    output: Option<PathBuf>,

    #[arg(long, default_value = "srt", value_parser = ["srt", "vtt", "ass", "txt"], help = "Output subtitle format")]
    format: String,

    #[arg(long, default_value = "whisper", value_parser = ["whisper", "parakeet"], help = "STT engine to use for transcription", hide = true)]
    engine: String,

    #[arg(
        long,
        help = "Explicit path to a Whisper GGML model file or Parakeet model directory"
    )]
    model: Option<String>,

    #[arg(long, help = "Explicit path to a Silero VAD model file")]
    vad_model: Option<String>,

    #[arg(
        long,
        help = "Disable Voice Activity Detection even if a VAD model is found"
    )]
    no_vad: bool,

    #[arg(long, help = "Explicit path to a CTC aligner ONNX model file")]
    aligner_model: Option<String>,

    #[arg(
        long,
        help = "Disable forced alignment; fall back to STT backend timestamps"
    )]
    no_align: bool,

    #[arg(long, help = "Prompt to bias Whisper toward specific vocabulary")]
    initial_prompt: Option<String>,

    #[arg(long, default_value = "remove-confident", value_parser = ["keep", "remove-confident", "remove-all"], help = "Filler word removal mode")]
    fillers: String,

    #[arg(long, default_value = "keep", value_parser = ["keep", "mask", "replace"], help = "Profanity filtering mode")]
    profanity: String,

    #[arg(long, default_value = "moderate", value_parser = ["off", "moderate", "aggressive"], help = "Hallucination detection aggressiveness")]
    hallucination_filter: String,

    #[arg(long, action = clap::ArgAction::Append, help = "Allow specific trailing phrases past the hallucination filter (repeatable)")]
    allow_phrase: Vec<String>,

    #[arg(long, help = "Explicit path to a hot-words file")]
    hot_words: Option<String>,

    #[arg(short, long, help = "Enable verbose debug logging")]
    verbose: bool,

    #[arg(
        long,
        help = "Burn subtitles into the video with a frosted glass capsule effect"
    )]
    burn: bool,

    #[arg(
        long,
        help = "Path to an existing SRT file to burn instead of transcribing (implies --burn)"
    )]
    srt: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Run accuracy benchmarks against a test corpus")]
    Bench {
        #[arg(long, help = "Path to the corpus directory")]
        corpus: PathBuf,
        #[arg(long, help = "Explicit path to a Whisper model file")]
        model: Option<String>,
        #[arg(long, help = "Explicit path to a VAD model file")]
        vad_model: Option<String>,
        #[arg(long, help = "Disable VAD")]
        no_vad: bool,
        #[arg(short, long, help = "Enable verbose logging")]
        verbose: bool,
    },
    #[command(about = "Download a pre-trained model")]
    DownloadModel {
        #[arg(
            help = "Model name: whisper-large-v3-turbo, silero-vad, wav2vec2-aligner, parakeet-tdt"
        )]
        name: String,
        #[arg(long, help = "Skip SHA256 hash verification")]
        skip_hash: bool,
    },
    #[command(about = "Download all required models (whisper + vad + aligner)")]
    Setup {
        #[arg(long, help = "Skip SHA256 hash verification")]
        skip_hash: bool,
    },
    #[command(about = "Remove caption and all downloaded models")]
    Uninstall,
}

fn main() {
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        let bundled = std::env::current_exe().ok().and_then(|exe| {
            let exe = exe.canonicalize().unwrap_or(exe);
            let dir = exe.parent()?;
            for name in &[
                "libonnxruntime.dylib",
                "libonnxruntime.so",
                "onnxruntime.dll",
            ] {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            None
        });

        if let Some(path) = bundled {
            std::env::set_var("ORT_DYLIB_PATH", &path);
        } else {
            let homebrew_path = "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib";
            if std::path::Path::new(homebrew_path).exists() {
                std::env::set_var("ORT_DYLIB_PATH", homebrew_path);
            }
        }
    }

    let cli = Cli::parse();

    let verbose = match &cli.command {
        Some(Commands::Bench { verbose, .. }) => *verbose,
        Some(Commands::DownloadModel { .. } | Commands::Setup { .. } | Commands::Uninstall) => {
            cli.verbose
        }
        None => cli.verbose,
    };

    let log_level = if verbose { "debug" } else { "error" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

    let result = match &cli.command {
        Some(Commands::Bench {
            corpus,
            model,
            vad_model,
            no_vad,
            ..
        }) => run_bench(corpus, model.as_deref(), vad_model.as_deref(), *no_vad),
        Some(Commands::DownloadModel { name, skip_hash }) => {
            download::download_model(name, *skip_hash).map(|_| ())
        }
        Some(Commands::Setup { skip_hash }) => run_setup(*skip_hash),
        Some(Commands::Uninstall) => run_uninstall(),
        None => run_transcribe(&cli),
    };

    if let Err(err) = result {
        eprintln!("\x1b[31merror:\x1b[0m {err:#}");
        process::exit(1);
    }
}

fn run_setup(skip_hash: bool) -> Result<()> {
    for name in &["whisper-large-v3-turbo", "silero-vad", "wav2vec2-aligner"] {
        download::download_model(name, skip_hash)?;
    }
    download::setup_dependencies()?;
    eprintln!("\nSetup complete. Run `caption --help` to get started.");
    Ok(())
}

fn run_uninstall() -> Result<()> {
    let exe = std::env::current_exe().context("Could not determine binary location")?;
    let exe = exe.canonicalize().unwrap_or(exe);
    let exe_dir = exe
        .parent()
        .context("Could not determine binary directory")?;

    let files_to_remove = [
        "ggml-large-v3-turbo-q8_0.bin",
        "ggml-large-v3-turbo-q5_0.bin",
        "silero_vad.onnx",
        "wav2vec2-base-960h.onnx",
        "ffmpeg",
        "ffmpeg.exe",
        "inter-bold.ttf",
        "libonnxruntime.dylib",
        "libonnxruntime.so",
        "onnxruntime.dll",
    ];
    for name in &files_to_remove {
        let path = exe_dir.join(name);
        if path.is_file() {
            std::fs::remove_file(&path)
                .with_context(|| format!("Failed to remove {}", path.display()))?;
            eprintln!("Removed {}", path.display());
        }
    }

    if let Some(data_dir) = dirs::data_dir() {
        let models_dir = data_dir.join("caption").join("models");
        if models_dir.is_dir() {
            std::fs::remove_dir_all(&models_dir)
                .with_context(|| format!("Failed to remove {}", models_dir.display()))?;
            eprintln!("Removed {}", models_dir.display());
        }
        let caption_dir = data_dir.join("caption");
        if caption_dir.is_dir() {
            let _ = std::fs::remove_dir(&caption_dir);
        }
    }

    eprintln!("\nAll models and dependencies removed.");
    eprintln!("To finish, delete the binary itself:");
    eprintln!("  rm {}", exe.display());

    Ok(())
}

fn run_burn_from_srt(cli: &Cli, input: &Path, srt_path: &Path) -> Result<()> {
    let start_time = Instant::now();

    let srt_content = std::fs::read_to_string(srt_path)
        .with_context(|| format!("Failed to read SRT file '{}'", srt_path.display()))?;
    let parsed = parse_srt(&srt_content)
        .with_context(|| format!("Failed to parse SRT file '{}'", srt_path.display()))?;
    let mut words = parsed.words;

    let ffmpeg_path = find_ffmpeg_public()
        .context("--srt requires FFmpeg. It should be bundled next to the caption binary; if missing, install with: brew install ffmpeg")?;

    let font_path = find_font(None)
        .context("--srt requires a bundled font (inter-bold.ttf) but none was found")?;

    let spinner = if !cli.verbose {
        let sp = indicatif::ProgressBar::new_spinner();
        sp.set_draw_target(indicatif::ProgressDrawTarget::stdout());
        sp.set_style(
            indicatif::ProgressStyle::with_template("{spinner:.dim} {msg} {elapsed:.dim}")
                .expect("valid template")
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
        );
        sp.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(sp)
    } else {
        None
    };

    let stderr_saved = if !cli.verbose {
        suppress_stderr()
    } else {
        None
    };

    if parsed.had_multi_word_cues && !cli.no_align {
        if let Some(sp) = &spinner {
            sp.set_message("Extracting Audio");
        }

        let audio = extract_audio(input).context("Failed to extract audio for alignment")?;
        let stt_samples =
            prepare_for_stt(audio).context("Failed to resample audio for alignment")?;

        if let Some(sp) = &spinner {
            sp.set_message("Aligning Words");
        }

        let aligner_result = match find_aligner_model(cli.aligner_model.as_deref()) {
            Ok(aligner_path) => {
                debug!("Using aligner model: {}", aligner_path.display());
                let path = aligner_path.clone();
                let prev_hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {}));
                let result = std::panic::catch_unwind(|| CtcAligner::new(&path));
                std::panic::set_hook(prev_hook);
                match result {
                    Ok(Ok(a)) => Some(a),
                    Ok(Err(e)) => {
                        debug!("Failed to load aligner: {e}");
                        None
                    }
                    Err(panic_info) => {
                        let reason = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown");
                        debug!("ONNX Runtime panicked loading aligner: {reason}");
                        None
                    }
                }
            }
            Err(_) => {
                debug!("No aligner model found");
                None
            }
        };

        if let Some(mut aligner) = aligner_result {
            let word_texts: Vec<String> = words.iter().map(|w| w.text.clone()).collect();
            match aligner.align(&stt_samples, 16_000, &word_texts) {
                Ok(alignments) => {
                    debug!(
                        "Aligned {} words from multi-word SRT cues",
                        alignments.len()
                    );
                    for (word, aligned) in words.iter_mut().zip(alignments.iter()) {
                        word.start = aligned.start;
                        word.end = aligned.end;
                    }
                }
                Err(e) => {
                    debug!("Alignment failed, using proportional timestamps: {e}");
                }
            }
        } else {
            debug!("No aligner available, using proportional timestamps from SRT");
        }
    }

    if let Some(sp) = &spinner {
        sp.set_message("Probing Video");
    }

    let video_info = probe_video(input, &ffmpeg_path).context("Failed to read video properties")?;
    debug!(
        "Video: {}x{} @ {:.2} fps",
        video_info.width, video_info.height, video_info.fps
    );

    let font_data = std::fs::read(&font_path)
        .with_context(|| format!("Failed to read font file '{}'", font_path.display()))?;

    if let Some(sp) = &spinner {
        sp.set_message("Burning Subtitles");
    }

    let input_stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let burn_output = input.with_file_name(format!("{input_stem}-captioned.mp4"));

    let burn_config = BurnConfig {
        input_path: input,
        output_path: &burn_output,
        ffmpeg_path: &ffmpeg_path,
        font_data: &font_data,
        video_info: &video_info,
        words: &words,
    };

    let burn_result = run_burn(&burn_config);

    restore_stderr(stderr_saved);
    if let Some(sp) = &spinner {
        sp.finish_and_clear();
    }

    burn_result.context("Failed to burn subtitles")?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let burn_name = burn_output
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| burn_output.display().to_string());
    let srt_name = srt_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| srt_path.display().to_string());

    eprintln!(
        "\x1b[32m+\x1b[0m {burn_name}  from {srt_name}  {} words  ({elapsed:.1}s)",
        words.len(),
    );

    Ok(())
}

fn run_transcribe(cli: &Cli) -> Result<()> {
    let input = cli
        .input
        .as_ref()
        .context("Missing required argument: <INPUT>. Provide a path to a video or audio file.")?;

    if let Some(srt_path) = &cli.srt {
        return run_burn_from_srt(cli, input, srt_path);
    }

    let formatter: Box<dyn SubtitleFormatter> = match cli.format.as_str() {
        "srt" => Box::new(SrtFormatter),
        "vtt" => Box::new(VttFormatter),
        "ass" => Box::new(AssFormatter),
        "txt" => Box::new(TextFormatter),
        other => anyhow::bail!("Unknown output format '{other}'."),
    };

    let output_path = cli
        .output
        .clone()
        .unwrap_or_else(|| input.with_extension(formatter.file_extension()));

    debug!("Input file: {}", input.display());
    debug!("Output file: {}", output_path.display());

    let start_time = Instant::now();

    let spinner = if !cli.verbose {
        let sp = indicatif::ProgressBar::new_spinner();
        sp.set_draw_target(indicatif::ProgressDrawTarget::stdout());
        sp.set_style(
            indicatif::ProgressStyle::with_template("{spinner:.dim} {msg} {elapsed:.dim}")
                .expect("valid template")
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
        );
        sp.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(sp)
    } else {
        None
    };

    let stderr_saved = if !cli.verbose {
        suppress_stderr()
    } else {
        None
    };

    if let Some(sp) = &spinner {
        sp.set_message("Extracting Audio");
    }

    let audio = extract_audio(input).map_err(|e| match e {
        CaptionError::FileNotFound(ref path) => {
            anyhow::anyhow!("Could not open '{}': file not found", path)
        }
        CaptionError::NoAudioTrack => {
            anyhow::anyhow!(
                "No audio track found in '{}'. Is this a valid video/audio file?",
                input.display()
            )
        }
        CaptionError::UnsupportedFormat(ref fmt) => {
            anyhow::anyhow!(
                "Unsupported audio format ({}) in '{}'. This codec is not yet supported.",
                fmt,
                input.display()
            )
        }
        CaptionError::ExtractionFailed(ref msg) => {
            anyhow::anyhow!(
                "Failed to extract audio from '{}': {}",
                input.display(),
                msg
            )
        }
        other => anyhow::anyhow!("{}", other),
    })?;

    debug!(
        "Extracted audio: {} samples, {} Hz, {} channels",
        audio.samples.len(),
        audio.sample_rate,
        audio.channels
    );

    if let Some(sp) = &spinner {
        sp.set_message("Preparing Audio");
    }

    let stt_samples =
        prepare_for_stt(audio).context("Failed to prepare audio for speech-to-text")?;

    let duration_secs = stt_samples.len() as f64 / 16_000.0;

    debug!(
        "Prepared for STT: {} samples ({:.2}s at 16 kHz)",
        stt_samples.len(),
        duration_secs
    );

    let mut warnings: Vec<String> = Vec::new();

    if let Some(sp) = &spinner {
        sp.set_message("Loading Models");
    }

    let backend: Box<dyn SpeechRecognizer> = match cli.engine.as_str() {
        #[cfg(feature = "parakeet")]
        "parakeet" => {
            let model_dir = find_parakeet_model(cli.model.as_deref())
                .context("Could not find a Parakeet TDT model directory")?;
            debug!("Using Parakeet model: {}", model_dir.display());
            Box::new(
                ParakeetBackend::new(&model_dir)
                    .context("Failed to load the Parakeet TDT model")?,
            )
        }
        #[cfg(not(feature = "parakeet"))]
        "parakeet" => {
            anyhow::bail!("Parakeet engine not available in this build. Rebuild with: cargo build --release --features parakeet");
        }
        _ => {
            let model_path =
                find_model(cli.model.as_deref()).context("Could not find a Whisper model file")?;
            debug!("Using Whisper model: {}", model_path.display());
            Box::new(WhisperBackend::new(&model_path).context("Failed to load the Whisper model")?)
        }
    };

    let hot_words_path = match &cli.hot_words {
        Some(explicit) => {
            let p = Path::new(explicit);
            if p.is_file() {
                Some(p.to_path_buf())
            } else {
                anyhow::bail!("Hot-words file not found at '{explicit}'");
            }
        }
        None => find_hot_words_file(),
    };

    let (vocab_prompt, substitution_map) = match &hot_words_path {
        Some(path) => {
            debug!("Using hot-words file: {}", path.display());
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read hot-words file '{}'", path.display()))?;
            let hot = parse_hot_words(&content);

            let prompt = if hot.vocab_hints.is_empty() {
                None
            } else {
                debug!("Hot-words vocab hints: {}", hot.vocab_hints.join(", "));
                Some(hot.vocab_hints.join(", "))
            };

            let mut subs = SubstitutionMap::with_defaults();
            for (k, v) in hot.substitutions {
                subs.replacements.insert(k, v);
            }
            (prompt, subs)
        }
        None => {
            debug!("No hot-words file found");
            (None, SubstitutionMap::with_defaults())
        }
    };

    let initial_prompt = match (&cli.initial_prompt, &vocab_prompt) {
        (Some(cli_prompt), Some(vocab)) => Some(format!("{cli_prompt}, {vocab}")),
        (Some(cli_prompt), None) => Some(cli_prompt.clone()),
        (None, Some(vocab)) => Some(vocab.clone()),
        (None, None) => None,
    };

    let vad_model_path = if cli.engine == "parakeet" || cli.no_vad {
        if cli.no_vad {
            debug!("VAD disabled via --no-vad");
        } else {
            debug!("VAD not used with Parakeet engine");
        }
        None
    } else {
        match find_vad_model(cli.vad_model.as_deref()) {
            Ok(path) => {
                debug!("Using VAD model: {}", path.display());
                Some(path.to_string_lossy().into_owned())
            }
            Err(_) => {
                if cli.vad_model.is_some() {
                    anyhow::bail!(
                        "VAD model not found at '{}'",
                        cli.vad_model.as_deref().unwrap_or("")
                    );
                }
                debug!("No VAD model found; proceeding without VAD");
                warnings.push("No VAD model found; speech detection skipped".into());
                None
            }
        }
    };

    let mut standalone_vad = if cli.no_vad || cli.engine == "parakeet" {
        None
    } else {
        match &vad_model_path {
            Some(path) => {
                let vad_path = Path::new(path).to_path_buf();
                let prev_hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {}));
                let result = std::panic::catch_unwind(|| StandaloneVad::new(&vad_path));
                std::panic::set_hook(prev_hook);
                match result {
                    Ok(Ok(vad)) => {
                        debug!("Standalone VAD loaded from: {path}");
                        Some(vad)
                    }
                    Ok(Err(e)) => {
                        debug!("Failed to load standalone VAD, proceeding without: {e}");
                        warnings.push(format!("Failed to load VAD model: {e}"));
                        None
                    }
                    Err(panic_info) => {
                        let reason = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown");
                        debug!("ONNX Runtime panicked loading VAD: {reason}");
                        warnings.push(ort_not_found_message("VAD and alignment"));
                        None
                    }
                }
            }
            None => None,
        }
    };

    let skip_align = cli.no_align || cli.format == "txt";
    let mut aligner: Option<CtcAligner> = if skip_align {
        if cli.no_align {
            debug!("Forced alignment disabled via --no-align");
        } else {
            debug!("Forced alignment skipped for plain text output");
        }
        None
    } else {
        match find_aligner_model(cli.aligner_model.as_deref()) {
            Ok(aligner_path) => {
                debug!("Using aligner model: {}", aligner_path.display());
                let path = aligner_path.clone();
                let prev_hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {}));
                let result = std::panic::catch_unwind(|| CtcAligner::new(&path));
                std::panic::set_hook(prev_hook);
                match result {
                    Ok(Ok(a)) => Some(a),
                    Ok(Err(e)) => {
                        debug!("Failed to load aligner model, using backend timestamps: {e}");
                        warnings.push(format!(
                            "Failed to load aligner model: {e}. Word timestamps may be less precise."
                        ));
                        None
                    }
                    Err(panic_info) => {
                        let reason = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown");
                        debug!("ONNX Runtime panicked loading aligner: {reason}");
                        if !warnings.iter().any(|w| w.contains("ONNX Runtime")) {
                            warnings.push(ort_not_found_message("alignment"));
                        }
                        None
                    }
                }
            }
            Err(_) => {
                if cli.aligner_model.is_some() {
                    anyhow::bail!(
                        "Aligner model not found at '{}'",
                        cli.aligner_model.as_deref().unwrap_or("")
                    );
                }
                debug!("No aligner model found; using backend timestamps");
                warnings.push("No aligner model found; word timestamps may be less precise".into());
                None
            }
        }
    };

    if let Some(sp) = &spinner {
        sp.set_message("Recognizing Speech");
    }

    let pipeline_config = PipelineConfig {
        transcribe_config: TranscribeConfig {
            language: "en".to_string(),
            initial_prompt,
            temperature: 0.0,
            vad_model_path,
        },
        padding_seconds: 0.3,
        min_chunk_seconds: 5.0,
        max_chunk_seconds: 30.0,
    };

    let transcription = transcribe_pipeline(
        &stt_samples,
        16_000,
        backend.as_ref(),
        standalone_vad.as_mut(),
        aligner.as_mut().map(|a| a as &mut dyn ForcedAligner),
        &pipeline_config,
    );

    restore_stderr(stderr_saved);

    let transcription = transcription.context("Transcription pipeline failed")?;
    let mut words = transcription.words;

    if let Some(sp) = &spinner {
        sp.set_message("Post-processing");
    }

    let hallucination_mode = match cli.hallucination_filter.as_str() {
        "off" => HallucinationMode::Off,
        "moderate" => HallucinationMode::Moderate,
        "aggressive" => HallucinationMode::Aggressive,
        _ => HallucinationMode::Moderate,
    };
    let filler_mode = match cli.fillers.as_str() {
        "keep" => FillerMode::KeepAll,
        "remove-confident" => FillerMode::RemoveConfident,
        "remove-all" => FillerMode::RemoveAll,
        _ => FillerMode::RemoveConfident,
    };
    let profanity_mode = match cli.profanity.as_str() {
        "keep" => ProfanityMode::KeepAll,
        "mask" => ProfanityMode::Mask,
        "replace" => ProfanityMode::Replace,
        _ => ProfanityMode::KeepAll,
    };

    let hallucination_filter = HallucinationFilter {
        mode: hallucination_mode,
        allowed_phrases: cli.allow_phrase.iter().map(|p| p.to_lowercase()).collect(),
    };
    let filler_remover = FillerRemover { mode: filler_mode };
    let profanity_filter = ProfanityFilter {
        mode: profanity_mode,
    };
    let punctuation_stripper = PunctuationStripper;
    let processors: Vec<&dyn PostProcessor> = vec![
        &hallucination_filter,
        &filler_remover,
        &punctuation_stripper,
        &profanity_filter,
        &substitution_map,
    ];
    run_postprocessing(&mut words, &processors);

    if cli.format != "txt" {
        for word in &mut words {
            word.start = (word.start - 0.10).max(0.0);
            word.end = (word.end - 0.10).max(0.0);
        }
    }

    if let Some(sp) = &spinner {
        sp.set_message("Writing Output");
    }

    let format_config = FormatConfig::default();
    let content = formatter
        .format(&words, &format_config)
        .context("Failed to format subtitles")?;

    std::fs::write(&output_path, &content)
        .with_context(|| format!("Failed to write output file '{}'", output_path.display()))?;

    if let Some(sp) = &spinner {
        sp.finish_and_clear();
    }

    for w in &warnings {
        eprintln!("\x1b[33mwarn:\x1b[0m {w}");
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let word_count = words.len();
    let unit = if cli.format == "txt" { "words" } else { "cues" };
    let mins = (duration_secs / 60.0) as u64;
    let secs = (duration_secs % 60.0) as u64;
    let output_name = output_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| output_path.display().to_string());

    eprintln!(
        "\x1b[32m+\x1b[0m {output_name}  {word_count} {unit}  {mins}:{secs:02}  ({elapsed:.1}s)",
    );

    if cli.burn {
        let start_burn = Instant::now();

        let ffmpeg_path = find_ffmpeg_public()
            .context("--burn requires FFmpeg. It should be bundled next to the caption binary; if missing, install with: brew install ffmpeg")?;

        let font_path = match find_font(None) {
            Ok(p) => {
                debug!("Using font: {}", p.display());
                p
            }
            Err(e) => {
                anyhow::bail!(
                    "--burn requires a bundled font (inter-bold.ttf) but none was found. {e}"
                );
            }
        };

        let burn_spinner = if !cli.verbose {
            let sp = indicatif::ProgressBar::new_spinner();
            sp.set_draw_target(indicatif::ProgressDrawTarget::stdout());
            sp.set_style(
                indicatif::ProgressStyle::with_template("{spinner:.dim} {msg} {elapsed:.dim}")
                    .expect("valid template")
                    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
            );
            sp.set_message("Probing Video");
            sp.enable_steady_tick(std::time::Duration::from_millis(80));
            Some(sp)
        } else {
            None
        };

        let video_info =
            probe_video(input, &ffmpeg_path).context("Failed to read video properties")?;
        debug!(
            "Video: {}x{} @ {:.2} fps",
            video_info.width, video_info.height, video_info.fps
        );

        let font_data = std::fs::read(&font_path)
            .with_context(|| format!("Failed to read font file '{}'", font_path.display()))?;

        if let Some(sp) = &burn_spinner {
            sp.set_message("Burning Subtitles");
        }

        let input_stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".to_string());
        let burn_output = input.with_file_name(format!("{input_stem}-captioned.mp4"));

        let burn_config = BurnConfig {
            input_path: input,
            output_path: &burn_output,
            ffmpeg_path: &ffmpeg_path,
            font_data: &font_data,
            video_info: &video_info,
            words: &words,
        };

        let stderr_saved_burn = if !cli.verbose {
            suppress_stderr()
        } else {
            None
        };

        let burn_result = run_burn(&burn_config);

        restore_stderr(stderr_saved_burn);
        if let Some(sp) = &burn_spinner {
            sp.finish_and_clear();
        }

        burn_result.context("Failed to burn subtitles")?;

        let burn_elapsed = start_burn.elapsed().as_secs_f64();
        let burn_name = burn_output
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| burn_output.display().to_string());

        eprintln!("\x1b[32m+\x1b[0m {burn_name}  ({burn_elapsed:.1}s)",);
    }

    Ok(())
}

fn discover_corpus_clips(corpus_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    if !corpus_dir.is_dir() {
        anyhow::bail!(
            "Corpus directory '{}' does not exist or is not a directory",
            corpus_dir.display()
        );
    }

    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();

    let entries = std::fs::read_dir(corpus_dir)
        .with_context(|| format!("Failed to read corpus directory '{}'", corpus_dir.display()))?;

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => continue,
        };

        if ext != "wav" && ext != "mp4" {
            continue;
        }

        let ground_truth = path.with_extension("txt");
        if ground_truth.is_file() {
            pairs.push((path, ground_truth));
        } else {
            debug!(
                "Skipping '{}': no matching .txt ground truth file",
                path.display()
            );
        }
    }

    pairs.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));

    if pairs.is_empty() {
        anyhow::bail!(
            "No clip/ground-truth pairs found in '{}'. \
             Each .wav or .mp4 file needs a matching .txt file with the same stem.",
            corpus_dir.display()
        );
    }

    Ok(pairs)
}

fn run_bench(
    corpus_dir: &Path,
    model_override: Option<&str>,
    vad_model_override: Option<&str>,
    no_vad: bool,
) -> Result<()> {
    let pairs = discover_corpus_clips(corpus_dir)?;

    eprintln!("Found {} clip(s) in corpus\n", pairs.len());

    let model_path = find_model(model_override).context("Could not find a Whisper model file")?;
    debug!("Using model: {}", model_path.display());

    let backend = WhisperBackend::new(&model_path).context("Failed to load the Whisper model")?;

    let vad_model_path = if no_vad {
        debug!("VAD disabled via --no-vad");
        None
    } else {
        match find_vad_model(vad_model_override) {
            Ok(path) => {
                debug!("Using VAD model: {}", path.display());
                Some(path.to_string_lossy().into_owned())
            }
            Err(_) => {
                if vad_model_override.is_some() {
                    anyhow::bail!(
                        "VAD model not found at '{}'",
                        vad_model_override.unwrap_or("")
                    );
                }
                debug!("No VAD model found; proceeding without VAD");
                None
            }
        }
    };

    let filler_remover = FillerRemover {
        mode: FillerMode::RemoveConfident,
    };
    let profanity_filter = ProfanityFilter {
        mode: ProfanityMode::KeepAll,
    };
    let processors: Vec<&dyn PostProcessor> = vec![&filler_remover, &profanity_filter];

    let text_formatter = TextFormatter;
    let format_config = FormatConfig::default();

    println!(
        "{:<30} {:>5}   {:>6}   {:>6}",
        "Clip", "Words", "WER", "Time"
    );

    let mut results: Vec<BenchResult> = Vec::new();

    for (audio_path, gt_path) in &pairs {
        let clip_name = audio_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());

        let start = Instant::now();

        let audio = extract_audio(audio_path)
            .with_context(|| format!("Failed to extract audio from '{}'", audio_path.display()))?;

        let stt_samples = prepare_for_stt(audio)
            .with_context(|| format!("Failed to resample '{}'", audio_path.display()))?;

        let config = TranscribeConfig {
            language: "en".to_string(),
            initial_prompt: None,
            temperature: 0.0,
            vad_model_path: vad_model_path.clone(),
        };

        let transcription = backend
            .transcribe(&stt_samples, &config)
            .with_context(|| format!("Transcription failed for '{}'", audio_path.display()))?;

        let mut words = transcription.words;
        run_postprocessing(&mut words, &processors);

        let hypothesis = text_formatter
            .format(&words, &format_config)
            .with_context(|| format!("Failed to format output for '{}'", audio_path.display()))?;

        let elapsed = start.elapsed();

        let reference = std::fs::read_to_string(gt_path)
            .with_context(|| format!("Failed to read ground truth '{}'", gt_path.display()))?;

        let wer = calculate_wer(&reference, &hypothesis);

        let ref_word_count = reference.split_whitespace().count();

        println!(
            "{:<30} {:>5}   {:>5.1}%   {:>5.1}s",
            clip_name,
            ref_word_count,
            wer * 100.0,
            elapsed.as_secs_f64()
        );

        results.push(BenchResult {
            clip_name,
            reference_word_count: ref_word_count,
            wer,
            duration: elapsed,
        });
    }

    let summary = summarize(results);

    println!("---");
    println!(
        "Mean WER: {:.1}%   Median WER: {:.1}%   Worst: {:.1}%   Mean time: {:.1}s",
        summary.mean_wer * 100.0,
        summary.median_wer * 100.0,
        summary.worst_wer * 100.0,
        summary.mean_duration.as_secs_f64()
    );

    if summary.mean_wer > 0.05 {
        eprintln!(
            "\nFAIL: Mean WER {:.1}% exceeds 5% threshold",
            summary.mean_wer * 100.0
        );
        process::exit(2);
    }

    Ok(())
}
