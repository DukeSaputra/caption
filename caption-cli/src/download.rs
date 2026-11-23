use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use sha2::{Digest, Sha256};

pub struct ModelInfo {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub filename: &'static str,
    pub needs_extraction: bool,
}

const SUPPORTED_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "whisper-large-v3-turbo",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q8_0.bin",
        sha256: "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        filename: "ggml-large-v3-turbo-q8_0.bin",
        needs_extraction: false,
    },
    ModelInfo {
        name: "silero-vad",
        url: "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
        sha256: "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
        filename: "silero_vad.onnx",
        needs_extraction: false,
    },
    ModelInfo {
        name: "wav2vec2-aligner",
        url: "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/729c1a6730fb549c20a1c73a3d3f96f11020225e/onnx/model_quantized.onnx",
        sha256: "1d9a366c27b2966625cd5035ac3db8847c53bba617169912bb251b42975a3a22",
        filename: "wav2vec2-base-960h.onnx",
        needs_extraction: false,
    },
    ModelInfo {
        name: "parakeet-tdt",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        sha256: "TODO_VERIFY_parakeet-tdt",
        filename: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        needs_extraction: true,
    },
];

pub fn find_model_info(name: &str) -> Option<&'static ModelInfo> {
    SUPPORTED_MODELS.iter().find(|m| m.name == name)
}

pub fn supported_model_names() -> Vec<&'static str> {
    SUPPORTED_MODELS.iter().map(|m| m.name).collect()
}

fn models_dir() -> Result<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(_exe_dir) = exe.parent() {
            let real_exe = exe.canonicalize().unwrap_or(exe);
            let mut dir = real_exe.parent();
            while let Some(d) = dir {
                let candidate = d.join("models");
                if candidate.is_dir() {
                    return Ok(candidate);
                }
                dir = d.parent();
            }
        }
    }

    let data_dir = dirs::data_dir().context(
        "Could not determine models directory. \
         Run from the caption workspace or set $HOME and try again.",
    )?;
    Ok(data_dir.join("caption").join("models"))
}

pub fn download_model(name: &str, skip_hash: bool) -> Result<PathBuf> {
    let info = find_model_info(name).with_context(|| {
        format!(
            "Unknown model '{}'. Supported models: {}",
            name,
            supported_model_names().join(", ")
        )
    })?;

    let target_dir = models_dir()?;
    fs::create_dir_all(&target_dir).with_context(|| {
        format!(
            "Failed to create models directory '{}'",
            target_dir.display()
        )
    })?;

    let final_path = target_dir.join(info.filename);
    let temp_path = target_dir.join(format!("{}.download", info.filename));

    if !info.needs_extraction && final_path.is_file() {
        eprintln!("Model already exists at {}", final_path.display());
        eprintln!("Delete it first if you want to re-download.");
        return Ok(final_path);
    }

    if info.needs_extraction {
        let extracted_name = info
            .filename
            .strip_suffix(".tar.bz2")
            .unwrap_or(info.filename);
        let extracted_path = target_dir.join(extracted_name);
        if extracted_path.is_dir() {
            eprintln!("Model already extracted at {}", extracted_path.display());
            eprintln!("Delete it first if you want to re-download.");
            return Ok(extracted_path);
        }
    }

    eprintln!("Downloading {} from {}", info.name, info.url);
    debug!("Target directory: {}", target_dir.display());

    download_with_progress(info.url, &temp_path)?;

    if skip_hash {
        eprintln!("Skipping SHA256 verification (--skip-hash)");
    } else if info.sha256.starts_with("TODO_VERIFY_") {
        eprintln!(
            "Warning: SHA256 hash not yet verified for this model. \
             Use --skip-hash to suppress this warning."
        );
        eprintln!("Computed SHA256: {}", compute_sha256(&temp_path)?);
    } else {
        eprintln!("Verifying SHA256 hash...");
        let computed = compute_sha256(&temp_path)?;
        if computed != info.sha256 {
            let _ = fs::remove_file(&temp_path);
            bail!(
                "SHA256 mismatch for {}!\n  Expected: {}\n  Computed: {}\n\
                 The downloaded file has been deleted. Try again or use --skip-hash.",
                info.name,
                info.sha256,
                computed
            );
        }
        eprintln!("SHA256 verified.");
    }

    fs::rename(&temp_path, &final_path).with_context(|| {
        format!(
            "Failed to rename '{}' to '{}'",
            temp_path.display(),
            final_path.display()
        )
    })?;

    let result_path = if info.needs_extraction {
        let extracted = extract_tar_bz2(&final_path, &target_dir)?;
        let _ = fs::remove_file(&final_path);
        extracted
    } else {
        final_path
    };

    eprintln!("Model saved to {}", result_path.display());
    Ok(result_path)
}

fn download_with_progress(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("caption-cli")
        .build()
        .context("Failed to create HTTP client")?;

    let response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to send request to {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error downloading from {url}"))?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .expect("valid progress bar template")
                .progress_chars("#>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})")
                .expect("valid spinner template"),
        );
        pb
    };

    let mut file = fs::File::create(dest)
        .with_context(|| format!("Failed to create file '{}'", dest.display()))?;

    let mut reader = response;
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .context("Failed to read from download stream")?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])
            .with_context(|| format!("Failed to write to '{}'", dest.display()))?;
        pb.inc(bytes_read as u64);
    }

    pb.finish_with_message("Download complete");
    Ok(())
}

pub fn compute_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("Failed to open '{}' for hashing", path.display()))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file
            .read(&mut buffer)
            .with_context(|| format!("Failed to read '{}' during hashing", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_tar_bz2(archive: &Path, target_dir: &Path) -> Result<PathBuf> {
    eprintln!("Extracting {}...", archive.display());

    let status = Command::new("tar")
        .arg("xjf")
        .arg(archive)
        .arg("-C")
        .arg(target_dir)
        .status()
        .context("Failed to run 'tar' for extraction. Is tar installed?")?;

    if !status.success() {
        bail!(
            "tar extraction failed with exit code {}",
            status.code().unwrap_or(-1)
        );
    }

    let archive_name = archive
        .file_name()
        .and_then(|n| n.to_str())
        .context("Invalid archive filename")?;

    let dir_name = archive_name
        .strip_suffix(".tar.bz2")
        .unwrap_or(archive_name);

    let extracted = target_dir.join(dir_name);
    if !extracted.is_dir() {
        bail!(
            "Expected extracted directory '{}' not found after extraction",
            extracted.display()
        );
    }

    eprintln!("Extracted to {}", extracted.display());
    Ok(extracted)
}

fn exe_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Could not determine binary location")?;
    let exe = exe.canonicalize().unwrap_or(exe);
    exe.parent()
        .map(|p| p.to_path_buf())
        .context("Could not determine binary directory")
}

fn download_file_to(url: &str, dest: &Path) -> Result<()> {
    if dest.is_file() {
        eprintln!("Already exists: {}", dest.display());
        return Ok(());
    }
    let temp = dest.with_extension("download");
    eprintln!("Downloading {}", url);
    download_with_progress(url, &temp)?;
    fs::rename(&temp, dest).with_context(|| format!("Failed to rename to {}", dest.display()))?;
    eprintln!("Saved: {}", dest.display());
    Ok(())
}

pub fn setup_dependencies() -> Result<()> {
    let dir = exe_dir()?;

    let font_path = dir.join("inter-bold.ttf");
    if !font_path.is_file() {
        download_file_to(
            "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuFuYMZg.ttf",
            &font_path,
        )?;
    } else {
        eprintln!("Already exists: {}", font_path.display());
    }

    let ffmpeg_name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let ffmpeg_path = dir.join(ffmpeg_name);
    if !ffmpeg_path.is_file() {
        download_ffmpeg(&dir)?;
    } else {
        eprintln!("Already exists: {}", ffmpeg_path.display());
    }

    let ort_path = dir.join(ORT_LIB);
    if !ort_path.is_file() {
        download_onnxruntime(&dir)?;
    } else {
        eprintln!("Already exists: {}", ort_path.display());
    }

    Ok(())
}

fn download_ffmpeg(dir: &Path) -> Result<()> {
    let base = "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1";
    let (url, binary_name) = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            (format!("{base}/ffmpeg-darwin-arm64.gz"), "ffmpeg")
        } else {
            (format!("{base}/ffmpeg-darwin-x64.gz"), "ffmpeg")
        }
    } else if cfg!(windows) {
        (format!("{base}/ffmpeg-win32-x64.gz"), "ffmpeg.exe")
    } else {
        (format!("{base}/ffmpeg-linux-x64.gz"), "ffmpeg")
    };

    let gz_path = dir.join("ffmpeg.gz");
    download_file_to(&url, &gz_path)?;

    eprintln!("Extracting FFmpeg...");
    let ffmpeg_dest = dir.join(binary_name);

    let gz_file = fs::File::open(&gz_path)
        .with_context(|| format!("Failed to open {}", gz_path.display()))?;
    let mut decoder = flate2::read::GzDecoder::new(gz_file);
    let mut out_file = fs::File::create(&ffmpeg_dest)
        .with_context(|| format!("Failed to create {}", ffmpeg_dest.display()))?;
    std::io::copy(&mut decoder, &mut out_file).context("Failed to decompress FFmpeg")?;

    let _ = fs::remove_file(&gz_path);

    #[cfg(unix)]
    {
        Command::new("chmod")
            .args(["+x"])
            .arg(&ffmpeg_dest)
            .status()
            .ok();
    }

    eprintln!("FFmpeg installed.");
    Ok(())
}

const ORT_LIB: &str = if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else if cfg!(windows) {
    "onnxruntime.dll"
} else {
    "libonnxruntime.so"
};

const ORT_NUGET_URL: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.23.1/microsoft.ml.onnxruntime.1.23.1.nupkg";
const ORT_NUGET_SHA256: &str = "e04551ecddfd42e1922c8cb916f5faf2c229ae5edfe2fe883cc1193c52e16a28";

fn ort_nuget_entry(os: &str, arch: &str) -> Option<String> {
    let (rid_os, lib) = match os {
        "macos" => ("osx", "libonnxruntime.dylib"),
        "windows" => ("win", "onnxruntime.dll"),
        "linux" => ("linux", "libonnxruntime.so"),
        _ => return None,
    };
    let rid_arch = match arch {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        _ => return None,
    };
    Some(format!("runtimes/{rid_os}-{rid_arch}/native/{lib}"))
}

fn download_onnxruntime(dir: &Path) -> Result<()> {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let Some(entry_name) = ort_nuget_entry(os, arch) else {
        bail!("ONNX Runtime has no prebuilt library for {os}-{arch}");
    };

    let package_path = dir.join("onnxruntime.nupkg");
    let result = install_ort_from_package(&package_path, &entry_name, &dir.join(ORT_LIB));
    let _ = fs::remove_file(&package_path);
    result?;

    eprintln!("ONNX Runtime installed.");
    Ok(())
}

fn install_ort_from_package(package_path: &Path, entry_name: &str, lib_dest: &Path) -> Result<()> {
    download_file_to(ORT_NUGET_URL, package_path)?;

    let computed = compute_sha256(package_path)?;
    if computed != ORT_NUGET_SHA256 {
        bail!("ONNX Runtime package checksum mismatch: expected {ORT_NUGET_SHA256}, got {computed}");
    }

    eprintln!("Extracting ONNX Runtime...");
    extract_zip_entry(package_path, entry_name, lib_dest).context("Failed to extract ONNX Runtime")
}

fn extract_zip_entry(archive: &Path, entry_name: &str, dest: &Path) -> Result<()> {
    let file = fs::File::open(archive)
        .with_context(|| format!("Failed to open archive '{}'", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive '{}'", archive.display()))?;
    let mut entry = zip
        .by_name(entry_name)
        .with_context(|| format!("'{entry_name}' not found in '{}'", archive.display()))?;

    // setup skips an existing lib, so a truncated one must never land at dest
    let partial = dest.with_extension("partial");
    let written = fs::File::create(&partial)
        .and_then(|mut out| std::io::copy(&mut entry, &mut out))
        .and_then(|_| fs::rename(&partial, dest));
    if let Err(e) = written {
        let _ = fs::remove_file(&partial);
        return Err(e).with_context(|| format!("Failed to write '{}'", dest.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_model_info_valid_whisper() {
        let info = find_model_info("whisper-large-v3-turbo");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "whisper-large-v3-turbo");
        assert_eq!(info.filename, "ggml-large-v3-turbo-q8_0.bin");
        assert!(!info.needs_extraction);
    }

    #[test]
    fn find_model_info_valid_parakeet() {
        let info = find_model_info("parakeet-tdt");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "parakeet-tdt");
        assert!(info.needs_extraction);
        assert!(info.filename.ends_with(".tar.bz2"));
    }

    #[test]
    fn find_model_info_invalid_returns_none() {
        assert!(find_model_info("nonexistent-model").is_none());
        assert!(find_model_info("").is_none());
        assert!(find_model_info("whisper").is_none());
    }

    #[test]
    fn supported_model_names_returns_all() {
        let names = supported_model_names();
        assert_eq!(names.len(), SUPPORTED_MODELS.len());
        assert!(names.contains(&"whisper-large-v3-turbo"));
        assert!(names.contains(&"parakeet-tdt"));
    }

    #[test]
    fn supported_model_names_no_duplicates() {
        let names = supported_model_names();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "Duplicate model names found");
    }

    #[test]
    fn sha256_known_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, b"hello world\n").unwrap();

        let hash = compute_sha256(&path).unwrap();
        assert_eq!(
            hash,
            "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
        );
    }

    #[test]
    fn sha256_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        fs::write(&path, b"").unwrap();

        let hash = compute_sha256(&path).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_nonexistent_file_returns_error() {
        let result = compute_sha256(Path::new("/tmp/does-not-exist-sha256-test"));
        assert!(result.is_err());
    }

    #[test]
    fn all_models_have_valid_urls() {
        for model in SUPPORTED_MODELS {
            assert!(
                model.url.starts_with("https://"),
                "Model '{}' has invalid URL: {}",
                model.name,
                model.url
            );
        }
    }

    #[test]
    fn all_models_have_nonempty_fields() {
        for model in SUPPORTED_MODELS {
            assert!(!model.name.is_empty(), "Model has empty name");
            assert!(
                !model.url.is_empty(),
                "Model '{}' has empty URL",
                model.name
            );
            assert!(
                !model.sha256.is_empty(),
                "Model '{}' has empty sha256",
                model.name
            );
            assert!(
                !model.filename.is_empty(),
                "Model '{}' has empty filename",
                model.name
            );
        }
    }

    const ORT_PACKAGE_LIBS: &[&str] = &[
        "runtimes/linux-arm64/native/libonnxruntime.so",
        "runtimes/linux-x64/native/libonnxruntime.so",
        "runtimes/osx-arm64/native/libonnxruntime.dylib",
        "runtimes/osx-x64/native/libonnxruntime.dylib",
        "runtimes/win-arm64/native/onnxruntime.dll",
        "runtimes/win-x64/native/onnxruntime.dll",
    ];

    #[test]
    fn ort_entry_exists_in_package_for_every_supported_platform() {
        for os in ["macos", "windows", "linux"] {
            for arch in ["x86_64", "aarch64"] {
                let entry = ort_nuget_entry(os, arch).unwrap();
                assert!(ORT_PACKAGE_LIBS.contains(&entry.as_str()), "{os}-{arch}: {entry}");
            }
        }
    }

    #[test]
    fn ort_entry_unsupported_platform_returns_none() {
        assert_eq!(ort_nuget_entry("freebsd", "x86_64"), None);
        assert_eq!(ort_nuget_entry("linux", "x86"), None);
        assert_eq!(ort_nuget_entry("macos", "riscv64"), None);
    }

    #[test]
    fn ort_entry_for_this_host_ends_with_installed_lib_name() {
        let entry = ort_nuget_entry(std::env::consts::OS, std::env::consts::ARCH).unwrap();
        assert!(entry.ends_with(&format!("/{ORT_LIB}")), "{entry}");
    }

    #[test]
    fn ort_package_url_is_pinned_and_hashed() {
        assert!(ORT_NUGET_URL.starts_with("https://api.nuget.org/"));
        assert!(!ORT_NUGET_URL.contains("github.com"));
        assert_eq!(ORT_NUGET_SHA256.len(), 64);
        assert!(ORT_NUGET_SHA256.chars().all(|c| c.is_ascii_hexdigit()));
    }

    fn write_test_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let mut zip = zip::ZipWriter::new(fs::File::create(path).unwrap());
        for (name, data) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap();
    }

    #[test]
    fn extract_zip_entry_writes_only_the_named_entry() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("p.nupkg");
        write_test_zip(
            &archive,
            &[
                ("runtimes/osx-arm64/native/libonnxruntime.dylib", b"arm"),
                ("runtimes/osx-x64/native/libonnxruntime.dylib", b"x64"),
            ],
        );
        let dest = dir.path().join("libonnxruntime.dylib");
        extract_zip_entry(&archive, "runtimes/osx-x64/native/libonnxruntime.dylib", &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"x64");
        assert!(!dest.with_extension("partial").exists());
        assert!(!dir.path().join("runtimes").exists());
    }

    #[test]
    fn extract_zip_entry_missing_entry_errors_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("p.nupkg");
        write_test_zip(&archive, &[("other.txt", b"x")]);
        let dest = dir.path().join("libonnxruntime.so");
        let err = extract_zip_entry(&archive, "runtimes/linux-x64/native/libonnxruntime.so", &dest)
            .unwrap_err();
        assert!(format!("{err:#}").contains("not found"));
        assert!(!dest.exists());
        assert!(!dest.with_extension("partial").exists());
    }

    #[test]
    fn extract_zip_entry_not_a_zip_errors() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("p.nupkg");
        fs::write(&archive, b"not a zip").unwrap();
        let dest = dir.path().join("onnxruntime.dll");
        assert!(extract_zip_entry(&archive, "x", &dest).is_err());
        assert!(!dest.exists());
    }

    #[test]
    #[ignore]
    fn real_package_matches_hash_and_holds_every_lib() {
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("onnxruntime.nupkg");        let entry = ort_nuget_entry(std::env::consts::OS, std::env::consts::ARCH).unwrap();
        install_ort_from_package(&package, &entry, &dir.path().join(ORT_LIB)).unwrap();
        assert!(fs::metadata(dir.path().join(ORT_LIB)).unwrap().len() > 1_000_000);
        let mut zip = zip::ZipArchive::new(fs::File::open(&package).unwrap()).unwrap();
        for lib in ORT_PACKAGE_LIBS {
            assert!(zip.by_name(lib).is_ok(), "{lib}");
        }
    }
}
