# caption

Generate subtitles from any video file. Runs entirely on your computer. No accounts, no uploads, no internet required after setup.

**Source:** [github.com/DukeSaputra/caption](https://github.com/DukeSaputra/caption) | **Author:** [Duke Saputra](https://saputra.co.uk) | **License:** [GPL-3.0](LICENSE)

## What It Does

```
caption video.mp4
```

Produces a subtitle file with precise word-level timing. Import into any video editor (Premiere Pro, DaVinci Resolve, CapCut, etc.) or upload directly to YouTube, Instagram, or TikTok.

**Supported formats:** MP4, MKV, MP3, FLAC, WAV, Vorbis (built-in), plus Opus, WebM, and more with FFmpeg.

## Setup

### Step 1: Download

[**Download the latest release**](https://github.com/DukeSaputra/caption/releases) for your platform and save it to your Downloads folder.

| Platform | File |
|----------|------|
| macOS (Apple Silicon & Intel) | `caption-macos-universal` |
| Windows | `caption-windows-x86_64.exe` |
| Linux | `caption-linux-x86_64` |

---

### Step 2: Install

No admin privileges required.

<details>
<summary><strong>macOS</strong></summary>

**Make sure the downloaded file is in your Downloads folder** (`~/Downloads/caption-macos-universal`). Safari may rename it (e.g. add `.dms`). If so, rename it back to `caption-macos-universal` before continuing.

**1.** Open **Terminal** (press `Cmd + Space`, type `Terminal`, hit Enter)

**2.** Copy and paste this entire block into Terminal, then press Enter:

```
if [ ! -f ~/Downloads/caption-macos-universal ]; then echo "caption-macos-universal not found. Move it to your Downloads folder and try again."; else mkdir -p ~/.local/bin && mv ~/Downloads/caption-macos-universal ~/.local/bin/caption && chmod +x ~/.local/bin/caption && xattr -d com.apple.quarantine ~/.local/bin/caption 2>/dev/null; (grep -q '.local/bin' ~/.zshrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc) && source ~/.zshrc && caption setup; fi
```

This installs the binary and downloads all required models (~1.1 GB). It may take 5-15 minutes depending on your connection. If it gets interrupted, just run `caption setup` to resume.

> **Troubleshooting:**
>
> `Killed: 9` or a security dialog means the quarantine flag wasn't fully removed. Run: `xattr -d com.apple.quarantine ~/.local/bin/caption`
>
> `command not found` after running `caption --help` means the PATH update didn't take effect. Close Terminal and open a new window, then try again.
>
> **macOS Mojave or older (bash):** replace `~/.zshrc` with `~/.bash_profile` in the command above. Not sure? Run `echo $SHELL`.

<details>
<summary>What does the install command do?</summary>

1. `mkdir -p ~/.local/bin` creates the install directory
2. `mv ~/Downloads/caption-macos-universal ~/.local/bin/caption` moves and renames the binary
3. `chmod +x` makes it executable
4. `xattr -d com.apple.quarantine` removes the macOS download quarantine
5. The `grep || echo` line adds `~/.local/bin` to your PATH (only if not already there)
6. `source ~/.zshrc` reloads the config so `caption` is available immediately

</details>

</details>

<details>
<summary><strong>Linux</strong></summary>

**Make sure the downloaded file is in your Downloads folder** (`~/Downloads/caption-linux-x86_64`).

**1.** Open a terminal

**2.** Copy and paste this entire block, then press Enter:

```
if [ ! -f ~/Downloads/caption-linux-x86_64 ]; then echo "caption-linux-x86_64 not found. Move it to your Downloads folder and try again."; else mkdir -p ~/.local/bin && mv ~/Downloads/caption-linux-x86_64 ~/.local/bin/caption && chmod +x ~/.local/bin/caption && (grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc) && source ~/.bashrc && caption setup; fi
```

This installs the binary and downloads all required models (~1.1 GB). If it gets interrupted, run `caption setup` to resume.

> `command not found` after running `caption --help` means the PATH update didn't take effect. Close your terminal and open a new one, then try again.

</details>

<details>
<summary><strong>Windows</strong></summary>

> **Note:** Windows Defender SmartScreen may show a warning when you first run the binary ("Windows protected your PC"). This is normal for unsigned open-source software. Click **More info**, then **Run anyway**. You can verify the source code at [github.com/DukeSaputra/caption](https://github.com/DukeSaputra/caption).

**Make sure the downloaded file is in your Downloads folder** (`caption-windows-x86_64.exe`).

**1.** Open **PowerShell** and paste this entire block, then press Enter:

```
if (!(Test-Path "$HOME\Downloads\caption-windows-x86_64.exe")) { Write-Host "caption-windows-x86_64.exe not found. Move it to your Downloads folder and try again." } else { New-Item -ItemType Directory -Force -Path "$HOME\caption" | Out-Null; Move-Item "$HOME\Downloads\caption-windows-x86_64.exe" "$HOME\caption\caption.exe" -Force; [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$HOME\caption", "User"); Write-Host "Installed. Close and reopen PowerShell to continue setup." }
```

Close and reopen PowerShell, then run:

```
caption setup
```

> `caption is not recognized` means the PATH update hasn't taken effect. Make sure you closed and reopened PowerShell after the install command.

</details>

---

### Step 3: Try It

Type `caption ` (with a space after it), then paste the path to any video or audio file. Press Enter.

**Generate subtitles:**
```
caption video.mp4
```

This creates `video.srt` in the same folder as the video.

**Generate subtitles and burn them into the video:**
```
caption video.mp4 --burn
```

This creates `video.srt` and `video-captioned.mp4` with the subtitles baked into the video.

> **Tip:** If the burned subtitles have errors, you can fix them without re-transcribing. First generate the subtitle file on its own, edit it in any text editor to fix mistakes, then burn the corrected version:
> ```
> caption path/to/video.mp4 --srt path/to/video.srt
> ```

> **Tip:** You don't need to type file paths manually.
> - **macOS:** Drag a file from Finder into Terminal to paste its path. Or right-click while holding **Option** and select **Copy as Pathname**.
> - **Windows:** Hold Shift, right-click the file in Explorer, and select **Copy as path**.
> - **Linux:** Most file managers let you right-click and copy the full path. Or drag the file into the terminal.

---

## Usage

<details open>
<summary><strong>Common commands</strong></summary>

```
caption video.mp4                        # SRT subtitles (default)
caption video.mp4 --format vtt           # WebVTT
caption video.mp4 --format txt           # Plain text transcript
caption video.mp4 --burn                 # Burn subtitles into video
caption video.mp4 -o my-subtitles.srt    # Custom output filename
```

</details>

<details>
<summary><strong>Burn subtitles into video</strong></summary>

Renders each word in a frosted glass capsule, one at a time. Designed for vertical/short-form video (Reels, Shorts, TikToks).

```
caption video.mp4 --burn
```

Produces `video.srt` + `video-captioned.mp4`.

</details>

<details>
<summary><strong>All options</strong></summary>

```
--format <srt|vtt|ass|txt>              Output format (default: srt)
--burn                                  Burn subtitles into video
--srt <PATH>                            Burn an existing SRT (skips transcription)
--initial-prompt <TEXT>                 Vocabulary hint for Whisper
--fillers <keep|remove-confident|remove-all>
                                        Filler word handling (default: remove-confident)
--profanity <keep|mask|replace>         Profanity filtering (default: keep)
--hallucination-filter <off|moderate|aggressive>
                                        False phrase detection (default: moderate)
--no-vad                                Disable speech detection
--no-align                              Disable word-level alignment
-o, --output <PATH>                     Custom output path
-v, --verbose                           Show detailed processing info
```

</details>

---

## Troubleshooting

<details>
<summary><strong>"command not found"</strong></summary>

Your system doesn't know where the binary is. Either `cd` into the folder containing it, or follow the PATH setup in Step 2. You can always run it directly: `~/.local/bin/caption video.mp4`

</details>

<details>
<summary><strong>"No audio track found"</strong></summary>

The input file has no audio. Verify it plays sound in a media player before trying again.

</details>

<details>
<summary><strong>"Could not find a Whisper model file"</strong></summary>

The models aren't where the binary expects them. Re-run `caption setup` to download them.

</details>

<details>
<summary><strong>Subtitles are inaccurate</strong></summary>

Add vocabulary hints for unusual words, names, or brands:

```
caption video.mp4 --initial-prompt "TikTok, iPhone, RSVP"
```

</details>

---

## What Gets Downloaded

`caption setup` downloads everything the tool needs to run locally. Make sure you have at least **1.5 GB of free disk space**.

| Component | Size | Source | Purpose |
|-----------|------|--------|---------|
| Whisper large-v3-turbo Q8_0 | 874 MB | [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp) | Speech recognition via whisper.cpp |
| Silero VAD v5 | 2 MB | [GitHub](https://github.com/snakers4/silero-vad) | Voice activity detection, reduces hallucinations |
| wav2vec2-base-960h | 91 MB | [HuggingFace](https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX) | CTC forced alignment (~20ms word timestamps) |
| FFmpeg | ~80 MB | [GitHub](https://github.com/eugeneware/ffmpeg-static) | Audio extraction from video files |
| ONNX Runtime | ~50 MB | [Microsoft GitHub](https://github.com/microsoft/onnxruntime) | Inference backend for alignment and VAD |
| Inter Bold | <1 MB | [Google Fonts](https://fonts.google.com/specimen/Inter) | Font for burned-in subtitles (`--burn`) |

Models are saved to `~/Library/Application Support/caption/models` on macOS, `%APPDATA%\caption\models` on Windows, and `~/.local/share/caption/models` on Linux. FFmpeg, ONNX Runtime, and the font are placed next to the `caption` binary.

SHA256 hashes are verified automatically. Use `--skip-hash` only if verification fails due to an upstream model update.

---

## Build from Source

<details>
<summary>Instructions</summary>

Requires [Rust](https://rustup.rs).

```
git clone https://github.com/DukeSaputra/caption.git && cd caption && cargo build --release
```

Binary at `target/release/caption-cli`. For Metal GPU acceleration on Apple Silicon (3-4x faster):

```
cargo build --release --features metal
```

</details>

## Uninstall

```
caption uninstall
```

Removes all downloaded models, dependencies (FFmpeg, ONNX Runtime, font), and tells you how to remove the binary itself.

## Security

- Models downloaded from official sources (HuggingFace, GitHub) with SHA256 verification
- Fully [open source](https://github.com/DukeSaputra/caption) and buildable from source
- No root/admin privileges required
- No network access after initial setup
- GPL-3.0 licensed
