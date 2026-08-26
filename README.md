# get_subs

`get_subs` recursively finds videos and writes adjacent SRT subtitles with the
official `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B` models.

```bash
get_subs "/path/to/videos"
```

The scan includes every nested directory, including directories literally named
`subs`. A video at `course/subs/1.mkv` produces `course/subs/1.srt`.

## Installation

Requirements:

- Python 3.10 through 3.13
- `ffmpeg` and `ffprobe`
- Enough disk space and memory for both Qwen models

Clone the repository, then expose its launcher from a directory on your `PATH`.
For example:

```bash
mkdir -p "$HOME/.local/bin"
ln -s "$(pwd)/get_subs" "$HOME/.local/bin/get_subs"
```

If necessary, add `$HOME/.local/bin` to your shell's `PATH`. FFmpeg is available
from common package managers such as Homebrew (`brew install ffmpeg`) and APT
(`sudo apt install ffmpeg`).

## First run

The launcher creates an isolated `.venv`, installs the pinned official runtime,
then checks both model snapshots. Missing models are downloaded into
`~/.cache/get_subs/huggingface` before video discovery begins. Dependency and
model downloads happen once and are reused by later invocations.

## What happens during a run

The terminal reports six explicit passes: preflight, model availability,
recursive discovery, media probing, runtime loading, and transcription. During
transcription it reports chunk count, elapsed inference time, measured realtime
speed, video percentage, cues written, and a continuously recalibrated run ETA.

Long videos are split near detected silence. Each chunk receives a small overlap
for word-boundary context, then midpoint ownership removes overlap duplicates.
The forced aligner's word/character timestamps are grouped into readable cues
while punctuation is restored from the ASR transcript.

After each completed chunk, the program atomically rewrites the real output
(`1.srt`, for example). It does not wait for the entire directory or video. A
hidden checkpoint beside the SRT makes interruption safe; running the same
command again resumes from the next unfinished chunk. Completed SRT files are
skipped unless `--overwrite` is supplied.

## Formats

The recursive extension set includes MP4, MKV, MOV, AVI, WebM, M4V, MPEG, MPG,
MTS, M2TS, M2V, MXF, TS, WMV, FLV/F4V, OGV, VOB, ASF, DivX, RM/RMVB, 3GP, and
3G2. FFmpeg handles the actual container and audio decoding.

## Useful controls

```bash
# Force a language instead of auto-detection.
get_subs "/path/to/videos" --language English

# Bias names or specialist vocabulary without forcing a language.
get_subs "/path/to/videos" --context "Acme Corp, PostgreSQL, Kubernetes"

# Regenerate SRT files that are already complete.
get_subs "/path/to/videos" --overwrite

# Choose a device explicitly (auto prefers CUDA, then Apple MPS, then CPU).
get_subs "/path/to/videos" --device mps

# Cap decoder work for an especially slow/fast-speech workload.
get_subs "/path/to/videos" --max-new-tokens 1024

# See every tuning option.
get_subs --help
```

### What `--context` means

`--context` is an optional recognition hint sent to Qwen as the prompt context
for every audio chunk. Use it for proper nouns, product names, acronyms, or
specialist vocabulary that the speaker is likely to say. It can improve how
the ASR model spells those terms; the forced aligner then timestamps the text
that ASR produced.

For example:

```bash
get_subs "/path/to/videos" \
  --context "Names: Ada Lovelace, Grace Hopper. Terms: Kubernetes, PostgreSQL."
```

The value is not inserted into the subtitles, is not a glossary or guaranteed
spelling replacement, and does not select a language. Keep it short and relevant
to the videos. Use `--language English` separately when the spoken language is
known. The same context is applied to each chunk, including chunks created when
the program resumes from a checkpoint.

The default 90-second chunks balance accelerator memory, recognition quality at
boundaries, and live update frequency. Automatic device selection prefers CUDA,
then Apple MPS, then CPU; GPU or MPS acceleration is strongly recommended. The
portable Transformers backend is used because timestamp generation requires the
forced aligner alongside ASR.

`--max-new-tokens` limits how much text Qwen may decode for one chunk. The
default 1,024 tokens covers ordinary 90-second speech while bounding inference
time; raise it for unusually dense or very fast speech, or lower it when
latency matters more than completeness.

The forced aligner officially supports Chinese, English, Cantonese, French,
German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. ASR covers
more languages, but timestamp quality outside the aligner's supported set is not
guaranteed by the model vendor.
