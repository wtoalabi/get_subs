# get_subs

`get_subs` recursively finds videos and writes adjacent SRT subtitles with the
official `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B` models.

```bash
get_subs "/path/to/videos"
```

The scan includes every nested directory, including directories literally named
`subs`. A video at `course/subs/1.mkv` produces `course/subs/1.srt`.

## First run

The launcher creates an isolated `.venv`, installs the pinned official runtime,
then checks both model snapshots. Missing models are downloaded into
`~/.cache/get_subs/huggingface` before video discovery begins. Dependency and
model downloads happen once and are reused by later invocations.

This checkout is exposed globally through:

```text
/opt/homebrew/bin/get_subs -> /Users/mac/dev/python/get_subs/get_subs
```

`ffmpeg` and `ffprobe` must already be available. They are present on the target
Mac; another Homebrew-based Mac can install them with `brew install ffmpeg`.

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
MTS, M2TS, TS, WMV, FLV, OGV, VOB, ASF, DivX, RM/RMVB, 3GP, and 3G2. FFmpeg
handles the actual container and audio decoding.

## Useful controls

```bash
# Force a language instead of auto-detection.
get_subs "/path/to/videos" --language English

# Bias names or specialist vocabulary without forcing a language.
get_subs "/path/to/videos" --context "StoreX, OSUK, Revolut"

# Regenerate SRT files that are already complete.
get_subs "/path/to/videos" --overwrite

# Choose a device explicitly (auto prefers CUDA, then Apple MPS, then CPU).
get_subs "/path/to/videos" --device mps

# See every tuning option.
get_subs --help
```

The default 90-second chunks are a deliberate balance for the target M2 with
16 GB unified memory: FP16 MPS inference keeps both required models resident,
silence-aware boundaries protect recognition quality, and each chunk publishes
useful SRT output promptly. No vLLM dependency is installed because vLLM is a
CUDA-oriented path and does not accelerate this Apple Silicon host.

The forced aligner officially supports Chinese, English, Cantonese, French,
German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. ASR covers
more languages, but timestamp quality outside the aligner's supported set is not
guaranteed by the model vendor.
