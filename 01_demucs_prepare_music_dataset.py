
"""
01_demucs_prepare_music_dataset.py

Purpose
-------
Take every supported audio file in songs/ and use pretrained Demucs to create:

    prepared/
        vocals/
            song1.wav
            song2.wav
            ...
        music/
            song1.wav
            song2.wav
            ...

The "music" directory is the dataset used by our next unsupervised
instrument-learning experiment.

No source1/source2 folders are required.

IMPORTANT
---------
Demucs' --two-stems=vocals gives:
    vocals
    no_vocals

The no_vocals output is what we call "music" here.

Hardware strategy
-----------------
AUTO:
    CUDA available -> one persistent Demucs model on GPU.
    CUDA unavailable -> CPU with multiple worker threads.

Why not launch many GPU Demucs processes?
-----------------------------------------
Each process/model needs GPU memory. Running several independent Demucs
models on one GPU usually duplicates model memory and can cause OOM.
Demucs itself uses parallel jobs for CPU processing; its current apply_model
implementation only creates that thread pool when the device is CPU.

For a single GPU, the correct strategy is:
    CPU decode -> one GPU inference stream -> disk

For CPU:
    multiple Demucs jobs -> multiple CPU threads

Supported input formats
-----------------------
.wav .flac .ogg .mp3 .m4a .aac .aif .aiff .opus .wma .webm .mp4 .mka

Run
---
python 01_demucs_prepare_music_dataset.py

Before running:
    pip install -U demucs

FFmpeg must be installed and available on PATH.

Outputs
-------
prepared/
    vocals/
    music/
    logs/
        separation_log.csv
    metadata.json
"""


import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("songs")

OUTPUT_DIR = Path("prepared")
VOCALS_DIR = OUTPUT_DIR / "vocals"
MUSIC_DIR = OUTPUT_DIR / "music"
LOG_DIR = OUTPUT_DIR / "logs"

# Demucs model.
#
# htdemucs:
#   standard 4-stem Hybrid Transformer Demucs.
#
# htdemucs_ft:
#   fine-tuned version; slower, potentially better.
#
# Start with htdemucs for the dataset creation experiment.
MODEL_NAME = "htdemucs"

# "auto" = GPU if CUDA is visible, otherwise CPU.
DEVICE_MODE = "auto"

# GPU memory safety.
#
# If GPU memory is sufficient, 7-8 seconds is reasonable for HTDemucs.
# Smaller values reduce VRAM use.
GPU_SEGMENT_SECONDS = 7

# Demucs shift trick.
#
# 1 = normal single prediction.
# 2/3 = more computation, potentially more stable separation.
#
# Start with 1. We can run a higher-quality pass later.
SHIFTS = 1

# Default overlap.
OVERLAP = 0.25

# CPU parallelism.
# None = automatically leave 2 logical CPUs for Windows/other work.
CPU_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# If True, don't process a song if both outputs already exist.
SKIP_EXISTING = True

# Save float32 WAV rather than int16.
# This is useful because these files become ML training data.
SAVE_FLOAT32 = True

# Only process these extensions.
AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac",
    ".aif", ".aiff", ".opus", ".wma", ".webm", ".mp4", ".mka",
}


# ============================================================
# HARDWARE
# ============================================================

def detect_device():

    cuda = torch.cuda.is_available()

    print("\n" + "=" * 75)
    print("HARDWARE DETECTION")
    print("=" * 75)

    print(f"PyTorch version       : {torch.__version__}")
    print(f"PyTorch CUDA version  : {torch.version.cuda}")
    print(f"CUDA available        : {cuda}")
    print(f"CPU logical cores     : {os.cpu_count()}")

    if not torch.backends.cuda.is_built():
        print(
            "CUDA support           : not included in this PyTorch build "
            "(CPU-only wheel)"
        )

    if cuda:
        print(f"GPU count             : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)

            print(
                f"GPU {i}                : {props.name}"
            )

            print(
                f"GPU {i} VRAM           : "
                f"{props.total_memory / 1024**3:.2f} GB"
            )

    if DEVICE_MODE == "cpu":
        return "cpu"

    if DEVICE_MODE == "cuda":
        if not cuda:
            raise RuntimeError(
                "DEVICE_MODE='cuda' but PyTorch cannot use CUDA. "
                "Install a CUDA-enabled PyTorch build, then rerun. "
                "The current build is CPU-only."
            )
        return "cuda"

    # AUTO
    if not cuda:
        print(
            "GPU fallback            : using CPU because CUDA is unavailable."
        )
    return "cuda" if cuda else "cpu"


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_audio_files():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{INPUT_DIR.resolve()}"
        )

    files = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    return files


# ============================================================
# OUTPUT PATHS
# ============================================================

def output_paths(input_path):

    # Flatten the filename into the output directories.
    # If two different folders contain the same filename, include
    # a relative-path-safe identifier.
    stem = input_path.stem

    vocals_path = VOCALS_DIR / f"{stem}.wav"
    music_path = MUSIC_DIR / f"{stem}.wav"

    return vocals_path, music_path


# ============================================================
# DEMUCS SEPARATOR
# ============================================================

def create_separator(device):

    try:
        import demucs.api
        from demucs.api import save_audio
    except ImportError:
        raise RuntimeError(
            "\nDemucs is not installed.\n"
            "Run:\n"
            "    pip install -U demucs\n"
        )

    print("\n" + "=" * 75)
    print("LOADING DEMUCS")
    print("=" * 75)

    print(f"Model                 : {MODEL_NAME}")
    print(f"Device                : {device}")

    if device == "cuda":
        print(
            f"GPU                   : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"GPU VRAM              : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

        print(
            f"Segment               : "
            f"{GPU_SEGMENT_SECONDS} sec"
        )

        print(
            "GPU parallelism       : single model / single GPU"
        )

    else:
        print(
            f"CPU workers           : {CPU_WORKERS}"
        )

        print(
            "CPU parallelism       : Demucs internal thread pool"
        )

    separator = demucs.api.Separator(
        model=MODEL_NAME,
        device=device,
        shifts=SHIFTS,
        overlap=OVERLAP,
        split=True,
        segment=(
            GPU_SEGMENT_SECONDS
            if device == "cuda"
            else None
        ),
        jobs=(
            0
            if device == "cuda"
            else CPU_WORKERS
        ),
        progress=False,
    )

    print(
        f"Demucs sample rate    : {separator.samplerate}"
    )

    print(
        f"Demucs channels       : {separator.audio_channels}"
    )

    print(
        f"Demucs sources        : "
        f"{', '.join(separator.model.sources)}"
    )

    return separator, save_audio


# ============================================================
# SINGLE GPU PROCESSING
# ============================================================

def process_gpu(
    separator,
    save_audio,
    files,
):

    print("\n" + "=" * 75)
    print("GPU SEPARATION")
    print("=" * 75)

    total = len(files)

    for index, path in enumerate(files, start=1):

        vocals_path, music_path = output_paths(path)

        if (
            SKIP_EXISTING
            and vocals_path.exists()
            and music_path.exists()
        ):
            print(
                f"[{index}/{total}] SKIP {path.name}"
            )
            continue

        start_time = time.time()

        print(
            f"\n[{index}/{total}] {path.name}"
        )

        try:

            origin, separated = (
                separator.separate_audio_file(path)
            )

            if "vocals" not in separated:
                raise RuntimeError(
                    "Demucs did not return a 'vocals' stem."
                )

            # "other" + "bass" + "drums" is equivalent to no_vocals
            # for the 4-stem model.
            #
            # But for exact reconstruction, use the sum of all
            # non-vocal stems rather than relying on a CLI-generated
            # no_vocals file.
            vocals = separated["vocals"]

            non_vocal_names = [
                name
                for name in separated.keys()
                if name != "vocals"
            ]

            music = sum(
                separated[name]
                for name in non_vocal_names
            )

            vocals_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            music_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            save_audio(
                vocals,
                str(vocals_path),
                samplerate=separator.samplerate,
                as_float=SAVE_FLOAT32,
                clip="rescale",
            )

            save_audio(
                music,
                str(music_path),
                samplerate=separator.samplerate,
                as_float=SAVE_FLOAT32,
                clip="rescale",
            )

            elapsed = time.time() - start_time

            print(
                f"    DONE | {elapsed:.1f} sec"
            )

            if torch.cuda.is_available():
                allocated = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )

                print(
                    f"    VRAM | "
                    f"{allocated:.2f} GB allocated / "
                    f"{reserved:.2f} GB reserved"
                )

        except Exception as e:

            print(
                f"    FAILED: {e}"
            )

            log_failure(
                path,
                e
            )

            traceback.print_exc()

        finally:

            if torch.cuda.is_available():

                # Don't constantly call empty_cache during normal
                # inference; it can slow things down. Synchronize
                # so timing/VRAM reporting is meaningful.
                torch.cuda.synchronize()


# ============================================================
# CPU PROCESSING
# ============================================================

def process_cpu(
    separator,
    save_audio,
    files,
):
    """
    CPU mode.

    IMPORTANT:
    We keep ONE Demucs model in RAM and let Demucs parallelize the
    chunk inference using its internal thread pool.

    This is much safer than creating one full Demucs model per CPU
    worker, which would multiply RAM usage.
    """

    print("\n" + "=" * 75)
    print("CPU PARALLEL SEPARATION")
    print("=" * 75)

    print(
        f"Demucs CPU workers: {CPU_WORKERS}"
    )

    results = []

    for index, path in enumerate(files, start=1):

        vocals_path, music_path = output_paths(path)

        if (
            SKIP_EXISTING
            and vocals_path.exists()
            and music_path.exists()
        ):
            print(
                f"[{index}/{len(files)}] "
                f"SKIP {path.name}"
            )

            results.append({
                "file": str(path),
                "status": "skipped",
                "elapsed": 0.0,
            })

            continue

        start = time.time()

        print(
            f"\n[{index}/{len(files)}] "
            f"{path.name}"
        )

        try:

            origin, separated = (
                separator.separate_audio_file(path)
            )

            if "vocals" not in separated:
                raise RuntimeError(
                    "Demucs did not return a vocals stem."
                )

            vocals = separated["vocals"]

            non_vocal_names = [
                name
                for name in separated.keys()
                if name != "vocals"
            ]

            music = sum(
                separated[name]
                for name in non_vocal_names
            )

            vocals_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            music_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            save_audio(
                vocals,
                str(vocals_path),
                samplerate=separator.samplerate,
                as_float=SAVE_FLOAT32,
                clip="rescale",
            )

            save_audio(
                music,
                str(music_path),
                samplerate=separator.samplerate,
                as_float=SAVE_FLOAT32,
                clip="rescale",
            )

            elapsed = time.time() - start

            result = {
                "file": str(path),
                "status": "done",
                "elapsed": elapsed,
            }

            results.append(result)

            print(
                f"    DONE | {elapsed:.1f} sec"
            )

        except Exception as e:

            print(
                f"    FAILED: {e}"
            )

            log_failure(
                path,
                e
            )

            results.append({
                "file": str(path),
                "status": "failed",
                "elapsed": time.time() - start,
            })

            traceback.print_exc()

    return results


# ============================================================
# FAILURE LOG
# ============================================================

FAILURE_LOG = []


def log_failure(path, error):

    FAILURE_LOG.append(
        {
            "file": str(path),
            "error": str(error),
        }
    )


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    files,
    device,
    elapsed,
):

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "input_directory": str(
            INPUT_DIR.resolve()
        ),
        "output_directory": str(
            OUTPUT_DIR.resolve()
        ),
        "model": MODEL_NAME,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu_vram_gb": (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
            if torch.cuda.is_available()
            else None
        ),
        "sample_rate": 44100,
        "shifts": SHIFTS,
        "overlap": OVERLAP,
        "gpu_segment_seconds": GPU_SEGMENT_SECONDS,
        "cpu_workers": CPU_WORKERS,
        "parallel_strategy": (
            "single GPU model; CPU decoding around GPU"
            if device == "cuda"
            else "single Demucs model with internal CPU thread pool"
        ),
        "input_file_count": len(files),
        "elapsed_seconds": elapsed,
        "failure_count": len(FAILURE_LOG),
    }

    with open(
        LOG_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    with open(
        LOG_DIR / "failures.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["file", "error"]
        )

        writer.writeheader()
        writer.writerows(FAILURE_LOG)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("DEMUCS VOCALS / MUSIC DATASET BUILDER")
    print("=" * 75)

    device = detect_device()

    files = find_audio_files()

    if not files:
        print(
            f"\nNo supported audio files found in:\n"
            f"{INPUT_DIR.resolve()}"
        )
        return

    print(
        f"\nFound {len(files)} audio files."
    )

    print(
        "\nOutput:"
        f"\n  vocals -> {VOCALS_DIR.resolve()}"
        f"\n  music  -> {MUSIC_DIR.resolve()}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    start = time.time()

    if device == "cuda":

        separator, save_audio = create_separator(
            device
        )

        process_gpu(
            separator,
            save_audio,
            files
        )

    else:

        # One model + Demucs internal CPU thread pool.
        # This is the RAM-efficient CPU-parallel path.
        separator, save_audio = create_separator(
            device
        )

        process_cpu(
            separator,
            save_audio,
            files
        )

    elapsed = time.time() - start

    save_metadata(
        files,
        device,
        elapsed
    )

    successful = sum(
        1
        for path in files
        if (
            VOCALS_DIR / f"{path.stem}.wav"
        ).exists()
        and (
            MUSIC_DIR / f"{path.stem}.wav"
        ).exists()
    )

    print("\n" + "=" * 75)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 75)

    print(
        f"Input songs             : {len(files)}"
    )

    print(
        f"Successfully separated  : {successful}"
    )

    print(
        f"Failed                   : {len(files) - successful}"
    )

    print(
        f"Total time               : {elapsed / 60:.2f} min"
    )

    print(
        "\nTraining dataset:"
        f"\n  {MUSIC_DIR.resolve()}"
    )

    print(
        "\nClean vocal reference:"
        f"\n  {VOCALS_DIR.resolve()}"
    )

    print(
        "\nLogs:"
        f"\n  {LOG_DIR.resolve()}"
    )


if __name__ == "__main__":

    # Required for Windows multiprocessing/threading safety.
    main()
