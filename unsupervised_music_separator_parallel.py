
"""
unsupervised_music_separator.py

Experimental UNSUPERVISED multi-source music separation.

IMPORTANT:
    This version intentionally does NOT use source1/source2 folders or instrument labels.
    It receives only mixed songs.

    The model has a MAXIMUM number of source slots, but it does NOT target that many
    instruments in every song. A learned activity gate can suppress unused slots.

    Example:
        Song A -> 3 active slots
        Song B -> 5 active slots
        Song C -> 2 active slots

    The slot identities are arbitrary:
        slot 0 is NOT always guitar
        slot 1 is NOT always vocals
        etc.

    This is a research prototype. Pure reconstruction loss cannot guarantee that
    the discovered slots correspond to real instruments. The purpose of this first
    experiment is to observe whether useful, repeatable source structure emerges.

Requirements:
    pip install torch torchaudio soundfile matplotlib numpy
    FFmpeg must be installed and available on PATH.

Run:
    python unsupervised_music_separator.py

Folder:
    songs/
        song1.mp3
        song2.m4a
        ...

Outputs:
    results/
        config.json
        model_parameters.txt
        training_log.csv
        loss_curve.png
        checkpoints/
        separated/
            song1/
                slot_00.wav
                slot_01.wav
                ...
"""

import os
import csv
import json
import math
import shutil
import subprocess
import time
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

SONG_DIR = Path("songs")
RESULT_DIR = Path("results")

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac",
    ".aif", ".aiff", ".opus", ".wma", ".webm", ".mp4", ".mka",
}

# This is CAPACITY, NOT the number of instruments to find.
# Unused slots are allowed to become inactive.
MAX_SOURCES = 8

SAMPLE_RATE = 16000

# STFT parameters
N_FFT = 1024
HOP_LENGTH = 256

# Training
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Number of random audio chunks sampled from each song per epoch.
CHUNKS_PER_SONG = 4

# Duration of each training chunk.
CHUNK_SECONDS = 6

# Model size
HIDDEN_CHANNELS = 64
LATENT_CHANNELS = 64

# Loss weights
LAMBDA_RECON = 1.0
LAMBDA_MASK_SUM = 0.1
LAMBDA_DIVERSITY = 0.05
LAMBDA_ACTIVITY = 0.001

# Save a checkpoint every N epochs
CHECKPOINT_EVERY = 5

SEED = 42

# ============================================================
# HARDWARE / PARALLELISM
# ============================================================

# "auto" lets PyTorch choose CUDA when a working CUDA build is detected.
# Set FORCE_CPU = True only for debugging.
FORCE_CPU = False

# CPU workers used by DataLoader. 0 disables multiprocessing.
# "auto" uses most CPU cores while leaving a couple free for the OS.
CPU_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# Persistent workers avoid recreating worker processes every epoch.
PERSISTENT_WORKERS = True

# Larger prefetch keeps the GPU fed while workers prepare audio chunks.
PREFETCH_FACTOR = 4

# Use CUDA memory efficiently when possible.
USE_AMP = True

# Allow cuDNN to select fast convolution algorithms.
ENABLE_CUDNN_BENCHMARK = True

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# CUDA DIAGNOSTICS
# ------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()

if FORCE_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK

# TF32 can substantially accelerate supported NVIDIA GPUs.
if DEVICE.type == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

print("\n" + "=" * 70)
print("HARDWARE DETECTION")
print("=" * 70)
print(f"PyTorch version          : {torch.__version__}")
print(f"CUDA available           : {torch.cuda.is_available()}")
print(f"PyTorch CUDA version     : {torch.version.cuda}")
print(f"cuDNN available          : {torch.backends.cudnn.is_available()}")
print(f"CPU logical cores        : {os.cpu_count()}")
print(f"Selected device          : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU count                : {torch.cuda.device_count()}")
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"GPU {gpu_id} name            : {props.name}")
        print(f"GPU {gpu_id} VRAM            : {props.total_memory / 1024**3:.2f} GB")
        print(f"GPU {gpu_id} compute capability: {props.major}.{props.minor}")
else:
    print("WARNING: CUDA is NOT visible to this PyTorch installation.")
    print("The code will use CPU multiprocessing.")


# ============================================================
# DATASET
# ============================================================

def load_audio(path):
    """Load an audio file through FFmpeg and return mono float32 samples."""

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg and reopen the terminal."
        )

    command = [
        ffmpeg,
        "-nostdin",
        "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "FFmpeg could not decode the file.")

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()

    if audio.size == 0:
        raise RuntimeError("The file contains no readable audio stream.")

    return audio

class MixedSongDataset(Dataset):
    """
    Dataset containing ONLY mixed songs.

    No clean sources.
    No instrument labels.
    No target masks.
    """

    def __init__(self, song_paths):
        self.song_paths = song_paths

        self.chunk_samples = CHUNK_SECONDS * SAMPLE_RATE

        self.audio = {}

        print("\nLoading songs...")
        for path in self.song_paths:
            try:
                x = load_audio(path)

                # Peak normalization for stable training.
                peak = np.max(np.abs(x)) + 1e-8
                x = x / peak

                self.audio[path] = torch.from_numpy(x)

                print(f"  {path.name:40s} {len(x)/SAMPLE_RATE:7.2f} sec")

            except Exception as e:
                print(f"Could not load {path}: {e}")

        self.song_paths = list(self.audio.keys())

    def __len__(self):
        return len(self.song_paths) * CHUNKS_PER_SONG

    def __getitem__(self, idx):
        song_index = idx % len(self.song_paths)

        path = self.song_paths[song_index]
        x = self.audio[path]

        if len(x) <= self.chunk_samples:
            chunk = F.pad(
                x,
                (0, self.chunk_samples - len(x))
            )
        else:
            start = random.randint(0, len(x) - self.chunk_samples)
            chunk = x[start:start + self.chunk_samples]

        return chunk


# ============================================================
# UNSUPERVISED SOURCE SEPARATOR
# ============================================================

class UnsupervisedSeparator(nn.Module):
    """
    Slot-based mask separator.

    Input:
        log magnitude STFT

    Output:
        MAX_SOURCES magnitude masks + activity gates.

    The model does not know what any slot represents.
    """

    def __init__(self, max_sources):
        super().__init__()

        self.max_sources = max_sources

        self.encoder = nn.Sequential(
            nn.Conv2d(1, HIDDEN_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.GELU(),

            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.GELU(),

            nn.Conv2d(HIDDEN_CHANNELS, LATENT_CHANNELS, 3, padding=1),
            nn.GELU(),
        )

        # Produces one mask per source slot.
        self.mask_head = nn.Conv2d(
            LATENT_CHANNELS,
            max_sources,
            kernel_size=1
        )

        # Produces an input-dependent activity score for each slot.
        self.activity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(LATENT_CHANNELS, max_sources)
        )

    def forward(self, x):
        """
        x shape:
            [B, 1, F, T]

        Returns:
            masks:
                [B, K, F, T]

            activity:
                [B, K]
        """

        z = self.encoder(x)

        raw_masks = self.mask_head(z)

        # Softmax makes the masks compete for the mixture.
        masks = torch.softmax(raw_masks, dim=1)

        activity_logits = self.activity_head(z)

        # Sigmoid = each source can independently be active/inactive.
        activity = torch.sigmoid(activity_logits)

        # Activity gate modifies source masks.
        gated_masks = masks * activity[:, :, None, None]

        # Re-normalize only where useful.
        # This preserves the possibility that some slots are inactive.
        denom = gated_masks.sum(dim=1, keepdim=True) + 1e-8

        normalized_masks = gated_masks / denom

        return normalized_masks, activity


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def reconstruction_loss(mixture_mag, masks):
    """
    The separated magnitudes should reconstruct the mixture magnitude.

    sum_i M_i ~= 1
    """

    reconstructed = masks.sum(dim=1, keepdim=True)

    return F.mse_loss(reconstructed, torch.ones_like(reconstructed))


def mask_sum_loss(masks):
    """
    Encourage source masks to explain the whole mixture.
    """

    mask_sum = masks.sum(dim=1)

    return F.mse_loss(
        mask_sum,
        torch.ones_like(mask_sum)
    )


def diversity_loss(masks):
    """
    Penalize different source masks from becoming identical.

    This is NOT a guarantee of physical source separation.
    It simply discourages trivial identical masks.
    """

    b, k, f, t = masks.shape

    flattened = masks.reshape(b, k, -1)

    flattened = F.normalize(flattened, dim=-1)

    similarity = torch.bmm(
        flattened,
        flattened.transpose(1, 2)
    )

    identity = torch.eye(
        k,
        device=masks.device
    )[None]

    off_diagonal = similarity * (1 - identity)

    return (off_diagonal ** 2).mean()


def activity_loss(activity):
    """
    Mild sparsity pressure.

    It encourages the network not to activate every slot strongly
    for every song.

    Keep this small. Too much pressure can kill useful sources.
    """

    return activity.mean()


# ============================================================
# STFT
# ============================================================

WINDOW = torch.hann_window(N_FFT)


def get_log_magnitude(x):
    """
    x:
        [B, samples]

    returns:
        [B, 1, frequency, time]
    """

    window = WINDOW.to(x.device)

    X = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(X)

    log_mag = torch.log1p(magnitude)

    return log_mag.unsqueeze(1)


# ============================================================
# PARAMETER REPORT
# ============================================================

def print_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    print(f"Model class             : {model.__class__.__name__}")
    print(f"Maximum source slots    : {MAX_SOURCES}")
    print(f"Trainable parameters    : {trainable:,}")
    print(f"Total parameters        : {total:,}")
    print(f"Device                  : {DEVICE}")

    print("\nArchitecture:")
    print(model)

    print("\n" + "=" * 70)
    print("HYPERPARAMETERS")
    print("=" * 70)

    params = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "chunk_seconds": CHUNK_SECONDS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_sources": MAX_SOURCES,
        "hidden_channels": HIDDEN_CHANNELS,
        "latent_channels": LATENT_CHANNELS,
        "optimizer": "AdamW",
        "lambda_reconstruction": LAMBDA_RECON,
        "lambda_mask_sum": LAMBDA_MASK_SUM,
        "lambda_diversity": LAMBDA_DIVERSITY,
        "lambda_activity": LAMBDA_ACTIVITY,
    }

    for key, value in params.items():
        print(f"{key:25s}: {value}")

    RESULT_DIR.mkdir(exist_ok=True)

    with open(
        RESULT_DIR / "model_parameters.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("MODEL PARAMETERS\n")
        f.write("=" * 70 + "\n")

        f.write(f"Model class: {model.__class__.__name__}\n")
        f.write(f"Maximum source slots: {MAX_SOURCES}\n")
        f.write(f"Trainable parameters: {trainable:,}\n")
        f.write(f"Total parameters: {total:,}\n")
        f.write(f"Device: {DEVICE}\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write("=" * 70 + "\n")

        for key, value in params.items():
            f.write(f"{key}: {value}\n")

        f.write("\nARCHITECTURE\n")
        f.write("=" * 70 + "\n")
        f.write(str(model))


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_config():
    RESULT_DIR.mkdir(exist_ok=True)

    config = {
        "song_directory": str(SONG_DIR),
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_sources": MAX_SOURCES,
        "chunk_seconds": CHUNK_SECONDS,
        "hidden_channels": HIDDEN_CHANNELS,
        "latent_channels": LATENT_CHANNELS,
        "lambda_reconstruction": LAMBDA_RECON,
        "lambda_mask_sum": LAMBDA_MASK_SUM,
        "lambda_diversity": LAMBDA_DIVERSITY,
        "lambda_activity": LAMBDA_ACTIVITY,
        "device": str(DEVICE),
    }

    with open(
        RESULT_DIR / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)


# ============================================================
# TRAINING
# ============================================================

def train(model, loader):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # CUDA AMP: lower precision for suitable operations, FP32 where needed.
    # This reduces VRAM traffic and can increase throughput substantially.
    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )

    history = []

    total_iterations = len(loader)

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(f"Songs                   : {len(loader.dataset.song_paths)}")
    print(f"Iterations / epoch      : {total_iterations}")
    print(f"Total iterations        : {total_iterations * EPOCHS}")
    print(f"Epochs                  : {EPOCHS}")
    print(f"Batch size              : {BATCH_SIZE}")
    print(f"Learning rate           : {LEARNING_RATE}")
    print(f"Optimizer               : AdamW")
    print(f"Scheduler               : CosineAnnealingLR")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_sum = 0.0
        epoch_div = 0.0
        epoch_activity = 0.0

        start_time = time.time()

        for iteration, mixture in enumerate(loader, start=1):

            mixture = mixture.to(
                DEVICE,
                non_blocking=True
            )

            # AMP context keeps the GPU busy while reducing memory bandwidth.
            if amp_enabled:
                autocast_context = torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                )
            else:
                autocast_context = torch.autocast(
                    device_type="cpu",
                    enabled=False
                )

            with autocast_context:
                log_mag = get_log_magnitude(mixture)

                masks, activity = model(log_mag)

                loss_recon = reconstruction_loss(
                    log_mag,
                    masks
                )

                loss_sum = mask_sum_loss(masks)

                loss_div = diversity_loss(masks)

                loss_act = activity_loss(activity)

                loss = (
                    LAMBDA_RECON * loss_recon
                    + LAMBDA_MASK_SUM * loss_sum
                    + LAMBDA_DIVERSITY * loss_div
                    + LAMBDA_ACTIVITY * loss_act
                )

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

            epoch_total += loss.item()
            epoch_recon += loss_recon.item()
            epoch_sum += loss_sum.item()
            epoch_div += loss_div.item()
            epoch_activity += loss_act.item()

            if iteration == 1 or iteration % 10 == 0:

                if DEVICE.type == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    gpu_mem = f" | VRAM {allocated:.2f}/{reserved:.2f} GB"
                else:
                    gpu_mem = ""

                print(
                    f"Epoch {epoch:03d}/{EPOCHS} | "
                    f"Iter {iteration:04d}/{total_iterations} | "
                    f"Loss {loss.item():.6f} | "
                    f"Recon {loss_recon.item():.6f} | "
                    f"Div {loss_div.item():.6f} | "
                    f"Grad {float(grad_norm):.4f}"
                    f"{gpu_mem}"
                )

        scheduler.step()

        n = total_iterations

        avg_total = epoch_total / n
        avg_recon = epoch_recon / n
        avg_sum = epoch_sum / n
        avg_div = epoch_div / n
        avg_activity = epoch_activity / n

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "loss": avg_total,
            "reconstruction_loss": avg_recon,
            "mask_sum_loss": avg_sum,
            "diversity_loss": avg_div,
            "activity_loss": avg_activity,
            "learning_rate": current_lr,
            "epoch_time_sec": elapsed,
        })

        print(
            f"\n>>> EPOCH {epoch:03d} COMPLETE | "
            f"Loss={avg_total:.6f} | "
            f"LR={current_lr:.8f} | "
            f"Time={elapsed:.1f}s\n"
        )

        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, epoch)

    save_checkpoint(
        model,
        optimizer,
        EPOCHS,
        final=True
    )

    return history


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    final=False
):

    checkpoint_dir = RESULT_DIR / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    name = (
        "final_model.pt"
        if final
        else f"checkpoint_epoch_{epoch:03d}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_dir / name
    )

    print(f"Checkpoint saved: {checkpoint_dir / name}")


# ============================================================
# LOSS CURVE
# ============================================================

def plot_loss(history):

    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [h["loss"] for h in history],
        label="Total loss"
    )

    plt.plot(
        epochs,
        [h["reconstruction_loss"] for h in history],
        label="Reconstruction"
    )

    plt.plot(
        epochs,
        [h["diversity_loss"] for h in history],
        label="Diversity"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Unsupervised Separation Training")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "loss_curve.png",
        dpi=150
    )

    plt.close()


# ============================================================
# CSV LOG
# ============================================================

def save_training_log(history):

    if not history:
        return

    keys = list(history[0].keys())

    with open(
        RESULT_DIR / "training_log.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# INFERENCE / SEPARATED AUDIO
# ============================================================

@torch.no_grad()
def separate_song(model, path):

    model.eval()

    x = load_audio(path)

    audio = torch.from_numpy(x).to(
        DEVICE,
        non_blocking=True
    )

    X = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW.to(DEVICE),
        return_complex=True
    )

    magnitude = torch.abs(X)

    log_mag = torch.log1p(magnitude)

    model_input = log_mag[None, None]

    masks, activity = model(model_input)

    masks = masks[0]
    activity = activity[0]

    # Use mixture phase.
    phase = torch.angle(X)

    output_dir = (
        RESULT_DIR
        / "separated"
        / path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(f"SEPARATION: {path.name}")
    print("=" * 70)

    print("Learned slot activity:")

    for i, a in enumerate(activity):
        print(
            f"  Slot {i:02d}: "
            f"{float(a):.4f}"
        )

    for i in range(MAX_SOURCES):

        source_mag = masks[i] * magnitude

        source_complex = (
            source_mag
            * torch.exp(1j * phase)
        )

        source_audio = torch.istft(
            source_complex,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=WINDOW.to(DEVICE),
            length=len(audio)
        )

        source_audio = source_audio.cpu().numpy()

        # Normalize conservatively.
        peak = np.max(
            np.abs(source_audio)
        ) + 1e-8

        if peak > 1.0:
            source_audio = source_audio / peak

        output_path = (
            output_dir
            / f"slot_{i:02d}.wav"
        )

        sf.write(
            output_path,
            source_audio,
            SAMPLE_RATE
        )

    with open(
        output_dir / "activity.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for i, a in enumerate(activity):
            f.write(
                f"slot_{i:02d}: "
                f"{float(a):.6f}\n"
            )

    print(f"\nSaved outputs to: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UNSUPERVISED MULTI-SOURCE MUSIC SEPARATOR")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    SONG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    song_paths = sorted(
        [
            p for p in SONG_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]
    )

    if len(song_paths) == 0:

        print("\nNo songs found.")

        print(
            "\nPut audio files such as .wav, .flac, .ogg, .mp3, or .m4a into:"
        )

        print(
            f"    {SONG_DIR.resolve()}"
        )

        return

    print(
        f"\nFound {len(song_paths)} songs."
    )

    dataset = MixedSongDataset(
        song_paths
    )

    if len(dataset.song_paths) == 0:
        print("No readable songs.")
        return

    # --------------------------------------------------------
    # CPU -> GPU pipeline
    #
    # CPU workers prepare random chunks in parallel while the
    # GPU trains on the previous batch.
    # --------------------------------------------------------
    worker_count = min(CPU_WORKERS, max(1, len(dataset)))

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": (DEVICE.type == "cuda"),
    }

    if worker_count > 0:
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    loader = DataLoader(**loader_kwargs)

    print("\n" + "=" * 70)
    print("PARALLEL PIPELINE")
    print("=" * 70)
    print(f"CPU DataLoader workers    : {worker_count}")
    print(f"Persistent workers        : {PERSISTENT_WORKERS if worker_count > 0 else False}")
    print(f"Prefetch factor           : {PREFETCH_FACTOR if worker_count > 0 else 0}")
    print(f"Pinned memory             : {DEVICE.type == 'cuda'}")
    print(f"Mixed precision (AMP)     : {USE_AMP and DEVICE.type == 'cuda'}")
    print(f"CUDA benchmark            : {ENABLE_CUDNN_BENCHMARK}")

    model = UnsupervisedSeparator(
        MAX_SOURCES
    ).to(DEVICE)

    save_config()

    print_model_parameters(model)

    history = train(
        model,
        loader
    )

    save_training_log(history)
    plot_loss(history)

    # --------------------------------------------------------
    # Separate the first few songs after training.
    # --------------------------------------------------------

    print("\nRunning post-training separation...")

    for path in song_paths[
        :min(5, len(song_paths))
    ]:
        separate_song(
            model,
            path
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    print(
        "\nInspect:"
        "\n  results/loss_curve.png"
        "\n  results/training_log.csv"
        "\n  results/model_parameters.txt"
        "\n  results/separated/"
    )


if __name__ == "__main__":
    main()
