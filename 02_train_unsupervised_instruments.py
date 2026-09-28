
"""
02_train_unsupervised_instruments.py

UNSUPERVISED MUSIC / INSTRUMENT-DISCOVERY EXPERIMENT - V2

Input:
    prepared/music/*.wav

No clean stems.
No instrument labels.
No fixed instrument identity per slot.

The model has SOURCE_CAPACITY slots (default 8), but activity gates allow
different songs/chunks to use different numbers of slots.

IMPORTANT:
    This is an unsupervised source-discovery experiment. Reconstruction alone
    cannot mathematically guarantee "slot 2 = guitar". The losses below are
    designed to discourage trivial solutions and encourage sparse, distinct,
    temporally coherent components.

Hardware:
    CUDA -> GPU training + CPU DataLoader workers
    CPU  -> multi-worker data preparation + CPU training

The script automatically uses CUDA when PyTorch can see it.

Outputs:
    results_unsupervised/
        config.json
        model_parameters.txt
        training_log.csv
        loss_curve.png
        source_activity.png
        checkpoints/
        separated/
            song_name/
                slot_00.wav
                ...
                activity.txt
"""


import csv
import json
import os
import random
import time
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

MUSIC_DIR = Path("prepared/music")
RESULT_DIR = Path("results_unsupervised")

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac",
    ".aif", ".aiff", ".opus", ".wma", ".webm", ".mp4", ".mka"
}

# ------------------------------------------------------------
# SOURCE CAPACITY
# ------------------------------------------------------------
# This is deliberately NOT "number of instruments".
#
# Example:
#   chunk A -> 3 active slots
#   chunk B -> 5 active slots
#   chunk C -> 2 active slots
#
# Eight is simply the maximum number of source representations available.
SOURCE_CAPACITY = 8

# ------------------------------------------------------------
# AUDIO
# ------------------------------------------------------------

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
CHUNK_SECONDS = 6

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

EPOCHS = 100
BATCH_SIZE = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

CHUNKS_PER_SONG = 12

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

HIDDEN = 96
DROPOUT = 0.05

# ------------------------------------------------------------
# LOSS WEIGHTS
# ------------------------------------------------------------

# Spectral reconstruction.
LAMBDA_RECON = 1.0

# Penalize masks that spread one TF bin over every source.
LAMBDA_ENTROPY = 0.01

# Encourage smooth TF regions rather than isolated checkerboard masks.
LAMBDA_SMOOTH = 0.02

# Make different source representations less correlated.
LAMBDA_ORTHOGONAL = 0.05

# Prevent every source slot from being active all the time.
LAMBDA_ACTIVITY = 0.002

# Encourage the network to use only as many slots as necessary.
LAMBDA_USAGE = 0.001

# ------------------------------------------------------------
# HARDWARE
# ------------------------------------------------------------

# Do not silently start a long CPU training run when GPU training is required.
REQUIRE_CUDA = True
FORCE_CPU = False

# Leave two logical CPU cores for Windows / background tasks.
CPU_WORKERS = max(1, (os.cpu_count() or 4) - 2)

PREFETCH_FACTOR = 4
PIN_MEMORY = True

USE_AMP = True

ENABLE_CUDNN_BENCHMARK = True

SEED = 42


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HARDWARE DETECTION
# ============================================================

def select_device():

    cuda_available = torch.cuda.is_available()

    if REQUIRE_CUDA and not cuda_available:
        raise RuntimeError(
            "GPU training is required, but CUDA is unavailable. "
            f"PyTorch={torch.__version__}, "
            f"CUDA build={torch.version.cuda}, "
            f"CUDA built={torch.backends.cuda.is_built()}. "
            "Install a CUDA-enabled PyTorch wheel in myenv."
        )

    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if cuda_available else "cpu"
        )

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("\n" + "=" * 78)
    print("HARDWARE DETECTION")
    print("=" * 78)

    print(f"PyTorch version        : {torch.__version__}")
    print(f"PyTorch CUDA version   : {torch.version.cuda}")
    print(f"CUDA available         : {cuda_available}")
    print(f"cuDNN available        : {torch.backends.cudnn.is_available()}")
    print(f"CPU logical cores      : {os.cpu_count()}")
    print(f"Selected device        : {device}")

    if cuda_available:

        print(
            f"GPU count              : "
            f"{torch.cuda.device_count()}"
        )

        for i in range(torch.cuda.device_count()):

            props = torch.cuda.get_device_properties(i)

            print(
                f"GPU {i}                  : {props.name}"
            )

            print(
                f"GPU {i} VRAM             : "
                f"{props.total_memory / 1024**3:.2f} GB"
            )

            print(
                f"GPU {i} compute capability: "
                f"{props.major}.{props.minor}"
            )

    else:

        print(
            "\nWARNING: PyTorch cannot see a CUDA GPU."
        )

        print(
            "Training will continue on CPU."
        )

    return device


DEVICE = select_device()


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    x, fs = sf.read(
        path,
        always_2d=False
    )

    if x.ndim > 1:
        x = np.mean(
            x,
            axis=1
        )

    x = x.astype(
        np.float32
    )

    # Resample with linear interpolation for this prototype.
    if fs != SAMPLE_RATE:

        duration = len(x) / fs

        new_length = int(
            duration * SAMPLE_RATE
        )

        old_t = np.linspace(
            0,
            duration,
            len(x),
            endpoint=False
        )

        new_t = np.linspace(
            0,
            duration,
            new_length,
            endpoint=False
        )

        x = np.interp(
            new_t,
            old_t,
            x
        ).astype(
            np.float32
        )

    # Per-song peak normalization.
    peak = np.max(
        np.abs(x)
    ) + 1e-8

    x = x / peak

    return torch.from_numpy(x)


# ============================================================
# DATASET
# ============================================================

class MusicDataset(Dataset):

    def __init__(self, paths):

        self.paths = paths

        self.chunk_samples = (
            CHUNK_SECONDS * SAMPLE_RATE
        )

        self.audio = {}

        print("\n" + "=" * 78)
        print("LOADING MUSIC DATASET")
        print("=" * 78)

        for path in paths:

            try:

                audio = load_audio(path)

                self.audio[path] = audio

                print(
                    f"{path.name:45s}"
                    f"{len(audio) / SAMPLE_RATE:8.2f} sec"
                )

            except Exception as e:

                print(
                    f"FAILED {path.name}: {e}"
                )

        self.paths = list(
            self.audio.keys()
        )

    def __len__(self):

        return (
            len(self.paths)
            * CHUNKS_PER_SONG
        )

    def __getitem__(self, idx):

        song_index = (
            idx % len(self.paths)
        )

        path = self.paths[song_index]

        audio = self.audio[path]

        if len(audio) < self.chunk_samples:

            chunk = F.pad(
                audio,
                (
                    0,
                    self.chunk_samples
                    - len(audio)
                )
            )

        else:

            start = random.randint(
                0,
                len(audio)
                - self.chunk_samples
            )

            chunk = audio[
                start:
                start + self.chunk_samples
            ]

        return chunk


# ============================================================
# MODEL
# ============================================================

class InstrumentDiscoveryNet(nn.Module):

    """
    Slot-based unsupervised separator.

    Input:
        log magnitude spectrogram

    Output:
        masks:    [B, K, F, T]
        activity: [B, K]

    K is SOURCE_CAPACITY, not a fixed instrument count.
    """

    def __init__(
        self,
        source_capacity
    ):

        super().__init__()

        self.source_capacity = source_capacity

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                HIDDEN,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm2d(HIDDEN),

            nn.GELU(),

            nn.Conv2d(
                HIDDEN,
                HIDDEN,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(HIDDEN),

            nn.GELU(),

            nn.Dropout2d(
                DROPOUT
            ),

            nn.Conv2d(
                HIDDEN,
                HIDDEN,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),
        )

        # Each output channel represents one discovered source slot.
        self.mask_head = nn.Conv2d(
            HIDDEN,
            source_capacity,
            kernel_size=1
        )

        # Global evidence for whether each slot is present.
        self.activity_head = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            ),

            nn.GELU(),

            nn.Linear(
                HIDDEN,
                source_capacity
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        mask_logits = self.mask_head(z)

        # Competitive allocation:
        # every TF bin is assigned across the source slots.
        masks = torch.softmax(
            mask_logits,
            dim=1
        )

        activity_logits = self.activity_head(z)

        activity = torch.sigmoid(
            activity_logits
        )

        # Activity controls source strength.
        gated = (
            masks
            * activity[:, :, None, None]
        )

        # IMPORTANT:
        # Do not renormalize here.
        #
        # If all activity values are small, the model is allowed
        # to leave energy unexplained. Reconstruction loss prevents
        # it from doing that everywhere.
        #
        # This makes activity meaningful instead of mathematically
        # forcing all eight slots to be active.

        return gated, activity


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def reconstruction_loss(
    mixture_mag,
    masks
):

    predicted_mag = (
        masks * mixture_mag
    ).sum(
        dim=1,
        keepdim=True
    )

    # Relative reconstruction error.
    target = mixture_mag

    return F.l1_loss(
        predicted_mag,
        target
    )


def entropy_loss(masks):

    # Normalize the masks for calculating assignment entropy.
    normalized = masks / (
        masks.sum(
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    entropy = -(
        normalized
        * torch.log(
            normalized + 1e-8
        )
    ).sum(
        dim=1
    )

    return entropy.mean()


def smoothness_loss(masks):

    time_diff = torch.abs(
        masks[:, :, :, 1:]
        - masks[:, :, :, :-1]
    ).mean()

    freq_diff = torch.abs(
        masks[:, :, 1:, :]
        - masks[:, :, :-1, :]
    ).mean()

    return time_diff + freq_diff


def orthogonal_loss(
    mixture_mag,
    masks
):

    """
    Penalize different source masks from becoming identical.

    We compare normalized TF vectors for each source.
    """

    b, k, f, t = masks.shape

    source_energy = (
        masks * mixture_mag
    )

    vectors = source_energy.reshape(
        b,
        k,
        -1
    )

    vectors = F.normalize(
        vectors,
        dim=-1
    )

    similarity = torch.bmm(
        vectors,
        vectors.transpose(1, 2)
    )

    identity = torch.eye(
        k,
        device=masks.device
    ).unsqueeze(0)

    off_diag = (
        similarity
        * (1 - identity)
    )

    return (
        off_diag ** 2
    ).mean()


def activity_loss(activity):

    return activity.mean()


def usage_loss(activity):

    """
    Prevent all slots from being permanently inactive.

    This is deliberately weak.
    """

    mean_activity = activity.mean(
        dim=0
    )

    target = 0.15

    return F.relu(
        target - mean_activity
    ).mean()


# ============================================================
# STFT
# ============================================================

WINDOW = torch.hann_window(
    N_FFT
)


def log_magnitude(
    audio
):

    window = WINDOW.to(
        audio.device
    )

    X = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(X)

    log_mag = torch.log1p(
        magnitude
    )

    return (
        log_mag.unsqueeze(1),
        X
    )


# ============================================================
# PARAMETER REPORT
# ============================================================

def parameter_report(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 78)
    print("MODEL PARAMETERS")
    print("=" * 78)

    print(
        f"Model                  : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Source capacity        : "
        f"{SOURCE_CAPACITY}"
    )

    print(
        f"Trainable parameters   : "
        f"{trainable:,}"
    )

    print(
        f"Total parameters       : "
        f"{total:,}"
    )

    print(
        f"Sample rate            : "
        f"{SAMPLE_RATE}"
    )

    print(
        f"N FFT                  : "
        f"{N_FFT}"
    )

    print(
        f"Hop length             : "
        f"{HOP_LENGTH}"
    )

    print(
        f"Chunk duration         : "
        f"{CHUNK_SECONDS} sec"
    )

    print(
        f"Batch size             : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs                 : "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate          : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay           : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Optimizer              : AdamW"
    )

    print(
        f"AMP                    : "
        f"{USE_AMP and DEVICE.type == 'cuda'}"
    )

    print("\nArchitecture:")
    print(model)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_DIR / "model_parameters.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "UNSUPERVISED INSTRUMENT DISCOVERY\n"
        )

        f.write(
            "=" * 78 + "\n\n"
        )

        f.write(
            f"Model: {model.__class__.__name__}\n"
        )

        f.write(
            f"Source capacity: {SOURCE_CAPACITY}\n"
        )

        f.write(
            f"Trainable parameters: {trainable:,}\n"
        )

        f.write(
            f"Total parameters: {total:,}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Sample rate: {SAMPLE_RATE}\n"
        )

        f.write(
            f"N FFT: {N_FFT}\n"
        )

        f.write(
            f"Hop length: {HOP_LENGTH}\n"
        )

        f.write(
            f"Chunk seconds: {CHUNK_SECONDS}\n"
        )

        f.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )

        f.write(
            f"Weight decay: {WEIGHT_DECAY}\n"
        )

        f.write(
            f"Optimizer: AdamW\n"
        )

        f.write(
            f"AMP: "
            f"{USE_AMP and DEVICE.type == 'cuda'}\n\n"
        )

        f.write(
            "LOSS WEIGHTS\n"
        )

        f.write(
            "=" * 78 + "\n"
        )

        f.write(
            f"LAMBDA_RECON: {LAMBDA_RECON}\n"
        )

        f.write(
            f"LAMBDA_ENTROPY: {LAMBDA_ENTROPY}\n"
        )

        f.write(
            f"LAMBDA_SMOOTH: {LAMBDA_SMOOTH}\n"
        )

        f.write(
            f"LAMBDA_ORTHOGONAL: "
            f"{LAMBDA_ORTHOGONAL}\n"
        )

        f.write(
            f"LAMBDA_ACTIVITY: "
            f"{LAMBDA_ACTIVITY}\n"
        )

        f.write(
            f"LAMBDA_USAGE: "
            f"{LAMBDA_USAGE}\n"
        )

        f.write(
            "\nARCHITECTURE\n"
        )

        f.write(
            "=" * 78 + "\n"
        )

        f.write(
            str(model)
        )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch
):

    directory = (
        RESULT_DIR
        / "checkpoints"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
        },
        directory
        / f"checkpoint_{epoch:03d}.pt"
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    model,
    loader
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

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

    iterations = len(loader)

    print("\n" + "=" * 78)
    print("TRAINING")
    print("=" * 78)

    print(
        f"Songs                   : "
        f"{len(loader.dataset.paths)}"
    )

    print(
        f"Chunks per song         : "
        f"{CHUNKS_PER_SONG}"
    )

    print(
        f"Iterations / epoch      : "
        f"{iterations}"
    )

    print(
        f"Total iterations        : "
        f"{iterations * EPOCHS}"
    )

    print(
        f"Batch size              : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs                  : "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate           : "
        f"{LEARNING_RATE}"
    )

    print(
        f"CPU workers             : "
        f"{CPU_WORKERS}"
    )

    print(
        f"GPU AMP                 : "
        f"{amp_enabled}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        totals = {
            "loss": 0.0,
            "recon": 0.0,
            "entropy": 0.0,
            "smooth": 0.0,
            "orthogonal": 0.0,
            "activity": 0.0,
            "usage": 0.0,
        }

        start_time = time.time()

        for iteration, audio in enumerate(
            loader,
            start=1
        ):

            audio = audio.to(
                DEVICE,
                non_blocking=True
            )

            if amp_enabled:

                context = torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                )

            else:

                context = torch.autocast(
                    device_type="cpu",
                    enabled=False
                )

            with context:

                log_mag, X = log_magnitude(
                    audio
                )

                magnitude = torch.abs(X)

                masks, activity = model(
                    log_mag
                )

                loss_recon = reconstruction_loss(
                    magnitude.unsqueeze(1),
                    masks
                )

                loss_entropy = entropy_loss(
                    masks
                )

                loss_smooth = smoothness_loss(
                    masks
                )

                loss_orthogonal = orthogonal_loss(
                    magnitude.unsqueeze(1),
                    masks
                )

                loss_activity = activity_loss(
                    activity
                )

                loss_usage = usage_loss(
                    activity
                )

                loss = (

                    LAMBDA_RECON
                    * loss_recon

                    + LAMBDA_ENTROPY
                    * loss_entropy

                    + LAMBDA_SMOOTH
                    * loss_smooth

                    + LAMBDA_ORTHOGONAL
                    * loss_orthogonal

                    + LAMBDA_ACTIVITY
                    * loss_activity

                    + LAMBDA_USAGE
                    * loss_usage
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            if amp_enabled:

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0
                    )
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0
                    )
                )

                optimizer.step()

            totals["loss"] += (
                loss.item()
            )

            totals["recon"] += (
                loss_recon.item()
            )

            totals["entropy"] += (
                loss_entropy.item()
            )

            totals["smooth"] += (
                loss_smooth.item()
            )

            totals["orthogonal"] += (
                loss_orthogonal.item()
            )

            totals["activity"] += (
                loss_activity.item()
            )

            totals["usage"] += (
                loss_usage.item()
            )

            if (
                iteration == 1
                or iteration % 10 == 0
            ):

                if DEVICE.type == "cuda":

                    allocated = (
                        torch.cuda
                        .memory_allocated()
                        / 1024**3
                    )

                    reserved = (
                        torch.cuda
                        .memory_reserved()
                        / 1024**3
                    )

                    gpu_info = (
                        f" | VRAM "
                        f"{allocated:.2f}/"
                        f"{reserved:.2f} GB"
                    )

                else:

                    gpu_info = ""

                print(
                    f"Epoch {epoch:03d}/{EPOCHS} | "
                    f"Iter {iteration:04d}/{iterations} | "
                    f"Loss {loss.item():.6f} | "
                    f"Recon {loss_recon.item():.6f} | "
                    f"Orth {loss_orthogonal.item():.6f} | "
                    f"Grad {float(grad_norm):.4f}"
                    f"{gpu_info}"
                )

        scheduler.step()

        n = iterations

        row = {
            "epoch": epoch,
            "loss": totals["loss"] / n,
            "reconstruction_loss":
                totals["recon"] / n,
            "entropy_loss":
                totals["entropy"] / n,
            "smoothness_loss":
                totals["smooth"] / n,
            "orthogonal_loss":
                totals["orthogonal"] / n,
            "activity_loss":
                totals["activity"] / n,
            "usage_loss":
                totals["usage"] / n,
            "learning_rate":
                optimizer.param_groups[0]["lr"],
            "epoch_time_sec":
                time.time() - start_time,
        }

        history.append(row)

        print(
            "\n"
            f">>> EPOCH {epoch:03d} COMPLETE | "
            f"Loss={row['loss']:.6f} | "
            f"Recon={row['reconstruction_loss']:.6f} | "
            f"Orth={row['orthogonal_loss']:.6f} | "
            f"LR={row['learning_rate']:.8f} | "
            f"Time={row['epoch_time_sec']:.1f}s\n"
        )

        if (
            epoch % 10 == 0
            or epoch == EPOCHS
        ):

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch
            )

    return history


# ============================================================
# SAVE LOG
# ============================================================

def save_history(history):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_DIR / "training_log.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        keys = list(
            history[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# PLOTS
# ============================================================

def plot_history(history):

    epochs = [
        row["epoch"]
        for row in history
    ]

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        epochs,
        [
            row["loss"]
            for row in history
        ],
        label="Total loss"
    )

    plt.plot(
        epochs,
        [
            row["reconstruction_loss"]
            for row in history
        ],
        label="Reconstruction"
    )

    plt.plot(
        epochs,
        [
            row["orthogonal_loss"]
            for row in history
        ],
        label="Orthogonal"
    )

    plt.plot(
        epochs,
        [
            row["entropy_loss"]
            for row in history
        ],
        label="Entropy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Unsupervised Instrument Discovery"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "loss_curve.png",
        dpi=150
    )

    plt.close()


# ============================================================
# SEPARATION
# ============================================================

@torch.no_grad()
def separate_song(
    model,
    path
):

    model.eval()

    audio = load_audio(
        path
    ).to(
        DEVICE
    )

    window = WINDOW.to(
        DEVICE
    )

    X = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(X)

    log_mag = torch.log1p(
        magnitude
    )

    masks, activity = model(
        log_mag[None, None]
    )

    masks = masks[0]

    activity = activity[0]

    output_dir = (
        RESULT_DIR
        / "separated"
        / path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 78)
    print(
        f"SEPARATION: {path.name}"
    )
    print("=" * 78)

    print(
        "Learned activity:"
    )

    for i in range(
        SOURCE_CAPACITY
    ):

        print(
            f"  Slot {i:02d}: "
            f"{float(activity[i]):.5f}"
        )

    phase = torch.angle(
        X
    )

    for i in range(
        SOURCE_CAPACITY
    ):

        source_mag = (
            masks[i]
            * magnitude
        )

        source_complex = (
            source_mag
            * torch.exp(
                1j * phase
            )
        )

        source_audio = (
            torch.istft(
                source_complex,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                window=window,
                length=len(audio)
            )
        )

        source_audio = (
            source_audio
            .cpu()
            .numpy()
        )

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

        for i in range(
            SOURCE_CAPACITY
        ):

            f.write(
                f"slot_{i:02d}: "
                f"{float(activity[i]):.6f}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 78)
    print(
        "UNSUPERVISED INSTRUMENT DISCOVERY - V2"
    )
    print("=" * 78)

    paths = sorted(
        p
        for p in MUSIC_DIR.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in AUDIO_EXTENSIONS
        )
    )

    if not paths:

        print(
            "\nNo music files found in:"
        )

        print(
            MUSIC_DIR.resolve()
        )

        print(
            "\nRun the Demucs preparation script first."
        )

        return

    print(
        f"\nFound {len(paths)} music files."
    )

    dataset = MusicDataset(
        paths
    )

    if not dataset.paths:

        print(
            "No readable music files."
        )

        return

    worker_count = min(
        CPU_WORKERS,
        max(
            1,
            len(dataset)
        )
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": (
            PIN_MEMORY
            and DEVICE.type == "cuda"
        ),
    }

    if worker_count > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

    loader = DataLoader(
        **loader_kwargs
    )

    print("\n" + "=" * 78)
    print("DATA PIPELINE")
    print("=" * 78)

    print(
        f"DataLoader workers      : "
        f"{worker_count}"
    )

    print(
        f"Pinned memory            : "
        f"{loader_kwargs['pin_memory']}"
    )

    print(
        f"Prefetch factor          : "
        f"{PREFETCH_FACTOR}"
    )

    model = (
        InstrumentDiscoveryNet(
            SOURCE_CAPACITY
        )
        .to(DEVICE)
    )

    parameter_report(
        model
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config = {
        "music_directory":
            str(MUSIC_DIR),
        "sample_rate":
            SAMPLE_RATE,
        "n_fft":
            N_FFT,
        "hop_length":
            HOP_LENGTH,
        "chunk_seconds":
            CHUNK_SECONDS,
        "source_capacity":
            SOURCE_CAPACITY,
        "epochs":
            EPOCHS,
        "batch_size":
            BATCH_SIZE,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "chunks_per_song":
            CHUNKS_PER_SONG,
        "hidden_channels":
            HIDDEN,
        "device":
            str(DEVICE),
        "cuda_available":
            torch.cuda.is_available(),
        "cpu_workers":
            worker_count,
        "amp":
            USE_AMP and DEVICE.type == "cuda",
        "lambda_recon":
            LAMBDA_RECON,
        "lambda_entropy":
            LAMBDA_ENTROPY,
        "lambda_smooth":
            LAMBDA_SMOOTH,
        "lambda_orthogonal":
            LAMBDA_ORTHOGONAL,
        "lambda_activity":
            LAMBDA_ACTIVITY,
        "lambda_usage":
            LAMBDA_USAGE,
    }

    with open(
        RESULT_DIR / "config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    history = train(
        model,
        loader
    )

    save_history(
        history
    )

    plot_history(
        history
    )

    print(
        "\nRunning post-training separation..."
    )

    for path in paths:

        separate_song(
            model,
            path
        )

    print("\n" + "=" * 78)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        "\nResults:"
    )

    print(
        f"  {RESULT_DIR.resolve()}"
    )

    print(
        "\nInspect first:"
        "\n  loss_curve.png"
        "\n  training_log.csv"
        "\n  model_parameters.txt"
        "\n  separated/"
    )


if __name__ == "__main__":
    main()
