"""
V4: acoustic latent learning + MIX-AND-SEPARATE (cross-song signal).

Why this exists
---------------
V3 trained on two augmentations of the SAME chunk. That is solved by a
per-song fingerprint, so slot loss collapsed to ~0 and nothing forced the
slots to represent separable components.

Fix: create synthetic mixtures with KNOWN ground truth.
    mix = chunk_from_song_i + gain * chunk_from_song_j      (i != j)
The slot network sees only `mix`. We know the two sources, so we can ask:
  1. SEPARATION: masks * |mix| should reproduce |source_1| and |source_2|
     in some slot pair (permutation-invariant, PIT), while the other slots
     output ~0 energy.
  2. EMBEDDING: those same two slots should have embeddings that match the
     encoder's embedding of each clean source (encoder is the target,
     detached).
  3. ACTIVITY: the two assigned slots are "active" (1), the rest "inactive"
     (0). This gives a real, trained activity head = variable source count.
Global contrastive loss (same-chunk InfoNCE) is kept for the encoder.

Nothing here uses instrument labels. At inference the model is applied to
real songs (already mixtures) and slots/activity/masks are inspected.

Run:
    python 04_train_mix_separate.py
    python 04_train_mix_separate.py --extract-only results_latent/checkpoints/checkpoint_080.pt
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse, csv, json, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- CONFIG ----------------
MUSIC_DIR = Path("prepared/music")
RESULT_DIR = Path("results_latent")
EXT = {".wav",".flac",".ogg",".mp3",".m4a",".aac",".aif",".aiff",".opus",".wma",".webm",".mp4",".mka"}

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128
CHUNK_SECONDS = 2

SOURCE_SLOTS = 4
LATENT_DIM = 128
CHANNELS = 24

EPOCHS = 80
BATCH_SIZE = 4
CHUNKS_PER_SONG = 40
LR = 3e-4
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.15
MIX_GAIN_DB = 6.0          # random relative level of source 2 vs source 1

L_GLOBAL = 1.0
L_SEP = 1.0
L_EMB = 0.5
L_ACT = 0.1
L_ORTH = 0.1

FORCE_CPU = False
CPU_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 2))
PREFETCH = 4
USE_AMP = True
SEED = 42

PERMS = [(i, j) for i in range(SOURCE_SLOTS) for j in range(SOURCE_SLOTS) if i != j]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CUDA = torch.cuda.is_available() and not FORCE_CPU
DEVICE = torch.device("cuda" if CUDA else "cpu")
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

WINDOW = torch.hann_window(N_FFT)

def print_hardware():
    print("\n" + "="*75 + "\nHARDWARE\n" + "="*75)
    print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda,
          "| available:", torch.cuda.is_available(), "| device:", DEVICE)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU 0: {p.name} | {p.total_memory/1024**3:.2f} GB")

# ---------------- AUDIO ----------------
def load_audio(path):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if fs != SAMPLE_RATE:
        dur = len(x) / fs
        n = int(dur * SAMPLE_RATE)
        x = np.interp(np.linspace(0,dur,n,endpoint=False),
                      np.linspace(0,dur,len(x),endpoint=False), x).astype(np.float32)
    x /= np.max(np.abs(x)) + 1e-8
    return torch.from_numpy(x)

def augment(x):
    y = x.clone()
    y *= 10 ** (random.uniform(-4,4)/20)
    if random.random() < .35:
        rms = torch.sqrt(torch.mean(y*y)+1e-8)
        y += rms*random.uniform(.001,.01)*torch.randn_like(y)
    if random.random() < .30:
        n = len(y)
        m = max(1,int(n*random.uniform(.03,.10)))
        s = random.randint(0,max(0,n-m))
        y[s:s+m] *= .05
    return y

def stft_mag(x):
    x = x.contiguous()
    X = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW.to(x.device),
                   return_complex=True, center=True)
    return torch.log1p(torch.abs(X)), X

# ---------------- DATASET ----------------
class MusicDataset(Dataset):
    """Returns [2, n]: a chunk from song i and a chunk from a DIFFERENT song j."""
    def __init__(self, paths):
        self.audio = {}
        self.n = CHUNK_SECONDS*SAMPLE_RATE
        print("\nLOADING MUSIC")
        for p in paths:
            try:
                self.audio[p] = load_audio(p)
                print(f"{p.name:40s} {len(self.audio[p])/SAMPLE_RATE:7.2f}s")
            except Exception as e:
                print("FAILED", p, e)
        self.paths = list(self.audio)
        if len(self.paths) < 2:
            raise RuntimeError("Mix-and-separate needs at least 2 songs.")

    def _crop(self, x):
        if len(x) <= self.n:
            return F.pad(x, (0, self.n-len(x)))
        s = random.randint(0, len(x)-self.n)
        return x[s:s+self.n]

    def __len__(self):
        return len(self.paths)*CHUNKS_PER_SONG

    def __getitem__(self, i):
        i0 = i % len(self.paths)
        i1 = random.choice([k for k in range(len(self.paths)) if k != i0])
        return torch.stack([self._crop(self.audio[self.paths[i0]]),
                            self._crop(self.audio[self.paths[i1]])])

# ---------------- NETWORK ----------------
class AcousticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        C = CHANNELS
        self.net = nn.Sequential(
            nn.Conv2d(1,C,5,padding=2), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C,C,3,stride=2,padding=1), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C,2*C,3,stride=2,padding=1), nn.BatchNorm2d(2*C), nn.GELU(),
            nn.Conv2d(2*C,2*C,3,stride=2,padding=1), nn.BatchNorm2d(2*C), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(2*C,2*C), nn.GELU(),
                                  nn.Linear(2*C,LATENT_DIM))
    def forward(self, x):
        return F.normalize(self.proj(self.net(x)), dim=-1)

class SlotSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        C = CHANNELS
        self.backbone = nn.Sequential(
            nn.Conv2d(1,C,5,padding=2), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C,C,3,padding=1), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C,C,3,stride=2,padding=1), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C,C,3,padding=1), nn.GELU())
        self.mask = nn.Conv2d(C, SOURCE_SLOTS, 1)
        self.proj = nn.Sequential(nn.Linear(C,C), nn.GELU(), nn.Linear(C,LATENT_DIM))
        self.activity = nn.Linear(C, 1)

    def forward(self, x):
        h = self.backbone(x)
        m = F.interpolate(self.mask(h), size=x.shape[-2:], mode="bilinear", align_corners=False)
        m = torch.sigmoid(m)                       # independent per-slot masks
        mw = F.interpolate(m, size=h.shape[-2:], mode="bilinear", align_corners=False)
        slots, acts = [], []
        for k in range(SOURCE_SLOTS):
            w = mw[:, k:k+1]
            pooled = (h*w).sum((2,3)) / (w.sum((2,3)) + 1e-6)
            slots.append(F.normalize(self.proj(pooled), dim=-1))
            acts.append(torch.sigmoid(self.activity(pooled)))
        return m, torch.stack(slots, 1), torch.cat(acts, 1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AcousticEncoder()
        self.slots = SlotSeparator()
    def forward(self, x):
        z = self.encoder(x)
        m, s, a = self.slots(x)
        return z, m, s, a

# ---------------- LOSSES ----------------
def info_nce(a, b, temp):
    n = a.shape[0]
    with torch.autocast(device_type=DEVICE.type, enabled=False):
        z = F.normalize(torch.cat([a, b], 0).float(), dim=-1)
        sim = (z @ z.T) / float(temp)
        eye = torch.eye(2*n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))
        target = torch.cat([torch.arange(n, device=z.device)+n, torch.arange(n, device=z.device)])
        return F.cross_entropy(sim, target)

def orth_loss(s):
    s = s.float()
    sim = torch.bmm(s, s.transpose(1, 2))
    eye = torch.eye(s.shape[1], device=s.device)[None]
    return (sim*(1-eye)).pow(2).mean()

def separation_losses(masks, slots, acts, mixmag, tA, tB, zA, zB):
    """
    Permutation-invariant loss over ordered pairs of DISTINCT slots.
    All in float32.
      masks [B,K,F,T]  slots [B,K,D]  acts [B,K]
      mixmag/tA/tB [B,F,T] linear magnitudes   zA/zB [B,D] target embeddings
    """
    with torch.autocast(device_type=DEVICE.type, enabled=False):
        masks, slots, acts = masks.float(), slots.float(), acts.float()
        B = masks.shape[0]
        denom = mixmag.mean((1, 2)) + 1e-6                          # scale-free
        est = masks * mixmag.unsqueeze(1)                           # [B,K,F,T]
        tgt = torch.stack([tA, tB], 1)                              # [B,2,F,T]
        err = (est.unsqueeze(2) - tgt.unsqueeze(1)).abs().mean((3, 4)) / denom[:, None, None]  # [B,K,2]
        zero = est.abs().mean((2, 3)) / denom[:, None]              # [B,K] energy of a slot
        zsum = zero.sum(1)
        tz = torch.stack([zA, zB], 1).float()                       # [B,2,D]
        sim = torch.bmm(slots, tz.transpose(1, 2))                  # [B,K,2]

        sep_c, emb_c = [], []
        for k1, k2 in PERMS:
            sep_c.append(err[:, k1, 0] + err[:, k2, 1] + zsum - zero[:, k1] - zero[:, k2])
            emb_c.append((1 - sim[:, k1, 0]) + (1 - sim[:, k2, 1]))
        sep_c = torch.stack(sep_c, 1); emb_c = torch.stack(emb_c, 1)          # [B,P]
        idx = (L_SEP*sep_c + L_EMB*emb_c).argmin(1)                           # best assignment
        ar = torch.arange(B, device=masks.device)
        sep = sep_c[ar, idx].mean()
        emb = emb_c[ar, idx].mean()

        k1s = torch.tensor([p[0] for p in PERMS], device=masks.device)[idx]
        k2s = torch.tensor([p[1] for p in PERMS], device=masks.device)[idx]
        t_act = torch.zeros_like(acts)
        t_act[ar, k1s] = 1.0; t_act[ar, k2s] = 1.0
        act = -(t_act*torch.log(acts+1e-6) + (1-t_act)*torch.log(1-acts+1e-6)).mean()
        return sep, emb, act

# ---------------- TRAIN ----------------
def train(model, loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    amp = USE_AMP and CUDA
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    hist = []
    print("\n" + "="*75 + "\nMIX-AND-SEPARATE TRAINING\n" + "="*75)
    print("Epochs:", EPOCHS, "| Iter/epoch:", len(loader), "| Batch:", BATCH_SIZE, "| AMP:", amp)

    for ep in range(1, EPOCHS+1):
        model.train()
        sums = np.zeros(6); t0 = time.time()

        for it, batch in enumerate(loader, 1):
            batch = batch.to(DEVICE, non_blocking=True)      # [B,2,n]
            a, b = batch[:, 0], batch[:, 1]
            B = a.shape[0]

            with torch.no_grad():
                # ---- synthetic mixture with known sources ----
                gain = 10 ** (torch.empty(B, device=DEVICE).uniform_(-MIX_GAIN_DB, MIX_GAIN_DB) / 20)
                b_s = b * gain[:, None]
                mix = a + b_s
                peak = mix.abs().amax(1, keepdim=True) + 1e-8
                a_n, b_n, mix_n = a/peak, b_s/peak, mix/peak
                lmix, Xmix = stft_mag(mix_n)
                lA, XA = stft_mag(a_n)
                lB, XB = stft_mag(b_n)
                mixmag, tA, tB = Xmix.abs(), XA.abs(), XB.abs()
                # ---- two augmented views of chunk a for contrastive loss ----
                v1 = torch.stack([augment(x) for x in a])
                v2 = torch.stack([augment(x) for x in a])
                m1, _ = stft_mag(v1)
                m2, _ = stft_mag(v2)

            ctx = torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp)
            with ctx:
                z1 = model.encoder(m1[:, None])
                z2 = model.encoder(m2[:, None])
                with torch.no_grad():
                    zt = model.encoder(torch.cat([lA, lB], 0)[:, None]).float()
                zA, zB = zt[:B], zt[B:]
                masks, slots, acts = model.slots(lmix[:, None])

                lg = info_nce(z1, z2, TEMPERATURE)
                ls, le, la = separation_losses(masks, slots, acts, mixmag, tA, tB, zA, zB)
                lo = orth_loss(slots)
                loss = L_GLOBAL*lg + L_SEP*ls + L_EMB*le + L_ACT*la + L_ORTH*lo

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ep={ep} it={it}: global={lg.item()} sep={ls.item()} "
                                   f"emb={le.item()} act={la.item()} orth={lo.item()}")

            opt.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()

            vals = [loss.item(), lg.item(), ls.item(), le.item(), la.item(), lo.item()]
            sums += vals
            gn_val = float(gn)
            del z1, z2, zt, zA, zB, masks, slots, acts, loss, lg, ls, le, la, lo

            if it == 1 or it % 20 == 0:
                gpu = f" | VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GB" if CUDA else ""
                print(f"Ep {ep:03d}/{EPOCHS} It {it:04d}/{len(loader)} | Loss {vals[0]:.4f} | "
                      f"Global {vals[1]:.4f} | Sep {vals[2]:.4f} | Emb {vals[3]:.4f} | "
                      f"Act {vals[4]:.4f} | Grad {gn_val:.2f}{gpu}")

        sched.step()
        r = sums/len(loader)
        row = dict(epoch=ep, loss=r[0], global_contrastive=r[1], separation=r[2],
                   embedding_match=r[3], activity=r[4], orthogonal=r[5],
                   learning_rate=opt.param_groups[0]["lr"], epoch_time_sec=time.time()-t0)
        hist.append(row)
        print(f">>> EPOCH {ep:03d} | Loss={r[0]:.4f} | Global={r[1]:.4f} | Sep={r[2]:.4f} | "
              f"Emb={r[3]:.4f} | Act={r[4]:.4f} | {row['epoch_time_sec']:.1f}s")
        if ep % 10 == 0 or ep == EPOCHS:
            d = RESULT_DIR/"checkpoints"; d.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict()},
                       d/f"checkpoint_{ep:03d}.pt")
    return hist

# ---------------- EXTRACTION / ANALYSIS ----------------
@torch.no_grad()
def extract(model, paths):
    model.eval()
    ge, se, ae, meta = [], [], [], []
    n = CHUNK_SECONDS*SAMPLE_RATE
    for p in paths:
        x = load_audio(p)
        if len(x) <= n:
            starts = [0]
        else:
            count = min(20, max(1, len(x)//SAMPLE_RATE))
            starts = np.linspace(0, len(x)-n, count).astype(int)
        for ci, s in enumerate(starts):
            c = x[s:s+n]
            if len(c) < n: c = F.pad(c, (0, n-len(c)))
            mag, _ = stft_mag(c.to(DEVICE)[None])
            z, m, slots, act = model(mag[:, None])
            ge.append(z[0].cpu().numpy()); se.append(slots[0].cpu().numpy())
            ae.append(act[0].cpu().numpy()); meta.append((p.name, ci, s/SAMPLE_RATE))
    return (np.asarray(ge, np.float32), np.asarray(se, np.float32),
            np.asarray(ae, np.float32), meta)

def slot_diagnostics(se, ae):
    """Collapse check: mean off-diagonal cosine between slots + activity stats."""
    K = se.shape[1]
    sim = np.einsum("ikd,ijd->ikj", se, se)
    off = sim[:, ~np.eye(K, dtype=bool)].mean()
    print("\nSLOT DIAGNOSTICS")
    print(f"Mean cosine between different slots (1.0 = collapsed): {off:.3f}")
    print("Mean activity per slot:", np.round(ae.mean(0), 3))
    print("Avg # active slots (act>0.5) per chunk:", f"{(ae > 0.5).sum(1).mean():.2f}")

def analyze_embeddings(e, meta):
    out = RESULT_DIR/"embeddings"; out.mkdir(parents=True, exist_ok=True)
    np.save(out/"chunk_embeddings.npy", e)
    with open(out/"metadata.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["song", "chunk", "start_seconds"]); w.writerows(meta)

    pca = PCA(n_components=2); xy = pca.fit_transform(e)
    print(f"\nPCA variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

    best = None
    for k in range(2, min(9, len(e))):
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(e)
        score = silhouette_score(e, labels)
        print(f"k={k:02d} | silhouette={score:.4f}")
        if best is None or score > best[0]:
            best = (score, k, labels)
    score, k, labels = best
    print(f"\nAnalysis cluster count: {k}")

    # song purity: if each cluster is ~100% one song, the encoder learned a fingerprint
    songs = np.array([m[0] for m in meta])
    print("\nCLUSTER SONG-PURITY (100% = cluster is just one song)")
    pur = []
    for c in range(k):
        cnt = Counter(songs[labels == c])
        top, n_top = cnt.most_common(1)[0]
        frac = n_top/sum(cnt.values()); pur.append(frac*sum(cnt.values()))
        print(f"cluster {c}: {sum(cnt.values()):3d} chunks | dominant song {top} ({frac*100:.0f}%)")
    print(f"Overall purity: {sum(pur)/len(labels)*100:.0f}%")

    with open(RESULT_DIR/"cluster_assignments.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["song", "chunk", "start_seconds", "cluster"])
        for m, l in zip(meta, labels): w.writerow([*m, int(l)])

    plt.figure(figsize=(10, 7))
    plt.scatter(xy[:, 0], xy[:, 1], c=labels, s=20, alpha=.75)
    plt.xlabel("Latent dimension 1"); plt.ylabel("Latent dimension 2")
    plt.title("Learned Acoustic Latent Space"); plt.grid(True); plt.tight_layout()
    plt.savefig(RESULT_DIR/"latent_pca.png", dpi=160); plt.close()

@torch.no_grad()
def export_slots(model, paths, seconds=8):
    """Apply masks to a real song excerpt (mixture phase) and write one wav per slot."""
    model.eval()
    root = RESULT_DIR/"separated"; root.mkdir(parents=True, exist_ok=True)
    for p in paths:
        x = load_audio(p)
        n = min(len(x), seconds*SAMPLE_RATE)
        s = max(0, len(x)//2 - n//2)
        seg = x[s:s+n].to(DEVICE)
        lm, X = stft_mag(seg[None])
        masks, slots, acts = model.slots(lm[:, None])
        est = masks[0] * X[0]                                        # [K,F,T] complex
        wav = torch.istft(est, n_fft=N_FFT, hop_length=HOP, window=WINDOW.to(DEVICE), length=n).cpu().numpy()
        d = root/p.stem; d.mkdir(parents=True, exist_ok=True)
        sf.write(d/"mixture.wav", seg.cpu().numpy(), SAMPLE_RATE)
        for k in range(SOURCE_SLOTS):
            w = wav[k] / (np.abs(wav[k]).max() + 1e-8) * 0.9
            sf.write(d/f"slot{k}_act{acts[0,k].item():.2f}.wav", w, SAMPLE_RATE)
    print("Wrote per-slot audio to", root)

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", default=None, help="checkpoint (.pt) to load; skips training")
    args = ap.parse_args()

    print_hardware()
    paths = sorted(p for p in MUSIC_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not paths:
        print("No music files found:", MUSIC_DIR.resolve()); return
    print(f"\nFound {len(paths)} music files.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR/"embeddings").mkdir(parents=True, exist_ok=True)

    model = Model().to(DEVICE)
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    if args.extract_only:
        ck = torch.load(args.extract_only, map_location=DEVICE)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print("Loaded", args.extract_only)
    else:
        ds = MusicDataset(paths)
        workers = min(CPU_WORKERS, max(1, len(ds)))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers,
                            pin_memory=CUDA, persistent_workers=True, prefetch_factor=PREFETCH,
                            drop_last=True)
        if CUDA: torch.cuda.empty_cache()
        with open(RESULT_DIR/"config_v4.json", "w") as f:
            json.dump(dict(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop=HOP, chunk_seconds=CHUNK_SECONDS,
                           source_slots=SOURCE_SLOTS, latent_dim=LATENT_DIM, epochs=EPOCHS,
                           batch_size=BATCH_SIZE, lr=LR, temperature=TEMPERATURE,
                           mix_gain_db=MIX_GAIN_DB, weights=dict(g=L_GLOBAL, sep=L_SEP, emb=L_EMB,
                           act=L_ACT, orth=L_ORTH)), f, indent=2)
        hist = train(model, loader)
        with open(RESULT_DIR/"training_log_v4.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
        ep = [r["epoch"] for r in hist]
        plt.figure(figsize=(10, 6))
        for key, label in [("loss","Total"),("global_contrastive","Global contrastive"),
                           ("separation","Separation (PIT)"),("embedding_match","Embedding match"),
                           ("activity","Activity")]:
            plt.plot(ep, [r[key] for r in hist], label=label)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(RESULT_DIR/"loss_curve_v4.png", dpi=150); plt.close()
        torch.save(model.state_dict(), RESULT_DIR/"acoustic_encoder_v4.pt")

    ge, se, ae, meta = extract(model, paths)
    np.save(RESULT_DIR/"embeddings"/"slot_embeddings.npy", se)
    np.save(RESULT_DIR/"embeddings"/"slot_activity.npy", ae)
    slot_diagnostics(se, ae)
    analyze_embeddings(ge, meta)
    export_slots(model, paths)
    print("\nDONE. Listen to results_latent/separated/<song>/slot*.wav")

if __name__ == "__main__":
    main()
