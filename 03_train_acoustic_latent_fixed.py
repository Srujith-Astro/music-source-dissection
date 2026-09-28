"""
V3 acoustic latent-space learning. (FIXED)

Input: prepared/music/*.wav
No instrument labels or stems are used.

The model learns:
1. a global acoustic embedding using self-supervised contrastive learning
2. K source-slot embeddings with permutation-free soft matching
3. weak reconstruction/diversity constraints

After training it extracts embeddings and runs PCA + KMeans only for
inspection. Cluster number is NOT imposed on the network.

Install:
    pip install torch soundfile matplotlib numpy scikit-learn

Run:
    python 03_train_acoustic_latent_fixed.py

CHANGES FROM V6
---------------
1. Fixed UnboundLocalError on `loss`:
   the per-iteration logging print statement ran AFTER `del ... loss`,
   so `loss.item()` referenced a name that no longer existed. Loss (and
   the other scalar sub-losses) are now captured into plain Python floats
   BEFORE the `del`, and the print/logging uses those floats.

2. Fixed the degenerate reconstruction objective:
   masks were produced with softmax across the K slot dimension, so
   sum_k M_k(f,t) == 1 everywhere by construction. That means
       X_hat = sum_k M_k * |X| = |X| * sum_k M_k = |X|
   identically, for ANY mask values — the reconstruction loss was
   mathematically guaranteed to be ~0 regardless of whether the masks
   meant anything. It provided zero learning signal.

   Masks are now produced with an independent sigmoid per slot instead
   of a softmax across slots. Sigmoid masks are NOT constrained to sum
   to 1, so `sum_k M_k * |X| ≈ |X|` is now a real constraint the network
   must learn to satisfy (a soft non-negative decomposition of the
   mixture across slots) rather than a tautology. Entropy loss (which
   assumed a probability simplex over slots from the softmax) is
   dropped in favor of an L1 sparsity penalty on the sigmoid masks,
   which is the correct analogue for independent masks.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv, json, random, time
from pathlib import Path
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
BATCH_SIZE = 4  # Contrastive learning needs negatives; batch=1 is not useful.
CHUNKS_PER_SONG = 10
LR = 3e-4
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.15

L_GLOBAL = 1.0
L_SLOT = 1.0
L_RECON = 0.25
L_ORTH = 0.10
L_SPARSITY = 0.01     # replaces L_ENTROPY (see docstring)
L_ACTIVITY = 0.005

FORCE_CPU = False
CPU_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 2))
PREFETCH = 4
USE_AMP = True
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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

def print_hardware():
    print("\n" + "="*75)
    print("HARDWARE")
    print("="*75)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA visible:", torch.cuda.is_available())
    print("CPU logical cores:", os.cpu_count())
    print("Selected device:", DEVICE)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name} | {p.total_memory/1024**3:.2f} GB | CC {p.major}.{p.minor}")
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i} free at start: {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")

WINDOW = torch.hann_window(N_FFT)

# ---------------- AUDIO ----------------
def load_audio(path):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if fs != SAMPLE_RATE:
        dur = len(x) / fs
        n = int(dur * SAMPLE_RATE)
        x = np.interp(
            np.linspace(0,dur,n,endpoint=False),
            np.linspace(0,dur,len(x),endpoint=False),
            x
        ).astype(np.float32)
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
    X = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP,
        window=WINDOW.to(x.device), return_complex=True,
        center=True
    )
    return torch.log1p(torch.abs(X)), X

# ---------------- DATASET ----------------
class MusicDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
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

    def __len__(self):
        return len(self.paths)*CHUNKS_PER_SONG

    def __getitem__(self, i):
        p = self.paths[i % len(self.paths)]
        x = self.audio[p]
        if len(x) <= self.n:
            return F.pad(x,(0,self.n-len(x)))
        s = random.randint(0,len(x)-self.n)
        return x[s:s+self.n]

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
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(2*C,2*C), nn.GELU(),
            nn.Linear(2*C,LATENT_DIM)
        )

    def forward(self,x):
        return F.normalize(self.proj(self.net(x)),dim=-1)

class SlotSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        C=CHANNELS
        self.backbone=nn.Sequential(
            nn.Conv2d(1,C,5,padding=2),nn.BatchNorm2d(C),nn.GELU(),
            nn.Conv2d(C,C,3,padding=1),nn.BatchNorm2d(C),nn.GELU(),
            nn.Conv2d(C,C,3,stride=2,padding=1),nn.BatchNorm2d(C),nn.GELU(),
            nn.Conv2d(C,C,3,padding=1),nn.GELU()
        )
        self.mask=nn.Conv2d(C,SOURCE_SLOTS,1)
        self.proj=nn.Sequential(
            nn.Linear(C,C),nn.GELU(),nn.Linear(C,LATENT_DIM)
        )
        self.activity=nn.Linear(C,1)

    def forward(self,x):
        h=self.backbone(x)
        m_logits = self.mask(h)
        m = F.interpolate(m_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # FIX: independent sigmoid per slot instead of softmax across slots.
        # Softmax forces sum_k m_k == 1 everywhere, which made the
        # reconstruction loss a tautology (see module docstring). Sigmoid
        # masks are unconstrained in their sum, so reconstruction becomes a
        # real, learnable non-negative decomposition of the mixture.
        m = torch.sigmoid(m)

        # Pool the lower-resolution feature map using corresponding masks.
        mw=F.interpolate(m,size=h.shape[-2:],mode="bilinear",align_corners=False)
        slots=[]; acts=[]
        for k in range(SOURCE_SLOTS):
            w=mw[:,k:k+1]
            pooled=(h*w).sum((2,3))/(w.sum((2,3))+1e-6)
            slots.append(F.normalize(self.proj(pooled),dim=-1))
            acts.append(torch.sigmoid(self.activity(pooled)))
        return m,torch.stack(slots,1),torch.cat(acts,1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AcousticEncoder()
        self.slots=SlotSeparator()
    def forward(self,x):
        z=self.encoder(x)
        m,s,a=self.slots(x)
        return z,m,s,a

# ---------------- LOSSES ----------------
def info_nce(a,b,temp):
    """
    InfoNCE must run completely in float32.

    IMPORTANT:
    This function is called inside the AMP autocast context. Merely doing
    .float() on z is not sufficient because matrix multiplication can still
    be autocast back to float16. float16 cannot represent -1e9, which caused
    the previous crash.
    """
    n = a.shape[0]

    # Explicitly disable AMP for the whole contrastive calculation.
    with torch.autocast(
        device_type="cuda",
        enabled=False
    ):
        z = torch.cat([a, b], dim=0).float()
        z = F.normalize(z, dim=-1)

        sim = (z @ z.T) / float(temp)

        eye = torch.eye(
            2 * n,
            device=z.device,
            dtype=torch.bool
        )

        # -inf is preferable here and is representable in float32.
        sim = sim.masked_fill(
            eye,
            float("-inf")
        )

        target = torch.cat([
            torch.arange(
                n,
                device=z.device
            ) + n,
            torch.arange(
                n,
                device=z.device
            )
        ])

        return F.cross_entropy(
            sim,
            target
        )

def slot_match(a,b):
    # Keep representation-similarity calculations in float32 under AMP.
    with torch.autocast(
        device_type="cuda",
        enabled=False
    ):
        a = a.float()
        b = b.float()
        sim = torch.bmm(
            a,
            b.transpose(1,2)
        )
        return 1.0 - 0.5 * (
            sim.max(2).values.mean()
            +
            sim.max(1).values.mean()
        )

def orth_loss(s):
    sim=torch.bmm(s,s.transpose(1,2))
    k=s.shape[1]
    eye=torch.eye(k,device=s.device)[None]
    return (sim*(1-eye)).pow(2).mean()

def sparsity_loss(m):
    # FIX: replaces entropy_loss. Entropy assumed m was a per-pixel
    # probability simplex over slots (true only under softmax). With
    # independent sigmoid masks, the correct analogue of "prefer
    # concentrated, confident assignments" is an L1 penalty pushing masks
    # toward 0/1 rather than staying near 0.5 everywhere.
    return m.mean()

def recon_loss(mag, m):
    # mag: [B, F, T]
    # m:   [B, K, F, T]
    # pred: [B, F, T]
    # NOTE: with sigmoid masks this is now a genuine constraint: the
    # network must learn masks whose sum reconstructs the mixture, rather
    # than trivially satisfying it via softmax normalization.
    pred = (m * mag.unsqueeze(1)).sum(dim=1)
    return F.l1_loss(pred, mag)

# ---------------- TRAIN ----------------
def train(model,loader):
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    amp=USE_AMP and CUDA
    try:
        scaler=torch.amp.GradScaler("cuda",enabled=amp)
    except Exception:
        scaler=torch.cuda.amp.GradScaler(enabled=amp)

    hist=[]
    print("\n" + "="*75)
    print("ACOUSTIC LATENT TRAINING")
    print("="*75)
    print("Epochs:",EPOCHS)
    print("Iterations/epoch:",len(loader))
    print("Total iterations:",len(loader)*EPOCHS)
    print("Batch:",BATCH_SIZE," LR:",LR," Temperature:",TEMPERATURE)
    print("AMP:",amp)
    print("InfoNCE precision: float32 (AMP disabled inside contrastive loss)")
    print("Slot-match precision: float32")
    print("Reconstruction: magnitude [B,F,T] <- sigmoid masks [B,K,F,T]")

    for ep in range(1,EPOCHS+1):
        model.train()
        sums=np.zeros(6,dtype=float)
        t0=time.time()

        for it,audio in enumerate(loader,1):
            audio=audio.to(DEVICE,non_blocking=True)
            a=torch.stack([augment(x) for x in audio])
            b=torch.stack([augment(x) for x in audio])

            # FIX: when not using AMP on a CUDA device, the previous code
            # built a "cpu" autocast context while running CUDA tensors
            # through it. It happened to be harmless because enabled=False,
            # but it's semantically wrong and confusing. Use the correct
            # device type either way.
            device_type = "cuda" if CUDA else "cpu"
            ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp)
            with ctx:
                ma,_=stft_mag(a)
                mb,_=stft_mag(b)
                za,msa,sa,aa=model(ma[:,None])
                zb,msb,sb,ab=model(mb[:,None])

                lg=info_nce(za,zb,TEMPERATURE)
                ls=slot_match(sa,sb)
                lr = 0.5 * (
                    recon_loss(ma, msa)
                    + recon_loss(mb, msb)
                )
                lo=.5*(orth_loss(sa)+orth_loss(sb))
                le=.5*(sparsity_loss(msa)+sparsity_loss(msb))
                la=.5*(aa.mean()+ab.mean())
                loss = (
                    L_GLOBAL * lg
                    + L_SLOT * ls
                    + L_RECON * lr
                    + L_ORTH * lo
                    + L_SPARSITY * le
                    + L_ACTIVITY * la
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch={ep}, iteration={it}: "
                    f"loss={loss.item()}, global={lg.item()}, slot={ls.item()}, "
                    f"recon={lr.item()}, orth={lo.item()}, sparsity={le.item()}, "
                    f"activity={la.item()}"
                )

            opt.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                opt.step()

            # FIX: capture every scalar we still need as plain Python floats
            # BEFORE deleting the tensors. The original code deleted `loss`
            # (and friends) and THEN referenced loss.item() in the logging
            # block below, which raised:
            #   UnboundLocalError: cannot access local variable 'loss'
            loss_val = loss.item()
            lg_val, ls_val, lr_val = lg.item(), ls.item(), lr.item()
            gn_val = float(gn)

            vals=[loss_val,lg_val,ls_val,lr_val,lo.item(),le.item()]
            sums+=vals

            # Explicitly release the computation graph before the next iteration.
            del za,zb,msa,msb,sa,sb,aa,ab,ma,mb,a,b,loss

            if it==1 or it%10==0:
                gpu=""
                if CUDA:
                    gpu=f" | VRAM {torch.cuda.memory_allocated()/1024**3:.2f}/{torch.cuda.memory_reserved()/1024**3:.2f} GB"
                print(f"Epoch {ep:03d}/{EPOCHS} | Iter {it:04d}/{len(loader)} | Loss {loss_val:.4f} | Global {lg_val:.4f} | Slot {ls_val:.4f} | Recon {lr_val:.4f} | Grad {gn_val:.3f}{gpu}")

        sched.step()
        if CUDA:
            peak=torch.cuda.max_memory_allocated()/1024**3
            print(f"Peak GPU memory this epoch: {peak:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        r=sums/len(loader)
        row=dict(epoch=ep,loss=r[0],global_contrastive=r[1],slot_contrastive=r[2],
                 reconstruction=r[3],orthogonal=r[4],sparsity=r[5],
                 learning_rate=opt.param_groups[0]["lr"],epoch_time_sec=time.time()-t0)
        hist.append(row)
        print(f">>> EPOCH {ep:03d} | Loss={r[0]:.4f} | Global={r[1]:.4f} | Slot={r[2]:.4f} | Recon={r[3]:.4f} | Time={row['epoch_time_sec']:.1f}s")
        if ep%10==0 or ep==EPOCHS:
            d=RESULT_DIR/"checkpoints"; d.mkdir(parents=True,exist_ok=True)
            torch.save({"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict()},
                       d/f"checkpoint_{ep:03d}.pt")
    return hist

# ---------------- EMBEDDINGS + CLUSTERING ----------------
@torch.no_grad()
def extract(model,paths):
    model.eval()
    ge=[]; se=[]; meta=[]
    n=CHUNK_SECONDS*SAMPLE_RATE

    for si,p in enumerate(paths):
        x=load_audio(p)
        if len(x)<=n:
            starts=[0]
        else:
            count=min(20,max(1,len(x)//SAMPLE_RATE))
            starts=np.linspace(0,len(x)-n,count).astype(int)

        for ci,s in enumerate(starts):
            c=x[s:s+n]
            if len(c)<n: c=F.pad(c,(0,n-len(c)))
            c=c.to(DEVICE)
            mag,_=stft_mag(c[None])
            z,m,slots,act=model(mag[:,None])
            ge.append(z[0].cpu().numpy())
            se.append(slots[0].cpu().numpy())
            meta.append((p.name,ci,s/SAMPLE_RATE))
    return np.asarray(ge,np.float32),np.asarray(se,np.float32),meta

def analyze_embeddings(e,meta):
    out=RESULT_DIR/"embeddings"; out.mkdir(parents=True,exist_ok=True)
    np.save(out/"chunk_embeddings.npy",e)

    with open(out/"metadata.csv","w",newline="",encoding="utf8") as f:
        w=csv.writer(f); w.writerow(["song","chunk","start_seconds"])
        w.writerows(meta)

    pca=PCA(n_components=2)
    xy=pca.fit_transform(e)
    print(f"PCA variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

    best=None
    for k in range(2,min(9,len(e))):
        km=KMeans(n_clusters=k,n_init=10,random_state=SEED)
        labels=km.fit_predict(e)
        score=silhouette_score(e,labels)
        print(f"k={k:02d} | silhouette={score:.4f}")
        if best is None or score>best[0]:
            best=(score,k,labels)

    score,k,labels=best
    print(f"\nAnalysis cluster count: {k} (NOT an imposed instrument count)")
    with open(RESULT_DIR/"cluster_assignments.csv","w",newline="",encoding="utf8") as f:
        w=csv.writer(f); w.writerow(["song","chunk","start_seconds","cluster"])
        for m,l in zip(meta,labels):
            w.writerow([*m,int(l)])

    plt.figure(figsize=(10,7))
    plt.scatter(xy[:,0],xy[:,1],c=labels,s=20,alpha=.75)
    plt.xlabel("Latent dimension 1"); plt.ylabel("Latent dimension 2")
    plt.title("Learned Acoustic Latent Space"); plt.grid(True); plt.tight_layout()
    plt.savefig(RESULT_DIR/"latent_pca.png",dpi=160); plt.close()

# ---------------- MAIN ----------------
def main():
    print_hardware()
    paths=sorted(p for p in MUSIC_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not paths:
        print("No music files found:",MUSIC_DIR.resolve()); return
    print(f"\nFound {len(paths)} music files.")

    ds=MusicDataset(paths)
    workers=min(CPU_WORKERS,max(1,len(ds)))
    kw=dict(dataset=ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=workers,
            pin_memory=CUDA)
    if workers:
        kw["persistent_workers"]=True
        kw["prefetch_factor"]=PREFETCH
    loader=DataLoader(**kw)

    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model=Model().to(DEVICE)
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nMODEL PARAMETERS")
    print("Total:",f"{total:,}"," Trainable:",f"{trainable:,}")
    print("Latent dimension:",LATENT_DIM," Source slots:",SOURCE_SLOTS)
    print(model)

    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    with open(RESULT_DIR/"config.json","w") as f:
        json.dump(dict(sample_rate=SAMPLE_RATE,n_fft=N_FFT,hop=HOP,
                       chunk_seconds=CHUNK_SECONDS,source_slots=SOURCE_SLOTS,
                       latent_dim=LATENT_DIM,epochs=EPOCHS,batch_size=BATCH_SIZE,
                       learning_rate=LR,temperature=TEMPERATURE,device=str(DEVICE),
                       cpu_workers=workers),f,indent=2)

    hist=train(model,loader)

    with open(RESULT_DIR/"training_log.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)

    e=[r["epoch"] for r in hist]
    plt.figure(figsize=(10,6))
    for key,label in [("loss","Total"),("global_contrastive","Global contrastive"),
                      ("slot_contrastive","Slot contrastive"),("reconstruction","Reconstruction")]:
        plt.plot(e,[r[key] for r in hist],label=label)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Acoustic Latent-Space Training")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(RESULT_DIR/"loss_curve.png",dpi=150); plt.close()

    torch.save(model.state_dict(),RESULT_DIR/"acoustic_encoder.pt")

    ge,se,meta=extract(model,paths)
    np.save(RESULT_DIR/"embeddings"/"slot_embeddings.npy",se)

    analyze_embeddings(ge,meta)

    with open(RESULT_DIR/"model_parameters.txt","w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {total:,}\nTrainable: {trainable:,}\n")

    print("\n" + "="*75)
    print("V3 COMPLETE")
    print("="*75)
    print("Inspect:")
    print("  results_latent/loss_curve.png")
    print("  results_latent/latent_pca.png")
    print("  results_latent/cluster_assignments.csv")
    print("  results_latent/embeddings/")

if __name__=="__main__":
    main()
