#data set loader
import torch
import torchaudio
import os
import soundfile as sf
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, mix_dir, src1_dir, src2_dir, n_fft=1024, hop=512):
        self.mix_files = sorted(os.listdir(mix_dir))
        self.mix_dir = mix_dir
        self.src1_dir = src1_dir
        self.src2_dir = src2_dir
        
        self.n_fft = n_fft
        self.hop = hop
        self.window = torch.hann_window(n_fft)

    def __len__(self):
        return len(self.mix_files)

    def load_audio(self, path):
        x, fs = sf.read(path)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return torch.tensor(x, dtype=torch.float32)

    def __getitem__(self, idx):
        file = self.mix_files[idx]

        mix = self.load_audio(os.path.join(self.mix_dir, file))
        s1  = self.load_audio(os.path.join(self.src1_dir, file))
        s2  = self.load_audio(os.path.join(self.src2_dir, file))

        min_len = min(len(mix), len(s1), len(s2))
        mix = mix[:min_len]
        s1  = s1[:min_len]
        s2  = s2[:min_len]

        X_mix = torch.stft(mix, self.n_fft, self.hop,
                           window=self.window,
                           return_complex=True)

        X1 = torch.stft(s1, self.n_fft, self.hop,
                        window=self.window,
                        return_complex=True)

        X2 = torch.stft(s2, self.n_fft, self.hop,
                        window=self.window,
                        return_complex=True)

        P1 = torch.abs(X1)**2
        P2 = torch.abs(X2)**2

        M1 = P1 / (P1 + P2 + 1e-8)

        X_mag = torch.log(torch.abs(X_mix) + 1e-8)

        return X_mag.unsqueeze(0), M1.unsqueeze(0)