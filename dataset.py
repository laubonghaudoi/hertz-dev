import os
from pathlib import Path
import torch as T
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000):
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        
        # Get all audio files
        self.files = []
        for ext in ['*.wav', '*.mp3']:
            self.files.extend(list(self.data_path.rglob(ext)))
            
        print(f"Found {len(self.files)} audio files")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Random crop to 6 seconds (matches blog post description)
        target_length = 6 * self.sample_rate
        if waveform.shape[1] > target_length:
            start_idx = T.randint(0, waveform.shape[1] - target_length, (1,))
            waveform = waveform[:, start_idx:start_idx + target_length]
        else:
            # Pad if too short
            padding = target_length - waveform.shape[1]
            waveform = T.nn.functional.pad(waveform, (0, padding))
            
        return waveform
