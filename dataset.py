# dataset.py
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, welch
from config import *

class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.segment_duration = SEGMENT_DURATION
        self.overlap = OVERLAP
        self.low_freq = FMIN
        self.high_freq = FMAX

        # Mel-spectrogram transform (if needed)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX
        )

    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                waveform = T.Resample(sr, self.sample_rate)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def bandpass_filter(self, waveform):
        """Apply bandpass filter to isolate heart sound frequencies"""
        try:
            sos = butter(4, [self.low_freq, self.high_freq], btype='band', fs=self.sample_rate, output='sos')
            filtered = sosfilt(sos, waveform.squeeze().numpy())
            return torch.from_numpy(filtered).unsqueeze(0).float()
        except Exception as e:
            print(f"Filter error: {e}")
            return waveform

    def standardize_segment(self, segment):
        """Normalize segment to zero mean and unit variance"""
        mean = segment.mean()
        std = segment.std()
        return (segment - mean) / (std + 1e-8)

    def segment_audio(self, waveform):
        """Split audio into overlapping segments"""
        segment_len = int(self.sample_rate * self.segment_duration)
        stride = int(segment_len * (1 - self.overlap))
        segments = []
        total_len = waveform.shape[1]
        
        # If audio is shorter than segment length, pad it
        if total_len < segment_len:
            padding = segment_len - total_len
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
            total_len = waveform.shape[1]
        
        for start in range(0, total_len - segment_len + 1, stride):
            segment = waveform[:, start:start+segment_len]
            segment = self.standardize_segment(segment)
            segments.append(segment)
            
        # Ensure at least one segment
        if not segments:
            segment = waveform[:, :segment_len] if total_len >= segment_len else waveform
            segment = self.standardize_segment(segment)
            segments.append(segment)
            
        return segments

    def preprocess_audio(self, waveform):
        """Complete preprocessing pipeline"""
        waveform = self.bandpass_filter(waveform)
        return waveform

    def augment_audio(self, waveform):
        """Apply data augmentation techniques"""
        if random.random() > AUGMENT_PROB:
            return waveform
            
        augmented = waveform.clone()
        
        # Add noise
        if random.random() < 0.5:
            snr_db = random.uniform(*NOISE_SNR_RANGE)
            signal_power = torch.mean(augmented ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(augmented) * torch.sqrt(noise_power)
            augmented = augmented + noise
            
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(*GAIN_RANGE)
            augmented = augmented * gain
            
        # Time shift
        if random.random() < 0.3:
            shift_samples = int(random.uniform(*SHIFT_RANGE) * augmented.shape[1])
            if shift_samples != 0:
                augmented = torch.roll(augmented, shift_samples, dims=1)
                
        return augmented

    def extract_mel_spectrogram(self, waveform):
        """Extract mel-spectrogram features"""
        mel = self.mel_spectrogram(waveform)
        return torch.log(mel + 1e-8)

    def plot_audio_analysis(self, waveform, title="Audio Analysis"):
        """Visualize audio analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Waveform
        axes[0, 0].plot(waveform.squeeze().numpy())
        axes[0, 0].set_title("Waveform")
        axes[0, 0].set_xlabel("Sample")
        axes[0, 0].set_ylabel("Amplitude")

        # Mel Spectrogram
        mel_spec = self.extract_mel_spectrogram(waveform)
        im = axes[0, 1].imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title("Mel Spectrogram")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Mel Frequency")
        plt.colorbar(im, ax=axes[0, 1])

        # Power Spectral Density
        f, psd = welch(waveform.squeeze().numpy(), fs=self.sample_rate)
        axes[1, 0].semilogy(f, psd)
        axes[1, 0].set_title("Power Spectral Density")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("PSD")

        # Spectrogram
        f, t, Sxx = plt.specgram(waveform.squeeze().numpy(), Fs=self.sample_rate, ax=axes[1, 1])
        axes[1, 1].set_title("Spectrogram")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Frequency (Hz)")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

class HeartSoundDataset(Dataset):
    def __init__(self, data_list, processor, augment=False, use_spectrogram=False):
        self.processor = processor
        self.augment = augment
        self.use_spectrogram = use_spectrogram
        self.segments = []
        self.labels = []
        self.file_indices = []

        print(f"📦 Processing {len(data_list)} files...")

        for file_idx, (path, label) in enumerate(data_list):
            waveform = self.processor.load_audio(path)
            if waveform is None:
                continue

            waveform = self.processor.preprocess_audio(waveform)
            segments = self.processor.segment_audio(waveform)
            
            for seg in segments:
                self.segments.append(seg)
                self.labels.append(0 if label == 'Normal' else 1)
                self.file_indices.append(file_idx)

        print(f"✅ Created {len(self.segments)} segments from {len(data_list)} files")
        normal = sum(1 for l in self.labels if l == 0)
        abnormal = len(self.labels) - normal
        print(f"📊 Class distribution: Normal={normal}, Abnormal={abnormal}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].clone()
        label = self.labels[idx]

        if self.augment:
            segment = self.processor.augment_audio(segment)

        if self.use_spectrogram:
            segment = self.processor.extract_mel_spectrogram(segment)

        return segment, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        counts = torch.bincount(torch.tensor(self.labels))
        total = len(self.labels)
        weights = total / (len(counts) * counts.float())
        return weights

    def get_file_level_labels(self):
        """Get majority vote labels for each file"""
        file_labels = {}
        for idx, (file_idx, label) in enumerate(zip(self.file_indices, self.labels)):
            file_labels.setdefault(file_idx, []).append(label)
        return {k: max(set(v), key=v.count) for k, v in file_labels.items()}

def create_weighted_sampler(dataset):
    """Create weighted random sampler for balanced training"""
    weights = dataset.get_class_weights()
    sample_weights = [weights[l] for l in dataset.labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )