# config.py
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio processing parameters (as per CTENN paper)
SAMPLE_RATE = 4000  # 4kHz as mentioned in paper
SEGMENT_DURATION = 5.0  # 5 seconds segments
OVERLAP = 0.5  # 50% overlap between segments

# Frequency filtering (based on heart sound characteristics)
FMIN = 25   # Minimum frequency (Hz)
FMAX = 1999 # Maximum frequency (Hz) - covers heart sound range

# Model architecture parameters (CTENN specifications)
D_MODEL = 512        # Transformer embedding dimension
NUM_HEADS = 8        # Number of attention heads
FF_DIM = 2048        # Feed-forward dimension (typically 4x d_model)
NUM_LAYERS = 6       # Number of transformer encoder layers
DROPOUT = 0.1        # Dropout rate
NUM_CLASSES = 2      # Binary classification (Normal vs Abnormal)

# Training parameters (based on paper methodology)
BATCH_SIZE = 32      # Batch size for training
LEARNING_RATE = 1e-4 # Initial learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization
NUM_EPOCHS = 100     # Maximum number of epochs
PATIENCE = 15        # Early stopping patience
WARMUP_EPOCHS = 10   # Learning rate warmup

# Cross-validation
K_FOLDS = 10         # 10-fold cross-validation as in paper

# Data augmentation parameters
AUGMENT_PROB = 0.5   # Probability of applying augmentation
NOISE_SNR_RANGE = (15, 30)  # SNR range for noise addition
GAIN_RANGE = (0.8, 1.2)     # Random gain range
SHIFT_RANGE = (-0.1, 0.1)   # Time shift range

# Mel-spectrogram parameters (if using spectrogram variant)
N_MELS = 128         # Number of mel bins
N_FFT = 1024         # FFT size
HOP_LENGTH = 512     # Hop length for STFT

# Dataset paths (update these according to your setup)
PHYSIONET_PATH = r"C:\Users\MAXIMUS8\Desktop\Heart\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
PHYSIONET_FOLDERS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
KAGGLE_PATH = r"C:\Users\MAXIMUS8\Desktop\Heart\Kaggle-PASCAL"

# Output directories
OUTPUT_DIR = "outputs"
MODEL_DIR = "outputs/models"
PLOT_DIR = "outputs/plots"
LOG_DIR = "outputs/logs"

# Model checkpoint parameters
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_loss'  # Metric to monitor for best model
MODE = 'min'  # 'min' for loss, 'max' for accuracy

# Reproducibility
SEED = 42

print(f"🔧 Configuration loaded - Device: {device}")
print(f"📊 Model: D_MODEL={D_MODEL}, HEADS={NUM_HEADS}, LAYERS={NUM_LAYERS}")
print(f"🎵 Audio: SR={SAMPLE_RATE}Hz, Duration={SEGMENT_DURATION}s")