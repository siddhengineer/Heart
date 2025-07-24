import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from model import CTENNClassifier
from dataset import AudioProcessor

# Paths
VALIDATION_PATH = r"C:\Users\MAXIMUS8\Desktop\Heart\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\validation"
REFERENCE_CSV = os.path.join(VALIDATION_PATH, "REFERENCE.csv")
BEST_MODEL_PATH = r"C:\Users\MAXIMUS8\Desktop\Heart\outputs\models\best_model_fold_5.pth"

# Step 1: Load Validation Data
def load_validation_data(validation_path, reference_csv):
    # Load the reference CSV
    df = pd.read_csv(reference_csv, header=None, names=["file_name", "label"])
    
    # Map labels: 1 -> Abnormal, -1 -> Normal
    label_mapping = {1: 1, -1: 0}  # 1 = Abnormal, 0 = Normal
    df["label"] = df["label"].map(label_mapping)
    
    # Create a list of (file_path, label) tuples
    data_list = [
        (os.path.join(validation_path, f"{row['file_name']}.wav"), row["label"])
        for _, row in df.iterrows()
    ]
    return data_list

# Step 2: Create Validation Dataset Class
class HeartSoundValidationDataset(Dataset):
    def __init__(self, data_list, audio_processor):
        """
        Args:
            data_list: List of (file_path, label) tuples.
            audio_processor: Instance of AudioProcessor for preprocessing.
        """
        self.data_list = data_list
        self.audio_processor = audio_processor
        self.target_length = int(audio_processor.segment_duration * audio_processor.sample_rate)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        
        # Load audio file
        waveform = self.audio_processor.load_audio(file_path)
        if waveform is None:
            raise ValueError(f"Error loading audio file: {file_path}")
        
        # Preprocess audio
        waveform = self.audio_processor.preprocess_audio(waveform)
        
        # Pad or truncate waveform to target length
        waveform = self.pad_or_truncate(waveform, self.target_length)
        
        # Ensure correct shape for Conv1D: [1, signal_length]
        waveform = waveform.squeeze(0)  # Remove extra dimension if present
        
        return waveform.unsqueeze(0), label  # Add channel dimension: [1, signal_length]

    def pad_or_truncate(self, waveform, target_length):
        """Pad or truncate the waveform to the target length."""
        length = waveform.size(-1)
        if length < target_length:
            # Pad with zeros
            padding = target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > target_length:
            # Truncate
            waveform = waveform[:, :target_length]
        return waveform

# Step 3: Evaluate the Model
def evaluate_model(model, validation_data, audio_processor, batch_size=32):
    # Create validation dataset and dataloader
    validation_dataset = HeartSoundValidationDataset(validation_data, audio_processor)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(model.device)  # Move to device
            labels = labels.to(model.device)
            
            # Forward pass
            outputs = model(waveforms)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 (Abnormal)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average="binary")
    auc = roc_auc_score(all_targets, all_probabilities)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return all_targets, all_predictions, accuracy, precision, recall, f1, auc

# Step 4: Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names=["Normal", "Abnormal"]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Main Testing Script
if __name__ == "__main__":
    # Load validation data
    validation_data = load_validation_data(VALIDATION_PATH, REFERENCE_CSV)
    
    # Initialize audio processor
    audio_processor = AudioProcessor()  # Replace with your actual AudioProcessor instance
    
    # Load the trained model
    model = CTENNClassifier()
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)
    
    # Evaluate the model
    all_targets, all_predictions, accuracy, precision, recall, f1, auc = evaluate_model(
        model, validation_data, audio_processor
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(all_targets, all_predictions)