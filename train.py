# train.py
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import logging
from datetime import datetime
import json

# Import your modules
try:
    from config import *
    from model import CTENNClassifier
    from dataset import HeartSoundDataset, AudioProcessor, create_weighted_sampler
    from data_loader import load_and_validate_datasets
    from utils import EarlyStopping, CosineAnnealingWithRestartsScheduler
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("- config.py")
    print("- model.py") 
    print("- dataset.py")
    print("- data_loader.py")
    print("- utils.py")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print(f"{device} : device")

class CTENNTrainer:
    def __init__(self):
        self.device = device
        self.setup_logging()
        self.setup_directories()
        self.set_seed()
        
        # Initialize processor
        self.processor = AudioProcessor()
        
        # Metrics storage
        self.fold_results = []
        self.training_history = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
            os.makedirs(directory, exist_ok=True)
            
    def set_seed(self):
        """Set random seeds for reproducibility"""
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def load_data(self):
        """Load and prepare datasets"""
        self.logger.info("🔄 Loading datasets...")
        
        # Check if pickle file exists
        pickle_path = os.path.join(OUTPUT_DIR, "combined_dataset.pkl")
        
        if os.path.exists(pickle_path):
            self.logger.info(f"📦 Loading existing dataset from {pickle_path}")
            with open(pickle_path, 'rb') as f:
                data_list = pickle.load(f)
        else:
            self.logger.info("🔍 Creating new dataset...")
            data_list = load_and_validate_datasets(
                PHYSIONET_PATH, 
                PHYSIONET_FOLDERS, 
                KAGGLE_PATH,
                save_path=pickle_path
            )
        
        self.logger.info(f"✅ Loaded {len(data_list)} audio files")
        return data_list
        
    def create_datasets(self, data_list, train_indices, val_indices):
        """Create training and validation datasets"""
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        
        # Create datasets
        train_dataset = HeartSoundDataset(
            train_data, 
            self.processor, 
            augment=True,
            use_spectrogram=False
        )
        
        val_dataset = HeartSoundDataset(
            val_data, 
            self.processor, 
            augment=False,
            use_spectrogram=False
        )
        
        return train_dataset, val_dataset
        
    def create_data_loaders(self, train_dataset, val_dataset):
        """Create data loaders with appropriate sampling"""
        # Create weighted sampler for balanced training
        train_sampler = create_weighted_sampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return train_loader, val_loader
        
    def create_model_and_optimizer(self, class_weights=None):
        """Create model, loss function, and optimizer"""
        model = CTENNClassifier().to(self.device)
        
        # Weighted CrossEntropy loss for class imbalance
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
            
        # AdamW optimizer as used in the paper
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with warm restarts
        scheduler = CosineAnnealingWithRestartsScheduler(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        return model, criterion, optimizer, scheduler
        
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
        return total_loss / len(train_loader), 100. * correct / total
        
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validating", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                
        # Calculate metrics
        accuracy = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary'
        )
        auc = roc_auc_score(all_targets, all_probabilities)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
    def train_fold(self, fold, train_dataset, val_dataset):
        """Train a single fold"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🔄 Training Fold {fold + 1}/{K_FOLDS}")
        self.logger.info(f"{'='*50}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset)
        
        # Get class weights
        class_weights = train_dataset.get_class_weights()
        self.logger.info(f"Class weights: {class_weights}")
        
        # Create model and optimizer
        model, criterion, optimizer, scheduler = self.create_model_and_optimizer(class_weights)
        
        # Early stopping
        checkpoint_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold+1}.pth")
        early_stopping = EarlyStopping(
            patience=PATIENCE,
            delta=0.001,
            checkpoint_path=checkpoint_path
        )
        
        # Training history for this fold
        fold_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [],
            'val_f1': [], 'val_auc': []
        }
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Store history
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_metrics['loss'])
            fold_history['val_acc'].append(val_metrics['accuracy'])
            fold_history['val_precision'].append(val_metrics['precision'])
            fold_history['val_recall'].append(val_metrics['recall'])
            fold_history['val_f1'].append(val_metrics['f1'])
            fold_history['val_auc'].append(val_metrics['auc'])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}"
            )
            
            # Early stopping check
            early_stopping(val_metrics['loss'], model)
            if early_stopping.early_stop:
                self.logger.info("🛑 Early stopping triggered")
                break
                
        # Load best model for final evaluation
        model.load_state_dict(torch.load(checkpoint_path))
        final_metrics = self.validate_epoch(model, val_loader, criterion)
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': len(fold_history['val_loss']) - early_stopping.counter,
            'metrics': final_metrics,
            'history': fold_history
        }
        
        self.fold_results.append(fold_result)
        self.training_history.append(fold_history)
        
        self.logger.info(f"✅ Fold {fold + 1} completed - Best Val Acc: {final_metrics['accuracy']:.2f}%")
        
        return fold_result
        
    def cross_validate(self, data_list):
        """Perform k-fold cross-validation"""
        self.logger.info(f"🔄 Starting {K_FOLDS}-fold cross-validation")
        
        # Prepare labels for stratified split
        labels = [1 if label == 'Abnormal' else 0 for _, label in data_list]
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        
        for fold, (train_indices, val_indices) in enumerate(skf.split(range(len(data_list)), labels)):
            # Create datasets
            train_dataset, val_dataset = self.create_datasets(data_list, train_indices, val_indices)
            
            # Train fold
            self.train_fold(fold, train_dataset, val_dataset)
            
        self.logger.info("✅ Cross-validation completed")
        
    def analyze_results(self):
        """Analyze and summarize cross-validation results"""
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 CROSS-VALIDATION RESULTS SUMMARY")
        self.logger.info("="*50)
        
        # Collect metrics
        accuracies = [result['metrics']['accuracy'] for result in self.fold_results]
        precisions = [result['metrics']['precision'] for result in self.fold_results]
        recalls = [result['metrics']['recall'] for result in self.fold_results]
        f1_scores = [result['metrics']['f1'] for result in self.fold_results]
        aucs = [result['metrics']['auc'] for result in self.fold_results]
        
        # Calculate statistics
        metrics_summary = {
            'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
            'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
            'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)},
            'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
            'auc': {'mean': np.mean(aucs), 'std': np.std(aucs)}
        }
        
        # Log summary
        for metric, stats in metrics_summary.items():
            self.logger.info(f"{metric.capitalize():10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
        # Save detailed results
        results_file = os.path.join(OUTPUT_DIR, "cross_validation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'summary': metrics_summary,
                'fold_results': [
                    {
                        'fold': result['fold'],
                        'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                  for k, v in result['metrics'].items() 
                                  if k not in ['predictions', 'targets', 'probabilities']}
                    }
                    for result in self.fold_results
                ]
            }, f, indent=2)
            
        self.logger.info(f"💾 Results saved to {results_file}")
        
        return metrics_summary
        
    def create_visualizations(self):
        """Create training and results visualizations"""
        self.logger.info("📈 Creating visualizations...")
        
        # 1. Training curves for all folds
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, history in enumerate(self.training_history):
            epochs = range(1, len(history['train_loss']) + 1)
            
            axes[0, 0].plot(epochs, history['train_loss'], alpha=0.3, color='blue')
            axes[0, 0].plot(epochs, history['val_loss'], alpha=0.3, color='red')
            
            axes[0, 1].plot(epochs, history['train_acc'], alpha=0.3, color='blue')
            axes[0, 1].plot(epochs, history['val_acc'], alpha=0.3, color='red')
            
            axes[1, 0].plot(epochs, history['val_f1'], alpha=0.3, color='green')
            axes[1, 1].plot(epochs, history['val_auc'], alpha=0.3, color='purple')
            
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend(['Train', 'Validation'])
        
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend(['Train', 'Validation'])
        
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        
        axes[1, 1].set_title('AUC')
        axes[1, 1].set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cross-validation results boxplot
        metrics_data = {
            'Accuracy': [result['metrics']['accuracy'] for result in self.fold_results],
            'Precision': [result['metrics']['precision'] for result in self.fold_results],
            'Recall': [result['metrics']['recall'] for result in self.fold_results],
            'F1-Score': [result['metrics']['f1'] for result in self.fold_results],
            'AUC': [result['metrics']['auc'] for result in self.fold_results]
        }
        
        plt.figure(figsize=(12, 6))
        positions = range(1, len(metrics_data) + 1)
        bp = plt.boxplot(metrics_data.values(), positions=positions, patch_artist=True)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.xticks(positions, metrics_data.keys())
        plt.ylabel('Score')
        plt.title('Cross-Validation Results Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, 'cv_results_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrices for all folds (adjust layout for 10 folds)
        n_folds = len(self.fold_results)
        n_cols = 5
        n_rows = (n_folds + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, result in enumerate(self.fold_results):
            cm = confusion_matrix(result['metrics']['targets'], result['metrics']['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'Fold {i+1}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
        # Hide unused subplots
        for i in range(len(self.fold_results), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("📊 Visualizations saved to plots directory")
        
    def run_training(self):
        """Main training pipeline"""
        self.logger.info("🚀 Starting CTENN Heart Sound Classification Training")
        self.logger.info(f"📱 Device: {self.device}")
        self.logger.info(f"🔧 Model: {D_MODEL}D, {NUM_HEADS} heads, {NUM_LAYERS} layers")
        
        try:
            # Load data
            data_list = self.load_data()
            
            # Run cross-validation
            self.cross_validate(data_list)
            
            # Analyze results
            metrics_summary = self.analyze_results()
            
            # Create visualizations
            self.create_visualizations()
            
            self.logger.info("🎉 Training completed successfully!")
            self.logger.info(f"📊 Final Results - Accuracy: {metrics_summary['accuracy']['mean']:.2f}% ± {metrics_summary['accuracy']['std']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise


def main():
    """Main function"""
    trainer = CTENNTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()