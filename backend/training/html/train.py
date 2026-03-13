"""
PhishLens - ELECTRA Training Script for HTML Phishing Detection
Trains ELECTRA-base model on preprocessed HTML data.

Features:
- Class weights for imbalanced data (46/54 split)
- Gradient checkpointing for memory efficiency
- Early stopping
- Comprehensive evaluation metrics
- Model checkpointing
"""

import os
import sys
import gc
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path FIRST (before other imports)
sys.path.insert(0, r"C:\PhishLens")

from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import logging

# Setup logging
log_dir = r"C:\PhishLens\logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "html_training.log"))
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# DATASET CLASSES (Embedded to avoid import issues)
# ============================================================

class HTMLDataset(Dataset):
    """PyTorch Dataset for HTML phishing detection with ELECTRA."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: ElectraTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class HTMLDataModule:
    """Data module for managing HTML dataset splits and loaders."""
    
    def __init__(
        self,
        data_path: str,
        model_name: str = "google/electra-base-discriminator",
        max_length: int = 512,
        batch_size: int = 8,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        num_workers: int = 0
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        
        self._load_data()
    
    def _load_data(self):
        """Load and split the data."""
        logger.info(f"Loading data from: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Total samples: {len(df)}")
        
        texts = df['processed_text'].fillna('').tolist()
        labels = df['label'].tolist()
        
        phishing_count = sum(labels)
        legitimate_count = len(labels) - phishing_count
        logger.info(f"Class distribution - Phishing: {phishing_count}, Legitimate: {legitimate_count}")
        
        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_ratio = self.val_size / (1 - self.test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val_labels
        )
        
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        
        logger.info(f"Train samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        logger.info(f"Test samples: {len(test_texts)}")
    
    def get_train_dataset(self) -> HTMLDataset:
        return HTMLDataset(self.train_texts, self.train_labels, self.tokenizer, self.max_length)
    
    def get_val_dataset(self) -> HTMLDataset:
        return HTMLDataset(self.val_texts, self.val_labels, self.tokenizer, self.max_length)
    
    def get_test_dataset(self) -> HTMLDataset:
        return HTMLDataset(self.test_texts, self.test_labels, self.tokenizer, self.max_length)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = np.array(self.train_labels)
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        logger.info(f"Class weights - Legitimate: {weights[0]:.4f}, Phishing: {weights[1]:.4f}")
        return torch.FloatTensor(weights)


# ============================================================
# METRICS FUNCTION
# ============================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================
# WEIGHTED TRAINER (Handles Class Imbalance)
# ============================================================

class WeightedTrainer(Trainer):
    """Custom Trainer that uses class weights for imbalanced data."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted cross-entropy loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


# ============================================================
# MAIN TRAINER CLASS
# ============================================================

class PhishLensHTMLTrainer:
    """Trainer class for ELECTRA-based HTML phishing detection."""
    
    def __init__(
        self,
        model_name: str = "google/electra-base-discriminator",
        output_dir: str = r"C:\PhishLens\backend\models\html",
        max_length: int = 512,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 3
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        
        # Set device (CPU for RTX 5070 compatibility)
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set CPU threads
        torch.set_num_threads(8)
        logger.info(f"CPU threads: {torch.get_num_threads()}")
        
        self._init_model()
    
    def _init_model(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        
        self.model = ElectraForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
        
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def train(self, data_path: str):
        """Train the model."""
        logger.info("=" * 60)
        logger.info("Starting ELECTRA Training for HTML Phishing Detection")
        logger.info("=" * 60)
        
        # Load data
        logger.info(f"\nLoading data from: {data_path}")
        data_module = HTMLDataModule(
            data_path=data_path,
            model_name=self.model_name,
            max_length=self.max_length,
            batch_size=self.batch_size,
            test_size=0.1,
            val_size=0.1,
            random_state=42
        )
        
        # Get datasets
        train_dataset = data_module.get_train_dataset()
        val_dataset = data_module.get_val_dataset()
        test_dataset = data_module.get_test_dataset()
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Get class weights
        class_weights = data_module.get_class_weights()
        logger.info(f"\n*** Class Weights Applied ***")
        logger.info(f"  Legitimate (0): {class_weights[0]:.4f}")
        logger.info(f"  Phishing (1): {class_weights[1]:.4f}")
        logger.info(f"  This compensates for the 46/54 class imbalance!\n")
        
        # Calculate training steps
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = steps_per_epoch * self.num_epochs
        
        logger.info(f"Training configuration:")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "checkpoints"),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            
            # Optimization
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            
            # Logging
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            report_to="none",
            
            # Best model
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Memory optimization
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            
            # CPU settings
            use_cpu=True,
            fp16=False,
            bf16=False,
        )
        
        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.early_stopping_patience
        )
        
        # Initialize WEIGHTED trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Train
        logger.info("\n" + "=" * 60)
        logger.info("Starting training with CLASS WEIGHTS...")
        logger.info("=" * 60 + "\n")
        
        start_time = datetime.now()
        gc.collect()
        
        train_result = trainer.train()
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"\nTraining completed in: {elapsed_time}")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        logger.info(f"\nSaving final model to: {final_model_path}")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Evaluate on test set
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating on test set...")
        logger.info("=" * 60)
        
        test_results = trainer.evaluate(test_dataset)
        
        logger.info("\nTest Results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Detailed predictions
        logger.info("\nGenerating detailed predictions...")
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        logger.info("\nConfusion Matrix:")
        logger.info(f"                 Predicted")
        logger.info(f"                 Legit    Phish")
        logger.info(f"Actual Legit   {cm[0][0]:6d}  {cm[0][1]:6d}")
        logger.info(f"       Phish   {cm[1][0]:6d}  {cm[1][1]:6d}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(labels, preds, target_names=['Legitimate', 'Phishing']))
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) * 100
        false_negative_rate = fn / (fn + tp) * 100
        
        logger.info(f"\nDetailed Metrics:")
        logger.info(f"  True Positives (Phishing caught): {tp}")
        logger.info(f"  True Negatives (Legit passed): {tn}")
        logger.info(f"  False Positives (Legit flagged): {fp} ({false_positive_rate:.2f}%)")
        logger.info(f"  False Negatives (Phishing missed): {fn} ({false_negative_rate:.2f}%)")
        
        # Save training info
        training_info = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'effective_batch_size': effective_batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'class_weights': class_weights.tolist(),
            'training_time': str(elapsed_time),
            'test_results': {k: float(v) for k, v in test_results.items()},
            'confusion_matrix': cm.tolist(),
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = os.path.join(final_model_path, "training_info.json")
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"\nTraining info saved to: {info_path}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {final_model_path}")
        logger.info(f"\n📊 Final Test Metrics:")
        logger.info(f"  Accuracy:  {test_results['eval_accuracy']*100:.2f}%")
        logger.info(f"  F1 Score:  {test_results['eval_f1']*100:.2f}%")
        logger.info(f"  Precision: {test_results['eval_precision']*100:.2f}%")
        logger.info(f"  Recall:    {test_results['eval_recall']*100:.2f}%")
        logger.info(f"\n⚠️  Error Rates:")
        logger.info(f"  False Positive Rate: {false_positive_rate:.2f}%")
        logger.info(f"  False Negative Rate: {false_negative_rate:.2f}%")
        
        return test_results


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main training function."""
    
    data_path = r"C:\PhishLens\data\processed\html\processed_html.csv"
    output_dir = r"C:\PhishLens\backend\models\html"
    
    print("=" * 70)
    print("PhishLens ELECTRA Training for HTML Phishing Detection")
    print("=" * 70)
    print("\n📋 Configuration:")
    print("  Model: google/electra-base-discriminator")
    print("  Max Length: 512 tokens")
    print("  Batch Size: 4 (effective: 32 with accumulation)")
    print("  Learning Rate: 2e-5")
    print("  Epochs: 3")
    print("  Device: CPU (RTX 5070 compatibility)")
    print("\n⚖️  Class Imbalance Handling:")
    print("  Using WEIGHTED loss function to handle 46/54 split")
    print("  Phishing class gets higher weight to compensate")
    print("\n⏱️  Estimated Training Time: 15-25 hours")
    print("=" * 70)
    
    confirm = input("\nStart training? (yes/no): ").strip().lower()
    if confirm not in ('yes', 'y'):
        print("Training cancelled.")
        return
    
    trainer = PhishLensHTMLTrainer(
        model_name="google/electra-base-discriminator",
        output_dir=output_dir,
        max_length=512,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        early_stopping_patience=3
    )
    
    results = trainer.train(data_path)
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"📁 Model saved to: {output_dir}/final_model")
    print("=" * 70)


if __name__ == "__main__":
    main()