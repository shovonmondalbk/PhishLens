"""
PhishLens - URL Analyzer Training Script (CPU Version)
Trains DeBERTa v3 model for phishing URL detection.

Hardware: CPU Training (32GB RAM)
Model: microsoft/deberta-v3-base
Estimated Time: 10-15 hours
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from url.dataset import URLDataModule

# Setup logging
log_dir = r"C:\PhishLens\logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "url_training.log"))
    ]
)
logger = logging.getLogger(__name__)


class URLModelTrainer:
    """
    Trainer class for URL phishing detection model (CPU Version).
    
    Uses HuggingFace Trainer with:
    - CPU training (no GPU required)
    - Gradient accumulation for effective larger batch sizes
    - Early stopping to prevent overfitting
    - Best model checkpointing
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        output_dir: str = r"C:\PhishLens\backend\models\url",
        max_length: int = 256,
        batch_size: int = 8,              # Reduced for CPU
        gradient_accumulation_steps: int = 4,  # Effective batch = 32
        learning_rate: float = 2e-5,
        num_epochs: int = 3,              # Reduced for faster training
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 3,
        seed: int = 42
    ):
        """
        Initialize the trainer.
        """
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
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Force CPU
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"RAM: 32 GB")
        logger.info("NOTE: Training on CPU will take approximately 10-15 hours")
        
    def load_model(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model for binary classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "legitimate", 1: "phishing"},
            label2id={"legitimate": 0, "phishing": 1}
        )
        
        # Model stays on CPU (default)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='binary', pos_label=1),
            'recall': recall_score(labels, preds, average='binary', pos_label=1),
            'f1': f1_score(labels, preds, average='binary', pos_label=1)
        }
    
    def train(self, data_path: str):
        """Train the model."""
        logger.info("=" * 60)
        logger.info("Starting URL Analyzer Training (CPU Mode)")
        logger.info("=" * 60)
        
        # Load model
        self.load_model()
        
        # Load data
        logger.info(f"Loading data from: {data_path}")
        data_module = URLDataModule(
            data_path=data_path,
            model_name=self.model_name,
            max_length=self.max_length,
            batch_size=self.batch_size
        )
        
        # Get datasets
        train_dataset = data_module.get_train_dataset()
        val_dataset = data_module.get_val_dataset()
        test_dataset = data_module.get_test_dataset()
        
        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Validation size: {len(val_dataset)}")
        logger.info(f"Test size: {len(test_dataset)}")
        
        # Calculate training steps
        steps_per_epoch = len(train_dataset) // self.batch_size // self.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        
        # Estimate time
        estimated_hours = (total_steps * 2.5) / 3600  # ~2.5 sec per step on CPU
        logger.info(f"Estimated training time: {estimated_hours:.1f} hours")
        
        # Training arguments - CPU optimized
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            
            # Training hyperparameters
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            
            # Optimizer
            optim="adamw_torch",
            
            # CPU Training - No FP16
            fp16=False,
            use_cpu=True,
            
            # Evaluation strategy
            evaluation_strategy="steps",
            eval_steps=1000,  # Evaluate less frequently to save time
            
            # Saving strategy
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Logging
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            report_to="none",
            
            # Other
            seed=self.seed,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_threshold=0.001
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Train
        logger.info("Starting training... (This will take several hours)")
        logger.info("You can monitor progress in the console.")
        logger.info("Feel free to leave this running overnight.")
        start_time = datetime.now()
        
        train_result = trainer.train()
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in: {training_time}")
        
        # Save final model
        logger.info("Saving final model...")
        final_model_path = os.path.join(self.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        for key, value in test_results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Get predictions for detailed report
        logger.info("Generating detailed predictions...")
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # Classification report
        report = classification_report(labels, preds, target_names=['legitimate', 'phishing'])
        logger.info("\nClassification Report:")
        logger.info("\n" + report)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"                 Predicted")
        logger.info(f"                 Leg    Phi")
        logger.info(f"Actual Leg      {cm[0][0]:5d}  {cm[0][1]:5d}")
        logger.info(f"       Phi      {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "training_time": str(training_time),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "device": "CPU",
            "test_results": {k: float(v) if isinstance(v, float) else v for k, v in test_results.items()},
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        with open(os.path.join(self.output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"\nModel saved to: {final_model_path}")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        
        return test_results


def main():
    """Main function to train the URL analyzer."""
    
    # Paths
    base_dir = r"C:\PhishLens"
    data_path = os.path.join(base_dir, "data", "processed", "url", "processed_urls.csv")
    output_dir = os.path.join(base_dir, "backend", "models", "url")
    
    print("=" * 60)
    print("PhishLens URL Analyzer - CPU Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Model: microsoft/deberta-v3-base")
    print(f"  Batch Size: 8 (effective: 32 with accumulation)")
    print(f"  Epochs: 3")
    print(f"  Device: CPU")
    print(f"\nEstimated Time: 10-15 hours")
    print("You can leave this running overnight.\n")
    print("=" * 60)
    
    # Training configuration - Optimized for CPU
    trainer = URLModelTrainer(
        model_name="microsoft/deberta-v3-base",
        output_dir=output_dir,
        max_length=256,
        batch_size=8,                     # Smaller batch for CPU
        gradient_accumulation_steps=4,    # Effective batch size = 32
        learning_rate=2e-5,
        num_epochs=3,                     # Reduced epochs for CPU
        warmup_ratio=0.1,
        weight_decay=0.01,
        early_stopping_patience=3,
        seed=42
    )
    
    # Train
    results = trainer.train(data_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {results['eval_f1']:.4f}")
    print(f"Test Precision: {results['eval_precision']:.4f}")
    print(f"Test Recall: {results['eval_recall']:.4f}")
    print("=" * 60)
    print(f"\nModel saved to: {output_dir}\\final_model")
    print("You can now use this model for inference.")


if __name__ == "__main__":
    main()