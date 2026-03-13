"""
PhishLens - HTML Dataset Module for ELECTRA Training
Handles data loading, tokenization, and batching.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLDataset(Dataset):
    """
    PyTorch Dataset for HTML phishing detection with ELECTRA.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: ElectraTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of processed HTML text
            labels: List of labels (0=legitimate, 1=phishing)
            tokenizer: ELECTRA tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
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
    """
    Data module for managing HTML dataset splits and loaders.
    """
    
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
        """
        Initialize the data module.
        
        Args:
            data_path: Path to processed HTML CSV
            model_name: ELECTRA model name for tokenizer
            max_length: Maximum sequence length
            batch_size: Batch size for DataLoaders
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            num_workers: DataLoader workers (0 for Windows)
        """
        self.data_path = data_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        
        # Load and prepare data
        self._load_data()
    
    def _load_data(self):
        """Load and split the data."""
        logger.info(f"Loading data from: {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        logger.info(f"Total samples: {len(df)}")
        
        # Get texts and labels
        texts = df['processed_text'].fillna('').tolist()
        labels = df['label'].tolist()
        
        # Log class distribution
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
        
        # Store splits
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
        """Get training dataset."""
        return HTMLDataset(
            self.train_texts,
            self.train_labels,
            self.tokenizer,
            self.max_length
        )
    
    def get_val_dataset(self) -> HTMLDataset:
        """Get validation dataset."""
        return HTMLDataset(
            self.val_texts,
            self.val_labels,
            self.tokenizer,
            self.max_length
        )
    
    def get_test_dataset(self) -> HTMLDataset:
        """Get test dataset."""
        return HTMLDataset(
            self.test_texts,
            self.test_labels,
            self.tokenizer,
            self.max_length
        )
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.get_val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.get_test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.
        Uses inverse frequency weighting.
        """
        labels = np.array(self.train_labels)
        class_counts = np.bincount(labels)
        total = len(labels)
        
        # Inverse frequency
        weights = total / (len(class_counts) * class_counts)
        
        logger.info(f"Class weights - Legitimate: {weights[0]:.4f}, Phishing: {weights[1]:.4f}")
        
        return torch.FloatTensor(weights)


def main():
    """Test the data module."""
    
    # Paths
    data_path = r"C:\PhishLens\data\processed\html\processed_html.csv"
    
    print("=" * 60)
    print("PhishLens HTML Dataset Module Test")
    print("=" * 60)
    
    # Initialize data module
    data_module = HTMLDataModule(
        data_path=data_path,
        model_name="google/electra-base-discriminator",
        max_length=512,
        batch_size=8,
        test_size=0.1,
        val_size=0.1,
        random_state=42
    )
    
    # Test dataloaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    test_loader = data_module.get_test_dataloader()
    
    print(f"\nDataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test a batch
    print("\nTesting a batch...")
    batch = next(iter(train_loader))
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  labels: {batch['labels'].tolist()}")
    
    # Get class weights
    weights = data_module.get_class_weights()
    print(f"\nClass weights: {weights}")
    
    print("\n" + "=" * 60)
    print("Dataset module test complete! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()