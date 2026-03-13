"""
PhishLens - URL Dataset for DeBERTa Training
PyTorch Dataset class for loading and batching URL data.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLDataset(Dataset):
    """
    PyTorch Dataset for URL phishing detection.
    
    This dataset handles:
    - Loading preprocessed URL data
    - Tokenizing text for DeBERTa
    - Returning tensors for training/evaluation
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of processed URL texts
            labels: List of labels (0=legitimate, 1=phishing)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
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
            'label': torch.tensor(label, dtype=torch.long)
        }


class URLDataModule:
    """
    Data module for managing URL datasets.
    
    Handles:
    - Loading data from CSV
    - Splitting into train/val/test
    - Creating DataLoaders
    """
    
    def __init__(
        self,
        data_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 256,
        batch_size: int = 16,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        num_workers: int = 0  # Set to 0 for Windows compatibility
    ):
        """
        Initialize the data module.
        
        Args:
            data_path: Path to processed CSV file
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
            batch_size: Batch size for DataLoaders
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_state: Random seed for reproducibility
            num_workers: Number of workers for DataLoader
        """
        self.data_path = data_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.num_workers = num_workers
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load and split data
        self._load_data()
        
    def _load_data(self):
        """Load data from CSV and split into train/val/test."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Extract texts and labels
        texts = df['processed_text'].tolist()
        labels = df['label'].tolist()
        
        logger.info(f"Total samples: {len(texts)}")
        logger.info(f"Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
        
        # First split: train and temp (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_state,
            stratify=labels  # Maintain class balance
        )
        
        # Second split: val and test from temp
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(1 - val_size),
            random_state=self.random_state,
            stratify=temp_labels
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
        
    def get_train_dataset(self) -> URLDataset:
        """Get training dataset."""
        return URLDataset(
            texts=self.train_texts,
            labels=self.train_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def get_val_dataset(self) -> URLDataset:
        """Get validation dataset."""
        return URLDataset(
            texts=self.val_texts,
            labels=self.val_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def get_test_dataset(self) -> URLDataset:
        """Get test dataset."""
        return URLDataset(
            texts=self.test_texts,
            labels=self.test_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.get_val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.get_test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_all_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all DataLoaders as a tuple."""
        return (
            self.get_train_dataloader(),
            self.get_val_dataloader(),
            self.get_test_dataloader()
        )


def main():
    """Test the dataset module."""
    
    # Paths
    base_dir = r"C:\PhishLens"
    data_path = os.path.join(base_dir, "data", "processed", "url", "processed_urls.csv")
    
    # Initialize data module
    print("=" * 60)
    print("Testing URL Dataset Module")
    print("=" * 60)
    
    data_module = URLDataModule(
        data_path=data_path,
        model_name="microsoft/deberta-v3-base",
        max_length=256,
        batch_size=16,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Get dataloaders
    train_loader, val_loader, test_loader = data_module.get_all_dataloaders()
    
    print(f"\nDataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test one batch
    print(f"\nTesting one batch from training data...")
    batch = next(iter(train_loader))
    
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['label'].shape}")
    print(f"  labels in batch: {batch['label'].tolist()}")
    
    # Decode one sample to verify
    sample_ids = batch['input_ids'][0]
    decoded = data_module.tokenizer.decode(sample_ids, skip_special_tokens=True)
    print(f"\nDecoded sample (first 200 chars):")
    print(f"  {decoded[:200]}...")
    
    print("\n" + "=" * 60)
    print("Dataset module test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()