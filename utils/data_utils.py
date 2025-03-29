"""
Data utilities for loading and preprocessing text data.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Union, Iterator, Any
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from .tokenizer import TransformerTokenizer
import itertools
from functools import lru_cache


class TextDataset(Dataset):
    """Dataset for text sequences."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: TransformerTokenizer,
        max_length: int = 1024,
        return_tensors: bool = True,
        cache_tokenization: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text sequences
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
            cache_tokenization: Whether to cache tokenization results
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.cache_tokenization = cache_tokenization
        self._tokenization_cache = {}
    
    def __len__(self) -> int:
        return len(self.texts)
    
    @lru_cache(maxsize=1024)
    def _tokenize_cached(self, text: str) -> Dict[str, Any]:
        """
        Tokenize a text with caching.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tokenized text
        """
        encoding = self.tokenizer.encode(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt" if self.return_tensors else None
        )
        return encoding
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tokenized sample
        """
        text = self.texts[idx]
        
        if self.cache_tokenization:
            if isinstance(text, str):
                # Use the cached version if available
                encoding = self._tokenize_cached(text)
            else:
                encoding = self.tokenizer.encode(
                    text, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt" if self.return_tensors else None
                )
        else:
            encoding = self.tokenizer.encode(
                text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True,
                return_tensors="pt" if self.return_tensors else None
            )
        
        return encoding


class SequencePairDataset(Dataset):
    """Dataset for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        tokenizer: TransformerTokenizer,
        max_length: int = 1024,
        return_tensors: bool = True,
        cache_tokenization: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            source_texts: List of source text sequences
            target_texts: List of target text sequences
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
            cache_tokenization: Whether to cache tokenization results
        """
        assert len(source_texts) == len(target_texts), "Source and target texts must have the same length"
        
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.cache_tokenization = cache_tokenization
        self._tokenization_cache = {}
    
    def __len__(self) -> int:
        return len(self.source_texts)
    
    @lru_cache(maxsize=1024)
    def _tokenize_pair_cached(self, source: str, target: str) -> Dict[str, Any]:
        """
        Tokenize a text pair with caching.
        
        Args:
            source: Source text
            target: Target text
            
        Returns:
            Tokenized text pair
        """
        # Tokenize source text
        source_encoding = self.tokenizer.encode(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if self.return_tensors else None
        )
        
        # Tokenize target text
        target_encoding = self.tokenizer.encode(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if self.return_tensors else None
        )
        
        # Create decoder input ids and labels for seq2seq training
        decoder_input_ids = target_encoding["input_ids"]
        labels = target_encoding["input_ids"].clone()
        
        # Combine into a single encoding
        encoding = {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }
        
        return encoding
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tokenized sample
        """
        source = self.source_texts[idx]
        target = self.target_texts[idx]
        
        if self.cache_tokenization:
            if isinstance(source, str) and isinstance(target, str):
                # Use the cached version if available
                encoding = self._tokenize_pair_cached(source, target)
            else:
                encoding = self._tokenize_pair(source, target)
        else:
            encoding = self._tokenize_pair(source, target)
        
        return encoding
    
    def _tokenize_pair(self, source: str, target: str) -> Dict[str, Any]:
        """
        Tokenize a text pair.
        
        Args:
            source: Source text
            target: Target text
            
        Returns:
            Tokenized text pair
        """
        # Tokenize source text
        source_encoding = self.tokenizer.encode(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if self.return_tensors else None
        )
        
        # Tokenize target text
        target_encoding = self.tokenizer.encode(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if self.return_tensors else None
        )
        
        # Create decoder input ids and labels for seq2seq training
        decoder_input_ids = target_encoding["input_ids"]
        labels = target_encoding["input_ids"].clone()
        
        # Combine into a single encoding
        encoding = {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }
        
        return encoding


class StreamingTextDataset(Dataset):
    """
    Dataset for streaming large text files.
    This avoids loading the entire dataset into memory.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: TransformerTokenizer,
        max_length: int = 1024,
        return_tensors: bool = True,
        chunk_size: int = 1000,
    ):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the text file
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
            chunk_size: Number of examples to load at once
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.chunk_size = chunk_size
        
        # Count the number of lines in the file
        with open(file_path, "r", encoding="utf-8") as f:
            self.num_examples = sum(1 for _ in f)
        
        # Initialize the chunk cache
        self._current_chunk = None
        self._chunk_start = -1
        self._chunk_end = -1
    
    def __len__(self) -> int:
        return self.num_examples
    
    def _load_chunk(self, chunk_start: int) -> None:
        """
        Load a chunk of examples into memory.
        
        Args:
            chunk_start: Starting index of the chunk
        """
        chunk_end = min(chunk_start + self.chunk_size, self.num_examples)
        
        # Load the chunk
        examples = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Skip to the chunk start
            for _ in itertools.islice(f, chunk_start):
                pass
            
            # Load the chunk
            for line in itertools.islice(f, self.chunk_size):
                examples.append(line.strip())
        
        self._current_chunk = examples
        self._chunk_start = chunk_start
        self._chunk_end = chunk_end
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tokenized sample
        """
        # Check if the index is in the current chunk
        if self._chunk_start <= idx < self._chunk_end:
            chunk_idx = idx - self._chunk_start
            text = self._current_chunk[chunk_idx]
        else:
            # Determine which chunk to load
            chunk_start = (idx // self.chunk_size) * self.chunk_size
            self._load_chunk(chunk_start)
            
            chunk_idx = idx - self._chunk_start
            text = self._current_chunk[chunk_idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt" if self.return_tensors else None
        )
        
        return encoding


class BucketedBatchSampler(Sampler):
    """
    Sampler that groups texts of similar lengths into batches.
    This can improve training efficiency by reducing padding.
    """
    
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize the sampler.
        
        Args:
            lengths: List of sequence lengths
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            drop_last: Whether to drop the last incomplete batch
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group indices by sequence length
        indices = list(range(len(lengths)))
        sorted_indices = [i for _, i in sorted(zip(lengths, indices))]
        
        # Create batches
        self.batches = []
        for i in range(0, len(sorted_indices), batch_size):
            if drop_last and i + batch_size > len(sorted_indices):
                continue
            self.batches.append(sorted_indices[i:i + batch_size])
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over the batches.
        
        Returns:
            Iterator over batches
        """
        if self.shuffle:
            random_batches = self.batches.copy()
            np.random.shuffle(random_batches)
            for batch in random_batches:
                yield batch
        else:
            for batch in self.batches:
                yield batch
    
    def __len__(self) -> int:
        return len(self.batches)


def load_json_dataset(
    file_path: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
    text_field: str = "text",
    cache_tokenization: bool = True,
) -> TextDataset:
    """
    Load a dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
        text_field: Name of the field containing the text in each JSON object
        cache_tokenization: Whether to cache tokenization results
    
    Returns:
        TextDataset containing the texts
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        if isinstance(data[0], str):
            texts = data
        elif isinstance(data[0], dict) and text_field in data[0]:
            texts = [item[text_field] for item in data]
        else:
            raise ValueError(f"Could not extract texts from JSON data using field '{text_field}'")
    else:
        raise ValueError("JSON file should contain a list of strings or objects")
    
    return TextDataset(texts, tokenizer, max_length, return_tensors, cache_tokenization)


def load_text_file(
    file_path: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
    cache_tokenization: bool = True,
    streaming: bool = False,
) -> TextDataset:
    """
    Load a dataset from a text file, with one text per line.
    
    Args:
        file_path: Path to the text file
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
        cache_tokenization: Whether to cache tokenization results
        streaming: Whether to use streaming mode for large files
    
    Returns:
        TextDataset containing the texts
    """
    if streaming:
        return StreamingTextDataset(file_path, tokenizer, max_length, return_tensors)
    
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    return TextDataset(texts, tokenizer, max_length, return_tensors, cache_tokenization)


def load_csv_dataset(
    file_path: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
    text_column: str = "text",
    label_column: Optional[str] = None,
    cache_tokenization: bool = True,
) -> Union[TextDataset, Tuple[TextDataset, List]]:
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
        text_column: Name of the column containing the text
        label_column: Name of the column containing the labels (optional)
        cache_tokenization: Whether to cache tokenization results
    
    Returns:
        TextDataset containing the texts, optionally with labels
    """
    df = pd.read_csv(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV file")
    
    texts = df[text_column].tolist()
    
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV file")
        labels = df[label_column].tolist()
        return TextDataset(texts, tokenizer, max_length, return_tensors, cache_tokenization), labels
    
    return TextDataset(texts, tokenizer, max_length, return_tensors, cache_tokenization)


def load_sequence_pair_dataset(
    source_file: str,
    target_file: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
    cache_tokenization: bool = True,
) -> SequencePairDataset:
    """
    Load a sequence-to-sequence dataset from two files.
    
    Args:
        source_file: Path to the file containing source sequences
        target_file: Path to the file containing target sequences
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
        cache_tokenization: Whether to cache tokenization results
    
    Returns:
        SequencePairDataset containing the source and target sequences
    """
    with open(source_file, "r", encoding="utf-8") as f:
        source_texts = [line.strip() for line in f if line.strip()]
    
    with open(target_file, "r", encoding="utf-8") as f:
        target_texts = [line.strip() for line in f if line.strip()]
    
    assert len(source_texts) == len(target_texts), "Source and target files must have the same number of lines"
    
    return SequencePairDataset(source_texts, target_texts, tokenizer, max_length, return_tensors, cache_tokenization)


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler: Optional[Sampler] = None,
    pin_memory: bool = True,
    use_bucketing: bool = False,
    drop_last: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for data loading
        sampler: Custom sampler for the training dataset (optional)
        pin_memory: Whether to pin memory for faster GPU transfer
        use_bucketing: Whether to use bucketing for efficient batching
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        Dictionary containing the dataloaders
    """
    # Create sampler for the training dataset
    train_sampler = sampler
    
    if use_bucketing and sampler is None:
        # If bucketing is enabled, create a bucketed batch sampler
        # Check if the dataset has a function to get lengths
        if hasattr(train_dataset, "get_lengths"):
            lengths = train_dataset.get_lengths()
        else:
            # Use a default length (input_ids length) for each item
            lengths = []
            for i in range(len(train_dataset)):
                try:
                    sample = train_dataset[i]
                    if isinstance(sample, dict) and 'input_ids' in sample:
                        # Handle tensor or list/array input_ids
                        if isinstance(sample['input_ids'], torch.Tensor):
                            if sample['input_ids'].dim() == 1:
                                lengths.append(sample['input_ids'].size(0))
                            elif sample['input_ids'].dim() == 2:
                                lengths.append(sample['input_ids'].size(1))
                            else:
                                # Default length if dimensions are unexpected
                                lengths.append(128)
                        else:
                            # Handle list/array
                            lengths.append(len(sample['input_ids']))
                    else:
                        # Default length if input_ids not found
                        lengths.append(128)
                except Exception:
                    # Fallback to a default length if there's any error
                    lengths.append(128)
        
        # Ensure lengths list is not empty
        if not lengths:
            print("Warning: Could not determine lengths for bucketing. Using default dataloader.")
            use_bucketing = False
        else:
            train_sampler = BucketedBatchSampler(lengths, batch_size, shuffle, drop_last)
            batch_size = 1  # When using a batch sampler, batch_size must be 1
            shuffle = False  # Shuffle is handled by the sampler
    
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle if train_sampler is None else False,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    }
    
    if val_dataset is not None:
        # For validation, we don't need to use bucketing or shuffling
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    
    if test_dataset is not None:
        # For testing, we don't need to use bucketing or shuffling
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    
    return dataloaders


def prepare_dummy_data(
    tokenizer: TransformerTokenizer,
    num_samples: int = 100,
    seq_length: int = 64,
    vocab_size: Optional[int] = None,
) -> TextDataset:
    """
    Create a dummy dataset for testing.
    
    Args:
        tokenizer: Tokenizer for encoding texts
        num_samples: Number of samples to generate
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (defaults to tokenizer's vocab size)
    
    Returns:
        TextDataset containing the dummy data
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    
    # Generate random sequences of token IDs
    dummy_texts = []
    for _ in range(num_samples):
        token_ids = np.random.randint(1, vocab_size, size=seq_length).tolist()
        text = tokenizer.decode(token_ids)
        dummy_texts.append(text)
    
    return TextDataset(dummy_texts, tokenizer, seq_length) 