from datasets import load_dataset
from torch.utils.data import DataLoader

from models.dataset import (
    MyanmarSegmentationDataset,
    collate_fn
)


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")


# Vocabulary file
vocab_path = "preprocessing/vocab.json"


# Create PyTorch dataset
train_dataset = MyanmarSegmentationDataset(
    dataset["train"],
    vocab_path
)


# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


print("\nDataLoader created!")


# Get one batch
input_ids, labels, attention_mask = next(
    iter(train_loader)
)


print("\nBatch information:")

print("Input IDs shape:")
print(input_ids.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nAttention mask shape:")
print(attention_mask.shape)


print("\nInput IDs:")
print(input_ids)

print("\nLabels:")
print(labels)

print("\nAttention mask:")
print(attention_mask)