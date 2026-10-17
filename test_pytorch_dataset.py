from datasets import load_dataset
import json
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models"
    )
)

from dataset import MyanmarSegmentationDataset


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")


# --------------------------------------------------
# Load training split
# --------------------------------------------------

train_dataset = dataset["train"]


# --------------------------------------------------
# Create PyTorch dataset
# --------------------------------------------------

pytorch_dataset = MyanmarSegmentationDataset(
    train_dataset,
    "preprocessing/vocab.json"
)

print("\nPyTorch Dataset created!")
print("Number of examples:", len(pytorch_dataset))


# --------------------------------------------------
# Inspect first example
# --------------------------------------------------

input_ids, labels = pytorch_dataset[0]

print("\nFirst example:")
print("Input IDs:")
print(input_ids)

print("\nLabels:")
print(labels)

print("\nNumber of tokens:", len(input_ids))
print("Number of labels:", len(labels))


# --------------------------------------------------
# Show original tokens and labels
# --------------------------------------------------

original_example = train_dataset[0]

tokens = original_example["tokens"]
tags = original_example["segment_tags"]

print("\nOriginal dataset tokens and labels:")
print("--------------------------------------")

for token, tag in zip(tokens, tags):

    tag_name = "B" if tag == 0 else "I"

    print(
        f"{token:15} -> {tag_name} ({tag})"
    )

print("--------------------------------------")