import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from models.dataset import (
    MyanmarSegmentationDataset,
    collate_fn
)

from models.bilstm_crf import BiLSTMCRF


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")


# -----------------------------------------
# 2. Create PyTorch dataset
# -----------------------------------------

vocab_path = "preprocessing/vocab.json"

train_dataset = MyanmarSegmentationDataset(
    dataset["train"],
    vocab_path
)


# -----------------------------------------
# 3. Create DataLoader
# -----------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


# -----------------------------------------
# 4. Get one batch
# -----------------------------------------

input_ids, labels, attention_mask = next(
    iter(train_loader)
)


print("\nBatch shape:")
print("Input IDs:", input_ids.shape)
print("Labels:", labels.shape)
print("Mask:", attention_mask.shape)


# -----------------------------------------
# 5. Create model
# -----------------------------------------

vocab_size = len(train_dataset.word_to_id)

model = BiLSTMCRF(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    num_tags=2,
    pad_idx=0
)


print("\nModel created!")


# -----------------------------------------
# 6. Calculate loss
# -----------------------------------------

loss = model(
    input_ids,
    attention_mask,
    labels
)


print("\nLoss:")
print(loss)


# -----------------------------------------
# 7. Make predictions
# -----------------------------------------

predictions = model(
    input_ids,
    attention_mask
)


print("\nPredictions:")
print(predictions)


print("\nNumber of predictions in batch:")

for i, prediction in enumerate(predictions):

    print(
        f"Example {i}: "
        f"{len(prediction)} predictions"
    )