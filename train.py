import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.dataset import (
    MyanmarSegmentationDataset,
    collate_fn
)

from models.bilstm_crf import BiLSTMCRF


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001

CHECKPOINT_DIR = "checkpoints"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")

split_dataset = dataset["train"].train_test_split(
    test_size=0.10,
    seed=42
)

train_data = split_dataset["train"]
validation_data = split_dataset["test"]

print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))


# ============================================================
# PYTORCH DATASETS
# ============================================================

vocab_path = "preprocessing/vocab.json"

print("\nCreating training dataset...")

train_dataset = MyanmarSegmentationDataset(
    train_data,
    vocab_path
)

print("\nCreating validation dataset...")

validation_dataset = MyanmarSegmentationDataset(
    validation_data,
    vocab_path
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("\nDataLoaders created!")

print("Training chunks:", len(train_dataset))
print("Validation chunks:", len(validation_dataset))


# ============================================================
# MODEL
# ============================================================

vocab_size = len(
    train_dataset.word_to_id
)

model = BiLSTMCRF(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    num_tags=2,
    pad_idx=0
)

model = model.to(device)

print("\nModel created!")
print("Vocabulary size:", vocab_size)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    labels
):

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for prediction, label in zip(
        predictions,
        labels
    ):

        for predicted_tag, true_tag in zip(
            prediction,
            label
        ):

            # B = 0
            # I = 1

            if predicted_tag == 0 and true_tag == 0:

                true_positive += 1

            elif predicted_tag == 0 and true_tag == 1:

                false_positive += 1

            elif predicted_tag == 1 and true_tag == 0:

                false_negative += 1


    precision = (
        true_positive /
        (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return precision, recall, f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, validation_loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in validation_loader:

            input_ids, labels, attention_mask = batch

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            predictions = model(
                input_ids,
                attention_mask
            )

            labels_cpu = labels.cpu()
            mask_cpu = attention_mask.cpu()

            for i, prediction in enumerate(
                predictions
            ):

                valid_length = int(
                    mask_cpu[i].sum().item()
                )

                true_labels = (
                    labels_cpu[i]
                    [:valid_length]
                    .tolist()
                )

                all_predictions.append(
                    prediction
                )

                all_labels.append(
                    true_labels
                )


    return calculate_metrics(
        all_predictions,
        all_labels
    )


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

best_f1 = 0.0


# ============================================================
# TRAINING
# ============================================================

print("\nStarting baseline training")
print("===========================")

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    model.train()

    total_loss = 0.0

    for batch_number, batch in enumerate(
        train_loader
    ):

        input_ids, labels, attention_mask = batch

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        loss = model(
            input_ids,
            attention_mask,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_loss += loss.item()

        if (
            batch_number + 1
        ) % 1000 == 0:

            print(
                f"Batch "
                f"{batch_number + 1}/"
                f"{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"\nAverage training loss: "
        f"{average_loss:.4f}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("Running validation...")

    precision, recall, f1 = evaluate(
        model,
        validation_loader
    )


    print("\nValidation Results")
    print("------------------")
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall:    {recall:.4f}"
    )
    print(
        f"F1 Score:  {f1:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        model_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": f1,
            },
            model_path
        )

        print(
            "\n⭐ New best model saved!"
        )
        print(
            "Path:",
            model_path
        )


print("\n===========================")
print("Baseline training complete!")
print(
    f"Best validation F1: {best_f1:.4f}"
)