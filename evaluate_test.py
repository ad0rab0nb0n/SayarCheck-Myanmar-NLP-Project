import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.dataset import (
    MyanmarSegmentationDataset,
    collate_fn
)

from models.bilstm_crf import BiLSTMCRF


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

test_data = dataset["test"]

print("Official test examples:", len(test_data))


# ============================================================
# CREATE TEST DATASET
# ============================================================

vocab_path = "preprocessing/vocab.json"

print("\nCreating test dataset...")

test_dataset = MyanmarSegmentationDataset(
    test_data,
    vocab_path
)

print(
    "Test chunks:",
    len(test_dataset)
)


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

print("Test DataLoader created!")


# ============================================================
# CREATE MODEL
# ============================================================

vocab_size = len(
    test_dataset.word_to_id
)

model = BiLSTMCRF(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    num_tags=2,
    pad_idx=0
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model_path = "checkpoints/best_model.pt"

print("\nLoading trained model...")

checkpoint = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "Best validation F1 from checkpoint:",
    checkpoint["f1"]
)

print("Model loaded successfully!")


# ============================================================
# METRICS
# ============================================================

true_positive = 0
false_positive = 0
false_negative = 0


# ============================================================
# TESTING
# ============================================================

print("\nRunning official test...")
print("========================")


with torch.no_grad():

    for batch_number, batch in enumerate(
        test_loader
    ):

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

            for predicted_tag, true_tag in zip(
                prediction,
                true_labels
            ):

                # B = 0
                # I = 1

                if predicted_tag == 0 and true_tag == 0:

                    true_positive += 1

                elif predicted_tag == 0 and true_tag == 1:

                    false_positive += 1

                elif predicted_tag == 1 and true_tag == 0:

                    false_negative += 1

        if (
            batch_number + 1
        ) % 1000 == 0:

            print(
                f"Processed "
                f"{batch_number + 1}/"
                f"{len(test_loader)} batches"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("========================================")
print("       OFFICIAL TEST RESULTS")
print("========================================")

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

print("========================================")

print("\nTest evaluation completed! 🎉")