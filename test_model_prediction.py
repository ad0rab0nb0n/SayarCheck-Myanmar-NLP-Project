import json
import torch

from datasets import load_dataset

from models.bilstm_crf import BiLSTMCRF
from preprocessing.unicode_normalizer import clean_tokens_and_tags


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# Load vocabulary
# --------------------------------------------------

with open(
    "preprocessing/vocab.json",
    "r",
    encoding="utf-8"
) as f:

    word_to_id = json.load(f)

unk_id = word_to_id["<UNK>"]


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = BiLSTMCRF(
    vocab_size=len(word_to_id),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    num_tags=2,
    pad_idx=0
)

checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("\nLoading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

test_dataset = dataset["test"]

print("Test examples:", len(test_dataset))


# --------------------------------------------------
# Test one example
# --------------------------------------------------

example = test_dataset[0]

tokens = example["tokens"]
gold_tags = example["segment_tags"]

tokens, gold_tags = clean_tokens_and_tags(
    tokens,
    gold_tags
)


# --------------------------------------------------
# Convert tokens to IDs
# --------------------------------------------------

input_ids = []

for token in tokens:

    token_id = word_to_id.get(
        token,
        unk_id
    )

    input_ids.append(token_id)


input_ids = torch.tensor(
    [input_ids],
    dtype=torch.long
).to(device)


attention_mask = torch.ones(
    input_ids.shape,
    dtype=torch.bool
).to(device)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    predictions = model(
        input_ids,
        attention_mask
    )[0]


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n======================================")
print(" MODEL PREDICTION TEST")
print("======================================")

print("\nTokens:")
print("--------------------------------------")

for token, gold, pred in zip(
    tokens,
    gold_tags,
    predictions
):

    gold_name = "B" if gold == 0 else "I"
    pred_name = "B" if pred == 0 else "I"

    mark = "✓" if gold == pred else "✗"

    print(
        f"{token:15} "
        f"Gold: {gold_name}   "
        f"Pred: {pred_name}   "
        f"{mark}"
    )

print("--------------------------------------")


# --------------------------------------------------
# Reconstruct gold segmentation
# --------------------------------------------------

gold_result = []

for token, tag in zip(tokens, gold_tags):

    if tag == 0 and gold_result:
        gold_result.append(" ")

    gold_result.append(token)


# --------------------------------------------------
# Reconstruct predicted segmentation
# --------------------------------------------------

predicted_result = []

for token, tag in zip(tokens, predictions):

    if tag == 0 and predicted_result:
        predicted_result.append(" ")

    predicted_result.append(token)


print("\nGold segmentation:")
print("".join(gold_result))

print("\nModel segmentation:")
print("".join(predicted_result))