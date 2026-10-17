import json
import torch

from datasets import load_dataset

from models.bilstm_crf import BiLSTMCRF
from preprocessing.syllable_tokenizer import tokenize_text
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


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

example = dataset["test"][0]

tokens = example["tokens"]
gold_tags = example["segment_tags"]

tokens, gold_tags = clean_tokens_and_tags(
    tokens,
    gold_tags
)


# --------------------------------------------------
# Create RAW unsegmented text
# --------------------------------------------------

raw_text = "".join(tokens)

print("\n======================================")
print(" RAW INFERENCE TEST")
print("======================================")

print("\nRaw text:")
print(raw_text)


# --------------------------------------------------
# Tokenize raw text again
# --------------------------------------------------

new_tokens = tokenize_text(raw_text)

print("\nTokens after raw-text tokenization:")
print("--------------------------------------")

print(new_tokens)

print("--------------------------------------")


# --------------------------------------------------
# Compare tokenization
# --------------------------------------------------

print("\nOriginal token count:", len(tokens))
print("New token count:     ", len(new_tokens))

if tokens == new_tokens:
    print("\n✓ Tokenization is IDENTICAL")
else:
    print("\n✗ Tokenization is DIFFERENT")


# --------------------------------------------------
# Convert tokens to IDs
# --------------------------------------------------

input_ids = []

for token in new_tokens:

    input_ids.append(
        word_to_id.get(
            token,
            unk_id
        )
    )


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
# Display predictions
# --------------------------------------------------

print("\nPredictions:")
print("--------------------------------------")

for token, tag in zip(
    new_tokens,
    predictions
):

    tag_name = "B" if tag == 0 else "I"

    print(
        f"{token:15} -> {tag_name}"
    )

print("--------------------------------------")


# --------------------------------------------------
# Reconstruct prediction
# --------------------------------------------------

result = []

for token, tag in zip(
    new_tokens,
    predictions
):

    if tag == 0 and result:
        result.append(" ")

    result.append(token)


print("\nPredicted segmentation:")
print("".join(result))