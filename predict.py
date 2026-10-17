import json
import torch

from models.bilstm_crf import BiLSTMCRF
from preprocessing.unicode_normalizer import normalize_token
from preprocessing.syllable_tokenizer import tokenize_text as syllable_tokenize


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

vocab_path = "preprocessing/vocab.json"

with open(vocab_path, "r", encoding="utf-8") as f:
    word_to_id = json.load(f)

unk_id = word_to_id["<UNK>"]


# --------------------------------------------------
# Load trained model
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
# Myanmar tokenization
# --------------------------------------------------

def tokenize_text(text):
    """
    Normalize the input and tokenize it using
    the same syllable tokenizer used during preprocessing.
    """

    text = normalize_token(text)

    return syllable_tokenize(text)


# --------------------------------------------------
# Segment text
# --------------------------------------------------

def segment_text(text):

    tokens = tokenize_text(text)

    if not tokens:
        return ""

    # --------------------------------------------------
    # Convert tokens to vocabulary IDs
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

    # --------------------------------------------------
    # Attention mask
    # --------------------------------------------------

    attention_mask = torch.ones(
        input_ids.shape,
        dtype=torch.bool
    ).to(device)

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------

    with torch.no_grad():

        predictions = model(
            input_ids,
            attention_mask
        )[0]

    # --------------------------------------------------
    # DEBUG: Show tokens and predicted tags
    # --------------------------------------------------

    print("\nTokens and predicted tags:")
    print("--------------------------------------")

    for token, tag in zip(tokens, predictions):

        tag_name = "B" if tag == 0 else "I"

        print(
            f"{token:15} -> {tag_name} ({tag})"
        )

    print("--------------------------------------")

    # --------------------------------------------------
    # Reconstruct segmented text
    # --------------------------------------------------

    result = []

    for token, tag in zip(tokens, predictions):

        if tag == 0:
            # B = Beginning of a new word

            if result:
                result.append(" ")

            result.append(token)

        else:
            # I = Inside an existing word

            result.append(token)

    return "".join(result)


# --------------------------------------------------
# Test
# --------------------------------------------------

print("\n======================================")
print(" Myanmar Word Segmentation")
print("======================================")

while True:

    text = input(
        "\nEnter Myanmar text (or 'q' to quit): "
    )

    if text.lower() == "q":
        break

    result = segment_text(text)

    print("\nInput:")
    print(text)

    print("\nSegmented:")
    print(result)