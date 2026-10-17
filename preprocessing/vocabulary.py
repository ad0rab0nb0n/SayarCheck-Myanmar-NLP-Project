from collections import Counter
from datasets import load_dataset
import sys
import os
import json


# Allow Python to find unicode_normalizer.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unicode_normalizer import clean_tokens_and_tags


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Minimum number of times a token must appear
MIN_FREQUENCY = 2


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")

print("Building vocabulary from cleaned training data...")


# --------------------------------------------------
# STEP 1: Count token frequencies
# --------------------------------------------------

counter = Counter()

for example in dataset["train"]:

    tokens = example["tokens"]
    tags = example["segment_tags"]

    clean_tokens, clean_tags = clean_tokens_and_tags(
        tokens,
        tags
    )

    for token in clean_tokens:
        counter[token] += 1


print("\nVocabulary analysis")
print("-------------------")

print("Unique tokens:", len(counter))

print("Tokens appearing only once:",
      sum(1 for count in counter.values() if count == 1))

print("Tokens appearing 2+ times:",
      sum(1 for count in counter.values() if count >= 2))


# --------------------------------------------------
# STEP 2: Create vocabulary
# --------------------------------------------------

word_to_id = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1
}


for token, count in counter.items():

    if count >= MIN_FREQUENCY:

        word_to_id[token] = len(word_to_id)


print("\nFinal vocabulary")
print("-------------------")

print("Minimum frequency:", MIN_FREQUENCY)

print("Vocabulary size:", len(word_to_id))


# --------------------------------------------------
# STEP 3: Save vocabulary
# --------------------------------------------------

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vocab.json"
)


with open(output_path, "w", encoding="utf-8") as f:

    json.dump(
        word_to_id,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nVocabulary saved to:")
print(output_path)


# --------------------------------------------------
# STEP 4: Show first 20 entries
# --------------------------------------------------

print("\nFirst 20 vocabulary entries:")

for token, idx in list(word_to_id.items())[:20]:

    print(f"{repr(token):20} -> {idx}")