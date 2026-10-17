from collections import Counter
from datasets import load_dataset
import json
import os

from preprocessing.unicode_normalizer import clean_tokens_and_tags


OUTPUT_PATH = "preprocessing/spelling_dictionary.json"


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

train = dataset["train"]

print("Dataset loaded!")
print("Training examples:", len(train))


# ============================================================
# BUILD WORD FREQUENCIES
# ============================================================

word_counter = Counter()

print("\nBuilding Myanmar word dictionary...")

for example in train:

    tokens, tags = clean_tokens_and_tags(
        example["tokens"],
        example["segment_tags"]
    )

    current_word = ""

    for token, tag in zip(tokens, tags):

        # B = beginning of a new word
        if tag == 0:

            if current_word:
                word_counter[current_word] += 1

            current_word = token

        # I = continuation of current word
        else:

            current_word += token

    # Add final word
    if current_word:
        word_counter[current_word] += 1


print("\nDictionary statistics")
print("---------------------")
print("Unique words:", len(word_counter))

print("\nMost common 20 words:")

for word, frequency in word_counter.most_common(20):

    print(
        f"{word:20} -> {frequency}"
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        word_counter,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nSpelling dictionary saved to:")
print(OUTPUT_PATH)