from datasets import load_dataset
import numpy as np

from preprocessing.unicode_normalizer import clean_tokens_and_tags


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")

train_data = dataset["train"]

# We will inspect the first 10,000 examples first
NUM_EXAMPLES = 10000

lengths = []

print(f"\nChecking {NUM_EXAMPLES} training examples...")

for i in range(NUM_EXAMPLES):

    example = train_data[i]

    tokens = example["tokens"]
    tags = example["segment_tags"]

    clean_tokens, clean_tags = clean_tokens_and_tags(
        tokens,
        tags
    )

    lengths.append(len(clean_tokens))


lengths = np.array(lengths)

print("\nSequence length analysis")
print("========================")

print("Examples checked:", len(lengths))
print("Minimum length:", lengths.min())
print("Maximum length:", lengths.max())
print("Mean length:", round(lengths.mean(), 2))
print("Median length:", np.median(lengths))

print("\nPercentiles")
print("-----------")
print("90th percentile:", np.percentile(lengths, 90))
print("95th percentile:", np.percentile(lengths, 95))
print("99th percentile:", np.percentile(lengths, 99))

print("\nLong sequences")
print("--------------")
print("Longer than 128:", np.sum(lengths > 128))
print("Longer than 256:", np.sum(lengths > 256))
print("Longer than 512:", np.sum(lengths > 512))

print("\nPercentage")
print("----------")
print(
    "Longer than 128:",
    round(np.mean(lengths > 128) * 100, 2),
    "%"
)

print(
    "Longer than 256:",
    round(np.mean(lengths > 256) * 100, 2),
    "%"
)

print(
    "Longer than 512:",
    round(np.mean(lengths > 512) * 100, 2),
    "%"
)