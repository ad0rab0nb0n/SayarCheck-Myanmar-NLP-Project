from datasets import load_dataset

from preprocessing.syllable_tokenizer import tokenize_text
from preprocessing.unicode_normalizer import clean_tokens_and_tags


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

train = dataset["train"]

print("Dataset loaded!")
print()


num_examples = 5000

exact_matches = 0
total_examples = 0

mismatch_examples = []


for i in range(num_examples):

    example = train[i]

    dataset_tokens, tags = clean_tokens_and_tags(
        example["tokens"],
        example["segment_tags"]
    )

    # Reconstruct the original text from dataset tokens.
    original_text = "".join(dataset_tokens)

    predicted_tokens = tokenize_text(original_text)

    total_examples += 1

    if dataset_tokens == predicted_tokens:

        exact_matches += 1

    else:

        if len(mismatch_examples) < 30:

            mismatch_examples.append(
                (
                    dataset_tokens,
                    predicted_tokens
                )
            )


print("=" * 60)
print("TOKENIZER COMPARISON")
print("=" * 60)

print("Examples checked:", total_examples)
print("Exact matches:", exact_matches)

match_rate = (
    exact_matches / total_examples * 100
)

print(
    f"Exact tokenization match rate: {match_rate:.2f}%"
)

print()
print("=" * 60)
print("MISMATCH EXAMPLES")
print("=" * 60)

for i, (dataset_tokens, predicted_tokens) in enumerate(
    mismatch_examples,
    start=1
):

    print()
    print(f"Example {i}")

    print(
        "Dataset:",
        " | ".join(dataset_tokens)
    )

    print(
        "Ours:   ",
        " | ".join(predicted_tokens)
    )