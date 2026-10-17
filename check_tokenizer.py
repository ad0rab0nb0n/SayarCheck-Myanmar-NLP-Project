from datasets import load_dataset

from preprocessing.syllable_tokenizer import tokenize_myanmar


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!\n")


# Check a few real examples
for i in range(10):

    example = dataset["train"][i]

    original_tokens = example["tokens"]

    # Join dataset tokens back into text
    original_text = "".join(original_tokens)

    predicted_tokens = tokenize_myanmar(original_text)

    print("=" * 60)

    print("Original text:")
    print(original_text)

    print("\nDataset tokens:")
    print(" | ".join(original_tokens))

    print("\nOur tokenizer:")
    print(" | ".join(predicted_tokens))