from datasets import load_dataset


# Load the dataset
dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)


# Get the first training example
example = dataset["train"][1]

tokens = example["tokens"]
tags = example["segment_tags"]


print("TOKENS:")
print(tokens)

print("\nTAGS:")
print(tags)


print("\nTOKEN + TAG:")
for token, tag in zip(tokens, tags):
    if tag == 0:
        label = "B"
    else:
        label = "I"

    print(f"{token:10} -> {label}")