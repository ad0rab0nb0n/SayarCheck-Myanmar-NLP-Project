from datasets import load_dataset
import random
import json
import os

NUM_EXAMPLES = 10000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

OUTPUT_PATH = "augmentation/augmented_data.json"

print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

train = dataset["train"]

print("Dataset loaded!")
print("Training examples:", len(train))


def create_noisy_example(tokens, tags):
    """
    Create synthetic spelling noise by:
    - deleting a syllable
    - duplicating a syllable

    The corresponding segmentation tag is
    removed/duplicated together with the syllable.
    """

    tokens = list(tokens)
    tags = list(tags)

    if len(tokens) < 3:
        return None

    operation = random.choice(["delete", "duplicate"])

    position = random.randrange(len(tokens))

    if operation == "delete":
        del tokens[position]
        del tags[position]

    else:
        tokens.insert(position, tokens[position])
        tags.insert(position, tags[position])

    return {
        "tokens": tokens,
        "segment_tags": tags
    }


augmented_examples = []

print("\nCreating synthetic augmented data...")

while len(augmented_examples) < NUM_EXAMPLES:

    example = random.choice(train)

    result = create_noisy_example(
        example["tokens"],
        example["segment_tags"]
    )

    if result is not None:
        augmented_examples.append(result)

    if len(augmented_examples) % 1000 == 0:
        print(
            f"Created {len(augmented_examples)}/"
            f"{NUM_EXAMPLES}"
        )


os.makedirs(
    "augmentation",
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        augmented_examples,
        f,
        ensure_ascii=False
    )


print("\n================================")
print("Synthetic augmentation complete!")
print("================================")
print("Examples:", len(augmented_examples))
print("Saved to:", OUTPUT_PATH)

print("\nFirst 5 examples:")

for example in augmented_examples[:5]:

    print(
        example["tokens"]
    )

    print(
        example["segment_tags"]
    )