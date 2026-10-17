from datasets import load_dataset
import random

from noise_generator import generate_noisy_tokens


NUM_EXAMPLES = 20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

train = dataset["train"]


print("Dataset loaded!")
print("Training examples:", len(train))


print("\nGenerating test examples...\n")


examples = []

while len(examples) < NUM_EXAMPLES:

    example = random.choice(train)

    tokens = example["tokens"]

    if len(tokens) <= 1:
        continue

    noisy_tokens = generate_noisy_tokens(tokens)

    if noisy_tokens == tokens:
        continue

    examples.append(
        (tokens, noisy_tokens)
    )


for i, (original, noisy) in enumerate(
    examples,
    start=1
):

    print(f"Example {i}")

    print(
        "Original:",
        " ".join(original)
    )

    print(
        "Noisy:   ",
        " ".join(noisy)
    )

    print()