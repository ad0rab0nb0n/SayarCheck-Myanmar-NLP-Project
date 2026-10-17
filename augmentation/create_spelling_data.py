import json
import random
import os
import sys

from datasets import load_dataset

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from preprocessing.unicode_normalizer import clean_tokens_and_tags


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = "augmentation/spelling_training_data.json"

NUM_EXAMPLES = 10000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# MYANMAR CHARACTER CONFUSION PAIRS
# ============================================================
#
# These represent common character-level substitutions.
#
# We use them only to CREATE synthetic noisy examples.
# They are NOT automatic correction rules.
# ============================================================

CONFUSION_PAIRS = {

    "ပ": ["ပြ"],
    "ပြ": ["ပ"],

    "ဖ": ["ဖြ"],
    "ဖြ": ["ဖ"],

    "တ": ["ထ"],
    "ထ": ["တ"],

    "က": ["ခ"],
    "ခ": ["က"],

    "စ": ["ဆ"],
    "ဆ": ["စ"],

    "လ": ["လှ"],
    "လှ": ["လ"],

    "န": ["နှ"],
    "နှ": ["န"],

    "မ": ["မြ"],
    "မြ": ["မ"],

    "ရ": ["ယ"],
    "ယ": ["ရ"],

    "င": ["ဉ"],
    "ဉ": ["င"],

}


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading Myanmar dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

train = dataset["train"]

print(
    "Training examples:",
    len(train)
)


# ============================================================
# RECONSTRUCT WORDS
# ============================================================

def reconstruct_words(tokens, tags):

    words = []

    current_word = ""

    for token, tag in zip(tokens, tags):

        if tag == 0:

            if current_word:

                words.append(
                    current_word
                )

            current_word = token

        else:

            current_word += token

    if current_word:

        words.append(
            current_word
        )

    return words


# ============================================================
# CREATE ONE SYNTHETIC ERROR
# ============================================================
def create_noise(word):
    # Only create noise for Myanmar words
    if len(word) < 2:
        return None

    # Skip words containing numbers, English, or other scripts
    if any(char.isdigit() for char in word):
        return None

    if any("A" <= char <= "Z" or "a" <= char <= "z" for char in word):
        return None

    # Myanmar Unicode range
    if not all(
        ("\u1000" <= char <= "\u109F")
        or char in "၊။,!?:;()- "
        for char in word
    ):
        return None

    characters = list(word)

    # Avoid modifying punctuation
    valid_indices = [
        i for i, char in enumerate(characters)
        if "\u1000" <= char <= "\u109F"
    ]

    if not valid_indices:
        return None

    operation = random.choices(
        ["replace", "delete", "duplicate"],
        weights=[50, 25, 25],
        k=1
    )[0]

    if operation == "delete":
        if len(valid_indices) < 2:
            return None

        index = random.choice(valid_indices)
        noisy = (
            characters[:index]
            + characters[index + 1:]
        )

    elif operation == "duplicate":
        index = random.choice(valid_indices)

        noisy = (
            characters[:index]
            + [characters[index]]
            + characters[index:]
        )

    else:
        possible_indices = [
            i for i in valid_indices
            if characters[i] in CONFUSION_PAIRS
        ]

        if not possible_indices:
            return None

        index = random.choice(possible_indices)

        char = characters[index]
        replacement = random.choice(
            CONFUSION_PAIRS[char]
        )

        characters[index] = replacement
        noisy = characters

    noisy_word = "".join(noisy)

    # Final safety checks
    if noisy_word == word:
        return None

    if len(noisy_word) < 2:
        return None

    return noisy_word
# ============================================================
# GENERATE DATA
# ============================================================

print(
    "\nGenerating synthetic spelling data..."
)

training_data = []

attempts = 0

max_attempts = NUM_EXAMPLES * 20


while (
    len(training_data) < NUM_EXAMPLES
    and attempts < max_attempts
):

    attempts += 1


    # Random training example
    example_index = random.randrange(
        len(train)
    )

    example = train[
        example_index
    ]


    tokens, tags = clean_tokens_and_tags(
        example["tokens"],
        example["segment_tags"]
    )


    words = reconstruct_words(
        tokens,
        tags
    )


    if not words:

        continue


    # Pick a word with enough characters
    candidate_words = [
        word
        for word in words
        if len(word) >= 2
    ]


    if not candidate_words:

        continue


    correct_word = random.choice(
        candidate_words
    )


    noisy_word = create_noise(
        correct_word
    )


    if noisy_word is None:

        continue


    # Store pair
    training_data.append({

        "correct":
            correct_word,

        "noisy":
            noisy_word

    })


# ============================================================
# SAVE DATA
# ============================================================

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
        training_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n================================"
)

print(
    "SYNTHETIC SPELLING DATA CREATED"
)

print(
    "================================"
)

print(
    "Examples:",
    len(training_data)
)

print(
    "Output:",
    OUTPUT_PATH
)


print(
    "\nSample examples:"
)


for example in training_data[:20]:

    print(
        f"  {example['noisy']}"
        f"  ->  "
        f"{example['correct']}"
    )