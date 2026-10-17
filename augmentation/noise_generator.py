import random


def delete_token(tokens):
    """Randomly delete one token."""
    if len(tokens) <= 1:
        return tokens

    position = random.randrange(len(tokens))

    return (
        tokens[:position]
        + tokens[position + 1:]
    )


def duplicate_token(tokens):
    """Randomly duplicate one token."""
    position = random.randrange(len(tokens))

    return (
        tokens[:position]
        + [tokens[position]]
        + tokens[position:]
    )


def generate_noisy_tokens(tokens):
    """
    Generate synthetic noise from real dataset tokens.

    The input is a list of tokens from the
    original Myanmar segmentation dataset.
    """

    if len(tokens) <= 1:
        return tokens.copy()

    noise_type = random.choice([
        "delete",
        "duplicate"
    ])

    if noise_type == "delete":
        return delete_token(tokens)

    return duplicate_token(tokens)


if __name__ == "__main__":

    tokens = [
        "ဗိုလ်",
        "မှူး",
        "ချုပ်",
        "တောက်",
        "ထွန်း"
    ]

    print("Original:")
    print(tokens)

    print("\nSynthetic noisy versions:")

    for i in range(10):

        noisy = generate_noisy_tokens(tokens)

        print(
            f"{i + 1}.",
            noisy
        )