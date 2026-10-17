import unicodedata


def normalize_token(token):
    """
    Normalize one token.

    1. Apply Unicode NFC normalization.
    2. Remove Zero Width Space (ZWSP).
    3. Return the cleaned token.
    """

    # Step 1: Unicode normalization
    token = unicodedata.normalize("NFC", token)

    # Step 2: Remove Zero Width Space
    token = token.replace("\u200b", "")

    return token


def clean_tokens_and_tags(tokens, tags):
    """
    Clean tokens while keeping tokens and tags aligned.
    """

    clean_tokens = []
    clean_tags = []

    for token, tag in zip(tokens, tags):

        cleaned_token = normalize_token(token)

        # Skip tokens that become empty after cleaning
        if cleaned_token == "":
            continue

        clean_tokens.append(cleaned_token)
        clean_tags.append(tag)

    return clean_tokens, clean_tags

# Test
if __name__ == "__main__":

    test_tokens = [
        "မြန်",
        "မာ",
        "\u200b",
        "စာ"
    ]

    test_tags = [
        0,
        1,
        1,
        0
    ]

    clean_tokens, clean_tags = clean_tokens_and_tags(
        test_tokens,
        test_tags
    )

    print("Original tokens:")
    print(test_tokens)

    print("\nOriginal tags:")
    print(test_tags)

    print("\nClean tokens:")
    print(clean_tokens)

    print("\nClean tags:")
    print(clean_tags)