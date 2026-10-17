from datasets import load_dataset
import unicodedata


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "")
    return text


# Load dataset
dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

examples_to_process = 1000

changed_tokens = 0
total_tokens = 0

print("=" * 60)
print("MYANMAR TEXT NORMALIZATION")
print("=" * 60)

for i in range(examples_to_process):

    tokens = dataset["train"][i]["tokens"]

    cleaned_tokens = []

    for token in tokens:

        total_tokens += 1

        cleaned_token = normalize_text(token)

        if token != cleaned_token:
            changed_tokens += 1

            if changed_tokens <= 10:
                print("\nOriginal :", repr(token))
                print("Cleaned  :", repr(cleaned_token))

        cleaned_tokens.append(cleaned_token)


print("\n" + "=" * 60)
print("NORMALIZATION SUMMARY")
print("=" * 60)

print("Examples processed :", examples_to_process)
print("Total tokens       :", total_tokens)
print("Changed tokens     :", changed_tokens)