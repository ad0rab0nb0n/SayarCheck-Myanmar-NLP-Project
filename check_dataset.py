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


examples_to_check = 1000

zwsp_count = 0
changed_count = 0


print("Examples containing ZWSP:")
print("-" * 50)


for i in range(examples_to_check):

    tokens = dataset["train"][i]["tokens"]

    for token in tokens:

        cleaned_token = normalize_text(token)

        if "\u200b" in token:

            zwsp_count += 1

            print(f"Original : {repr(token)}")
            print(f"Cleaned  : {repr(cleaned_token)}")
            print()


        if token != cleaned_token:
            changed_count += 1


print("-" * 50)
print("Checked examples:", examples_to_check)
print("Tokens containing ZWSP:", zwsp_count)
print("Tokens changed by normalization:", changed_count)