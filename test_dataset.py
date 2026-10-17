from datasets import load_dataset

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset:")
print(dataset)

print("\nFirst training example:")
print(dataset["train"][0])

print("\nSecond training example:")
print(dataset["train"][1])

print("\nThird training example:")
print(dataset["train"][2])