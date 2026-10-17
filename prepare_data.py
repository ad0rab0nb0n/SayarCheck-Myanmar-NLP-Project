from datasets import load_dataset

from models.dataset import MyanmarSegmentationDataset


print("Loading dataset...")

dataset = load_dataset(
    "chuuhtetnaing/myanmar-text-segmentation-dataset"
)

print("Dataset loaded!")

# --------------------------------------------------
# Split the original training set
# --------------------------------------------------

print("\nCreating train/validation split...")

split_dataset = dataset["train"].train_test_split(
    test_size=0.10,
    seed=42
)

train_data = split_dataset["train"]
validation_data = split_dataset["test"]

test_data = dataset["test"]

print("\nDataset sizes")
print("-------------")
print("Original training:", len(dataset["train"]))
print("Training:", len(train_data))
print("Validation:", len(validation_data))
print("Official test:", len(test_data))


# --------------------------------------------------
# Create PyTorch datasets
# --------------------------------------------------

vocab_path = "preprocessing/vocab.json"

print("\nCreating training dataset...")
train_dataset = MyanmarSegmentationDataset(
    train_data,
    vocab_path
)

print("\nCreating validation dataset...")
validation_dataset = MyanmarSegmentationDataset(
    validation_data,
    vocab_path
)

print("\nCreating test dataset...")
test_dataset = MyanmarSegmentationDataset(
    test_data,
    vocab_path
)


# --------------------------------------------------
# Print final information
# --------------------------------------------------

print("\nFinal dataset information")
print("=========================")

print("Training examples:", len(train_data))
print("Training chunks:", len(train_dataset))

print("Validation examples:", len(validation_data))
print("Validation chunks:", len(validation_dataset))

print("Test examples:", len(test_data))
print("Test chunks:", len(test_dataset))