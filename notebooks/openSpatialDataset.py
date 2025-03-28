from datasets import load_dataset

# Load the dataset (default split is 'train')
dataset = load_dataset("a8cheng/OpenSpatialDataset")

# You can access splits like this:
train_data = dataset["train"]
# If there's a 'test' or 'validation' split:
# test_data = dataset["test"]

# Print the number of samples
print(f"Total samples in train: {len(train_data)}\n")

# Example: Iterate and print first few samples
for i, example in enumerate(train_data):
    print(f"Sample {i}:")
    for key, value in example.items():
        print(f"  {key}: {value}")
    print("-" * 40)
    if i >= 4:  # Print only first 5
        break
