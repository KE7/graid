import time
from itertools import islice

import torch

from graid.data.ImageLoader import Bdd10kDataset
from graid.models.Detectron import Detectron_seg
from graid.utilities.common import get_default_device

# Test configuration
NUM_EXAMPLES = 5
BATCH_SIZE = 3

# Initialize dataset
print("Loading BDD10K dataset...")
bdd = Bdd10kDataset(split="val")

# Initialize Detectron2 segmentation model
print("Loading Detectron2 segmentation model...")
config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
weights_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

model = Detectron_seg(
    config_file=config_file,
    weights_file=weights_file,
    threshold=0.5,
    device=get_default_device(),
)

print(f"Model loaded on device: {get_default_device()}")
print(f"Testing with batch size: {BATCH_SIZE}")

# Collect batch of images
print(f"\nCollecting {BATCH_SIZE} images from dataset...")
images = []
for i, data in enumerate(islice(bdd, BATCH_SIZE)):
    image = data["image"]  # Extract image from the data dictionary
    print(f"  Image {i+1}: {image.shape}")
    images.append(image)

# Convert to tensor batch
print("\nConverting images to tensor batch...")
# Images are already tensors from the dataset, just stack them
tensor_images = torch.stack(images)
print(f"Batch tensor shape: {tensor_images.shape}")

# Test single image processing (for comparison)
print("\n=== Testing Single Image Processing ===")
start_time = time.time()
single_results = []
for i in range(tensor_images.shape[0]):
    single_img = tensor_images[i]
    result = model.identify_for_image(single_img)
    single_results.append(result)
    print(f"  Image {i+1}: Found {len(result)} instances")
single_time = time.time() - start_time
print(f"Single image processing time: {single_time:.3f}s")

# Test batch processing
print("\n=== Testing Batch Processing ===")
start_time = time.time()
try:
    batch_results = model.identify_for_image_batch(tensor_images)
    batch_time = time.time() - start_time
    print(f"Batch processing time: {batch_time:.3f}s")
    print(f"Speedup: {single_time/batch_time:.2f}x")

    # Verify results consistency
    print("\n=== Verifying Results Consistency ===")
    for i in range(len(single_results)):
        single_count = len(single_results[i])
        batch_count = len(batch_results[i])
        print(f"  Image {i+1}: Single={single_count}, Batch={batch_count}")

        if single_count != batch_count:
            print(f"    WARNING: Count mismatch!")
        else:
            print(f"    ✓ Counts match")

except Exception as e:
    print(f"Batch processing failed: {e}")
    import traceback

    traceback.print_exc()

print("\nTest completed!")
