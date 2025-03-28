from datasets import load_dataset
import requests
from PIL import Image
import io
from pycocotools import mask as mask_utils
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset("a8cheng/OpenSpatialDataset", split="train", streaming=True)

print("Samples from the training set:\n")

for i, example in enumerate(dataset):
    if i >= 5:
        break
    print(f"Sample {i}:")
    # Sample 3:
    #     filename
    #     conversations
    #     rle
    #     bbox
    # Load and display image from OpenImages using the filename
    
    def load_openimages_by_id(image_id, subset="train"):
        url = f"https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Failed to load image {image_id}: {e}")
            return None
        
    image = load_openimages_by_id(example["filename"])
    image.show()
    if image:
        plt.imshow(image)
        import matplotlib.patches as patches
        bboxes = example["bbox"]
        rle_mask = example["rle"]
        # Display RLE masks if available
        if rle_mask:
            try:
                # Try to use pycocotools for RLE decoding
                
                # Process each RLE mask
                for i, mask_data in enumerate(rle_mask):
                    # Decode RLE to binary mask
                    binary_mask = mask_utils.decode(mask_data)
                    
                    # Create a colored mask overlay with transparency
                    h, w = binary_mask.shape
                    mask_rgba = np.zeros((h, w, 4))
                    
                    # Use different colors for different masks
                    colors = [(1, 0, 0, 0.3), (0, 1, 0, 0.3), (0, 0, 1, 0.3),
                             (1, 1, 0, 0.3), (1, 0, 1, 0.3), (0, 1, 1, 0.3)]
                    color = colors[i % len(colors)]
                    
                    mask_rgba[binary_mask == 1] = color
                    
                    # Add the mask overlay to the plot
                    plt.imshow(mask_rgba)
                    
            except ImportError:
                print("pycocotools not installed. Install with: pip install pycocotools")
            except Exception as e:
                print(f"Failed to decode RLE mask: {e}")

        # Assuming bbox format is [x_min, y_min, x_max, y_max]
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, 
                   linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        
        plt.axis('off')
        plt.title(f"Image ID: {example['filename']}")
        plt.show()
    for key, value in example.items():
        if key == "filename" or key == "conversations":
            print(f"  {key}: {value}")
    print("-" * 80)
