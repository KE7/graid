#!/usr/bin/env python3
"""
Test script to verify the 3 depth questions (FrontOf, BehindOf, DepthRanking) work correctly.
Creates a stacked visualization with all images vertically arranged.
"""

import sys
import os
sys.path.insert(0, 'graid/src')

try:
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from pathlib import Path
    
    # Try importing GRAID modules with fallback handling
    try:
        from graid.data.ImageLoader import Bdd100kDataset
        DATASET_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import dataset loader: {e}")
        DATASET_AVAILABLE = False
    
    try:
        from graid.questions.ObjectDetectionQ import FrontOf, BehindOf, DepthRanking
        from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI, ObjectDetectionUtils
        QUESTIONS_AVAILABLE = True
    except ImportError as e:
        print(f"Error: Could not import question modules: {e}")
        QUESTIONS_AVAILABLE = False
        sys.exit(1)
    
    try:
        from graid.models.DepthPro import DepthPro
        DEPTH_MODEL_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import DepthPro: {e}")
        DEPTH_MODEL_AVAILABLE = False

except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    sys.exit(1)

def draw_boxes(
    image: np.ndarray,
    detections: list[ObjectDetectionResultI],
    alpha: float = 1.0,
) -> np.ndarray:
    """Overlay detections (xyxy) on an RGB image and return the visualised image."""
    colours: list[tuple[int, int, int]] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (255, 192, 203),
        (128, 128, 0),
    ]

    # Ensure image is in a format that PIL can handle
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[0] in [1, 3]:  # PyTorch tensor format (C, H, W)
        image = image.transpose(1, 2, 0)
    
    # Ensure it's a contiguous array
    image = np.ascontiguousarray(image)

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    for i, det in enumerate(detections):
        try:
            colour = colours[i % len(colours)] + (255,)
            xyxy = det.as_xyxy()
            # Support tensor/ndarray/list
            if hasattr(xyxy, 'shape') and len(xyxy.shape) > 1:
                for j in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = xyxy[j][:4]
                    draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
                    # Add label
                    label = str(det.label[j].item()) if isinstance(det.label, torch.Tensor) else str(det.label)
                    draw.text((x1, y1-15), label, fill=colour)
            else:
                x1, y1, x2, y2 = xyxy[:4]
                draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
                # Add label
                label = str(det.label)
                draw.text((x1, y1-15), label, fill=colour)
        except Exception as e:
            print(f"Error drawing detection {i}: {e}")
            continue

    return np.array(pil_img)


def test_depth_questions():
    """Test depth questions and create stacked visualization"""
    print("=== Testing Depth Questions ===")

    # Initialize results storage
    all_results = []

    if not DATASET_AVAILABLE:
        print("Bdd100kDataset not available, skipping test.")
        return

    print("Loading dataset...")
    try:
        # Load dataset
        dataset = Bdd100kDataset(split="val")
        images_and_detections = []
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            # Convert tensor image to PIL Image for processing
            pil_image = Image.fromarray(
                (data["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            images_and_detections.append((pil_image, data["labels"]))

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize question classes
    try:
        front_question = FrontOf(margin_ratio=0.1)
        behind_question = BehindOf(margin_ratio=0.1)
        depth_ranking = DepthRanking(k=3, margin_ratio=0.1)

        print("✅ Successfully initialized depth questions")
    except Exception as e:
        print(f"❌ Error initializing questions: {e}")
        return

    # Process each image
    for idx, (pil_image, detections) in enumerate(images_and_detections):
        print(f"\nProcessing image {idx+1}...")

        # Filter detections to only include relevant classes
        relevant_classes = ["traffic light", "traffic sign", "person", "car", "bus", "truck"]
        filtered_detections = []

        for detection in detections:
            if isinstance(detection.label, str):
                if detection.label in relevant_classes:
                    filtered_detections.append(detection)
            elif isinstance(detection.label, torch.Tensor):
                # Handle tensor labels - flatten and filter
                flattened = detection.flatten()
                for flat_det in flattened:
                    if str(flat_det.label) in relevant_classes:
                        filtered_detections.append(flat_det)

        print(f"Found {len(filtered_detections)} relevant detections")

        if len(filtered_detections) < 2:
            print("Not enough detections for depth questions, skipping...")
            continue

        # Test each question type
        questions_to_test = [
            ("FrontOf", front_question),
            ("BehindOf", behind_question),
            ("DepthRanking", depth_ranking),
        ]

        image_results = {
            "image": pil_image,
            "detections": filtered_detections,
            "questions": {},
        }

        for question_name, question in questions_to_test:
            try:
                print(f"  Testing {question_name}...")

                # Check if question is applicable
                if question.is_applicable(pil_image, filtered_detections):
                    # Apply the question
                    qa_pairs = question.apply(pil_image, filtered_detections)
                    print(f"    Generated {len(qa_pairs)} question-answer pairs")

                    for q, a in qa_pairs:
                        print(f"      Q: {q}")
                        print(f"      A: {a}")

                    image_results["questions"][question_name] = qa_pairs
                else:
                    print(f"    {question_name} not applicable to this image")
                    image_results["questions"][question_name] = []

            except Exception as e:
                print(f"    ❌ Error with {question_name}: {e}")
                image_results["questions"][question_name] = []

        all_results.append(image_results)

    # Create stacked visualization
    create_stacked_visualization(all_results)

def create_stacked_visualization(all_results):
    """Create a single stacked image with all test results"""
    if not all_results:
        print("No results to visualize")
        return
        
    print(f"\nCreating stacked visualization with {len(all_results)} images...")
    
    # Calculate dimensions for stacked image
    img_width = 640
    img_height = 480
    text_height = 200  # Space for question/answer text
    total_height_per_image = img_height + text_height
    total_height = total_height_per_image * len(all_results)
    
    # Create the stacked image
    stacked_image = Image.new('RGB', (img_width, total_height), color=(255, 255, 255))
    
    current_y = 0
    
    for idx, result in enumerate(all_results):
        # Get image with detections overlay
        image_with_boxes = draw_boxes(np.array(result['image']), result['detections'])
        image_with_boxes = Image.fromarray(image_with_boxes)
        
        # Resize if necessary
        if image_with_boxes.size != (img_width, img_height):
            image_with_boxes = image_with_boxes.resize((img_width, img_height))
        
        # Paste the image
        stacked_image.paste(image_with_boxes, (0, current_y))
        current_y += img_height
        
        # Add text with questions and answers
        text_img = Image.new('RGB', (img_width, text_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(text_img)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        y_offset = 10
        line_height = 15
        
        # Image header
        draw.text((10, y_offset), f"Image {idx+1} - Depth Questions Test Results:", 
                 fill=(0, 0, 0), font=font)
        y_offset += line_height * 2
        
        # Add questions and answers
        for question_type, qa_pairs in result['questions'].items():
            if qa_pairs:
                draw.text((10, y_offset), f"{question_type}:", fill=(0, 0, 255), font=font)
                y_offset += line_height
                
                for q, a in qa_pairs[:3]:  # Limit to first 3 to fit in space
                    # Wrap long questions
                    q_short = q[:80] + "..." if len(q) > 80 else q
                    draw.text((20, y_offset), f"Q: {q_short}", fill=(0, 100, 0), font=font)
                    y_offset += line_height
                    draw.text((20, y_offset), f"A: {a}", fill=(100, 0, 0), font=font)
                    y_offset += line_height
                    
                if len(qa_pairs) > 3:
                    draw.text((20, y_offset), f"... and {len(qa_pairs)-3} more", 
                             fill=(100, 100, 100), font=font)
                    y_offset += line_height
            else:
                draw.text((10, y_offset), f"{question_type}: No applicable questions", 
                         fill=(150, 150, 150), font=font)
                y_offset += line_height
            
            y_offset += line_height // 2
        
        # Paste the text section
        stacked_image.paste(text_img, (0, current_y))
        current_y += text_height
    
    # Save the stacked visualization
    output_path = "depth_questions_test_results_stacked.png"
    stacked_image.save(output_path)
    print(f"✅ Stacked visualization saved to: {output_path}")
    
    # Also display using matplotlib if available
    try:
        plt.figure(figsize=(12, len(all_results) * 6))
        plt.imshow(np.array(stacked_image))
        plt.axis('off')
        plt.title('Depth Questions Test Results (All Images Stacked)')
        plt.tight_layout()
        plt.savefig("depth_questions_matplotlib.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Also displayed with matplotlib")
    except Exception as e:
        print(f"Note: Could not display with matplotlib: {e}")

if __name__ == "__main__":
    test_depth_questions()