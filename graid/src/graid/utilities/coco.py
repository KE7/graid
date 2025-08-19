# Official COCO Panoptic Categories
# Source: https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

from typing import Any, Optional

# Standard COCO Detection classes (80 classes, 0-79)
coco_labels = {
    -1: "undefined",
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# COCO Panoptic classes (133 classes, uses original COCO IDs)
coco_panoptic_labels = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    92: "banner",
    93: "blanket",
    95: "bridge",
    100: "cardboard",
    107: "counter",
    109: "curtain",
    112: "door-stuff",
    118: "floor-wood",
    119: "flower",
    122: "fruit",
    125: "gravel",
    128: "house",
    130: "light",
    133: "mirror-stuff",
    138: "net",
    141: "pillow",
    144: "platform",
    145: "playingfield",
    147: "railroad",
    148: "river",
    149: "road",
    151: "roof",
    154: "sand",
    155: "sea",
    156: "shelf",
    159: "snow",
    161: "stairs",
    166: "tent",
    168: "towel",
    171: "wall-brick",
    175: "wall-stone",
    176: "wall-tile",
    177: "wall-wood",
    178: "water-other",
    180: "window-blind",
    181: "window-other",
    184: "tree-merged",
    185: "fence-merged",
    186: "ceiling-merged",
    187: "sky-other-merged",
    188: "cabinet-merged",
    189: "table-merged",
    190: "floor-other-merged",
    191: "pavement-merged",
    192: "mountain-merged",
    193: "grass-merged",
    194: "dirt-merged",
    195: "paper-merged",
    196: "food-other-merged",
    197: "building-other-merged",
    198: "rock-merged",
    199: "wall-other-merged",
    200: "rug-merged",
}

# Backward compatibility
inverse_coco_label = {v: k for k, v in coco_labels.items()}
inverse_coco_panoptic_label = {v: k for k, v in coco_panoptic_labels.items()}

# For backward compatibility, add undefined label
coco_labels[-1] = "undefined"
inverse_coco_label["undefined"] = -1


def validate_coco_objects(objects: list[str]) -> tuple[bool, Optional[str]]:
    """
    Validate that all objects in the list are valid COCO object names.

    Args:
        objects: List of object names to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not objects:
        return True, None

    valid_coco_objects = set(coco_labels.values())
    # Remove 'undefined' from valid objects as it's not a real COCO class
    valid_coco_objects.discard("undefined")

    invalid_objects = []
    for obj in objects:
        if obj not in valid_coco_objects:
            invalid_objects.append(obj)

    if invalid_objects:
        return (
            False,
            f"Invalid COCO objects: {invalid_objects}. Valid objects: {sorted(valid_coco_objects)}",
        )

    return True, None


def get_valid_coco_objects() -> list[str]:
    """
    Get a list of valid COCO object names.

    Returns:
        List of valid COCO object names
    """
    valid_objects = list(coco_labels.values())
    # Remove 'undefined' as it's not a real COCO class
    if "undefined" in valid_objects:
        valid_objects.remove("undefined")
    return sorted(valid_objects)


def filter_detections_by_allowable_set(
    detections: list[dict[str, Any]], allowable_set: Optional[list[str]]
) -> list[dict[str, Any]]:
    """
    Filter detections to only include objects in the allowable set.

    Args:
        detections: List of detection dictionaries with 'class' or 'label' field
        allowable_set: List of allowed COCO object names, or None to allow all

    Returns:
        Filtered list of detections
    """
    if not allowable_set:
        return detections

    allowable_set_normalized = set(allowable_set)
    filtered_detections = []

    for detection in detections:
        # Handle different possible keys for class name
        class_name = (
            detection.get("class") or detection.get("label") or detection.get("name")
        )

        if class_name and class_name in allowable_set_normalized:
            filtered_detections.append(detection)

    return filtered_detections
