from __future__ import annotations

import argparse
import json
import types  # For monkey-patching
from collections import Counter
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

# Imports for custom loss
import torch.nn.functional as F
import wandb
from graid.data.ImageLoader import Bdd100kDataset
from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI
from numpy.typing import NDArray
from PIL import Image, ImageDraw

# LR scheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Metric for COCO-style mAP
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops.boxes import box_convert, generalized_box_iou
from tqdm import tqdm
from transformers import (
    ConditionalDetrForObjectDetection,
    DetrForObjectDetection,
    DetrImageProcessor,
)


def _sigmoid_focal_loss_with_class_weight(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Weighted version of Focal Loss, inspired by transformers library implementation.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        class_weights: A float tensor of shape (num_classes,).
        num_boxes: The number of boxes in the batch.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Alpha factor
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    # Apply per-class weights
    # The shape of targets is (num_queries, num_classes)
    # We can use it to index into the class_weights tensor
    if class_weights is not None:
        # Create a weight map of shape (num_queries, num_classes)
        # where each row is the weight for the corresponding target class
        weight_map = class_weights[targets.argmax(dim=1)]
        # Since targets is one-hot, we can multiply directly
        # and the zero elements will cancel out non-target weights
        loss = loss * weight_map.unsqueeze(1)

    return loss.mean(1).sum() / num_boxes


def apply_custom_losses(
    model: Union[DetrForObjectDetection, ConditionalDetrForObjectDetection],
    area_loss_power: float,
    class_weights: torch.Tensor | None = None,
):
    """
    Apply custom area-weighted and class-weighted losses to a DETR model
    by monkey-patching its loss functions.
    """
    is_conditional = isinstance(model, ConditionalDetrForObjectDetection)

    # ----------------------------------------------------------------------
    # 1. Area-weighted box/GIoU loss (compatible with both models)
    # ----------------------------------------------------------------------
    if area_loss_power > 0 and hasattr(model, "loss"):

        def loss_boxes_area_weighted(loss_self, outputs, targets, indices, num_boxes):
            """Area-weighted replica of DETR loss_boxes."""
            assert "pred_boxes" in outputs
            idx = loss_self._get_src_permutation_idx(indices)
            src_boxes = outputs["pred_boxes"][idx]
            tgt_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            areas = tgt_boxes[:, 2] * tgt_boxes[:, 3]
            w = areas.clamp(min=1e-6) ** area_loss_power
            ptr = 0
            for _, tgt_idx in indices:
                if len(tgt_idx):
                    segment = w[ptr : ptr + len(tgt_idx)]
                    w[ptr : ptr + len(tgt_idx)] = segment / segment.mean()
                ptr += len(tgt_idx)
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none")
            losses = {}
            losses["loss_bbox"] = (loss_bbox * w.unsqueeze(1)).sum() / num_boxes
            giou = generalized_box_iou(
                box_convert(src_boxes, "center_x_center_y_width_height", "xyxy"),
                box_convert(tgt_boxes, "center_x_center_y_width_height", "xyxy"),
            )
            loss_giou = 1 - torch.diag(giou)
            losses["loss_giou"] = (loss_giou * w).sum() / num_boxes
            return losses

        model.loss.loss_boxes = types.MethodType(loss_boxes_area_weighted, model.loss)
        print(f"✓ Enabled per-box area weighting (power={area_loss_power})")

    # ----------------------------------------------------------------------
    # 2. Class-weighted classification loss
    # ----------------------------------------------------------------------
    if class_weights is not None and hasattr(model, "loss"):
        if is_conditional:
            # Conditional DETR uses Focal Loss. We need to patch loss_labels.
            def loss_labels_class_weighted(
                loss_self, outputs, targets, indices, num_boxes, log=True
            ):
                """Class-weighted version of Conditional DETR's label loss."""
                assert "pred_logits" in outputs
                src_logits = outputs["pred_logits"]
                idx = loss_self._get_src_permutation_idx(indices)
                target_classes_o = torch.cat(
                    [t["class_labels"][J] for t, (_, J) in zip(targets, indices)]
                )
                target_classes = torch.full(
                    src_logits.shape[:2],
                    loss_self.num_classes,
                    dtype=torch.int64,
                    device=src_logits.device,
                )
                target_classes[idx] = target_classes_o
                target_classes_onehot = torch.zeros(
                    [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                    dtype=src_logits.dtype,
                    layout=src_logits.layout,
                    device=src_logits.device,
                )
                target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
                target_classes_onehot = target_classes_onehot[:, :, :-1]

                loss_ce = (
                    _sigmoid_focal_loss_with_class_weight(
                        src_logits,
                        target_classes_onehot,
                        class_weights,
                        num_boxes,
                        alpha=loss_self.focal_loss_alpha,
                        gamma=loss_self.focal_loss_gamma,
                    )
                    * src_logits.shape[1]
                )

                losses = {"loss_ce": loss_ce}
                if log:
                    losses["class_error"] = (
                        100
                        * (target_classes_o != src_logits[idx].argmax(-1))
                        .float()
                        .mean()
                    )
                return losses

            model.loss.loss_labels = types.MethodType(
                loss_labels_class_weighted, model.loss
            )
            print("✓ Enabled class-weighting for Conditional DETR (Focal Loss)")

        else:
            # Standard DETR uses cross-entropy and has a 'weight' parameter
            model.class_weight = class_weights.to(model.device)
            model.config.class_weight = class_weights.tolist()
            print("✓ Enabled class-weighting for standard DETR (Cross-Entropy)")


# ---------------------------------------------------------------------------
# Class-imbalance utilities
# ---------------------------------------------------------------------------


def compute_median_freq_weights(
    dataset, num_classes: int, workers: int = 16
) -> torch.Tensor:
    """Compute median-frequency balancing weights for foreground classes.

    For each class i: w_i = median(freq) / freq_i. Missing classes get weight 0.
    Returned tensor shape: (num_classes,)
    """
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "bdd_class_counts.json"

    if cache_file.exists():
        counter_data = json.loads(cache_file.read_text())
        counter = Counter({int(k): v for k, v in counter_data.items()})
    else:
        counter: Counter[int] = Counter()
        dl = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            collate_fn=lambda x: x,
        )
        for batch in dl:
            item = batch[0]
            for det in item["labels"]:
                counter[int(det.cls)] += 1
        cache_file.write_text(json.dumps(counter))

    counts = torch.tensor(
        [counter.get(i, 0) for i in range(num_classes)], dtype=torch.float
    )

    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    if nonzero.any():
        median = torch.median(counts[nonzero])
        weights[nonzero] = median / counts[nonzero]

    return weights


# ---------------------------------------------------------------------------
# Class mapping utilities
# ---------------------------------------------------------------------------


def get_original_detr_mapping():
    """Get the original DETR model's class mappings."""
    temp_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    return temp_model.config.id2label, temp_model.config.label2id


def create_bdd_to_detr_mapping():
    """Create mapping from BDD100K COCO class IDs to DETR model class IDs."""
    _, original_label2id = get_original_detr_mapping()

    # BDD100K uses these COCO class IDs (from dataset analysis)
    bdd_coco_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        6: "train",
        7: "truck",
        9: "traffic light",
        11: "stop sign",
    }

    # Create mapping from our COCO class ID to DETR class ID
    our_id_to_detr_id = {}
    for our_id, class_name in bdd_coco_classes.items():
        if class_name in original_label2id:
            detr_id = original_label2id[class_name]
            our_id_to_detr_id[our_id] = detr_id
        else:
            print(f"Warning: {class_name} not found in DETR model, mapping to 0 (N/A)")
            our_id_to_detr_id[our_id] = 0

    print("BDD100K COCO -> DETR class mapping:")
    for our_id, detr_id in our_id_to_detr_id.items():
        class_name = bdd_coco_classes[our_id]
        print(f"  {our_id} ({class_name}) -> {detr_id}")

    return our_id_to_detr_id


def create_bdd_direct_mapping():
    """Create direct BDD100K class mapping (no COCO intermediate step)."""
    # BDD100K has 12 original classes (0-11)
    bdd_classes = {
        0: "pedestrian",
        1: "person",
        2: "rider",
        3: "car",
        4: "truck",
        5: "bus",
        6: "train",
        7: "motorcycle",
        8: "bicycle",
        9: "traffic light",
        10: "traffic sign",
        11: "sidewalk",
    }

    # Direct mapping (identity function)
    direct_mapping = {i: i for i in range(12)}

    print("BDD100K Direct class mapping:")
    for bdd_id, mapped_id in direct_mapping.items():
        class_name = bdd_classes.get(bdd_id, f"unknown_{bdd_id}")
        print(f"  {bdd_id} ({class_name}) -> {mapped_id}")

    return direct_mapping, bdd_classes


_COLOURS = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 249, 10),
    (10, 249, 72),
    (10, 249, 146),
    (10, 249, 249),
    (10, 146, 249),
    (10, 72, 249),
    (72, 10, 249),
    (146, 10, 249),
    (249, 10, 249),
    (249, 10, 146),
    (249, 10, 72),
    (249, 10, 10),
]


def _draw_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    model: Union[DetrForObjectDetection, ConditionalDetrForObjectDetection],
    bdd_classes: dict[int, str] | None = None,
) -> NDArray[np.uint8]:
    """Overlay bounding-boxes on an image and return the result."""
    draw = ImageDraw.Draw(image)
    for i, (box, score, label_id) in enumerate(zip(boxes, scores, labels)):
        colour = _COLOURS[label_id % len(_COLOURS)]
        # The processor's post_process returns (x1, y1, x2, y2) already
        x1, y1, x2, y2 = map(int, box)

        # Use BDD class names if available, otherwise use model's id2label
        if bdd_classes:
            label_name = bdd_classes.get(label_id, f"CLS_{label_id}")
        else:
            label_name = model.config.id2label.get(label_id, f"CLS_{label_id}")

        caption = f"{label_name}: {score:.2%}"

        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        text_w = draw.textlength(caption)

        # Draw text background
        draw.rectangle([x1, y1, x1 + text_w + 4, y1 + 15], fill=colour)
        draw.text((x1 + 2, y1), caption, fill=(0, 0, 0))

    return np.array(image)


# ---------------------------------------------------------------------------
# Data loading and collation
# ---------------------------------------------------------------------------


class DetrDataCollator:
    """Collator to prepare data for DETR, adapted from BDD100K format."""

    def __init__(self, processor: DetrImageProcessor, class_mapping: dict[int, int]):
        self.processor = processor
        self.class_mapping = class_mapping

    def __call__(self, batch: list[dict[str, Any]]) -> Any:
        # Extract images and annotations from the batch
        images = [item["image"] for item in batch]
        annotations = []
        for idx, item in enumerate(batch):
            img_annots = []
            labels: list[ObjectDetectionResultI] = item["labels"]
            for label in labels:
                # Map our class ID to target class ID
                our_cls = int(label.cls)  # Ensure it's an int
                target_cls = self.class_mapping.get(our_cls, 0)  # fallback to 0

                # Convert box from (x1, y1, x2, y2) to COCO format (x_min, y_min, width, height)
                xyxy = label.as_xyxy()[0].tolist()
                x1, y1, x2, y2 = xyxy
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                img_annots.append(
                    {
                        "bbox": coco_bbox,
                        "category_id": target_cls,  # Use mapped class ID
                        "area": label.get_area().item(),
                    }
                )
            # Use stable integer image_id within the batch; the value is only
            # used to group annotations that belong to the same image.
            annotations.append({"image_id": idx, "annotations": img_annots})

        # Process batch with DETR processor
        processed_batch = self.processor(
            images=images, annotations=annotations, return_tensors="pt"
        )
        return processed_batch


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: Union[DetrForObjectDetection, ConditionalDetrForObjectDetection],
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1} [TRAIN]")
    for batch in progress:
        # Move batch to device
        inputs: dict[str, Any] = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        # The processor formats labels into a list of dicts, must be handled separately
        inputs["labels"] = [
            {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
        ]

        # Forward pass
        outputs = model(**inputs)

        # Loss: From DETR config
        # "class_cost": 1,              # Classification weight in Hungarian matching
        # "bbox_cost": 5,               # L1 bbox weight in Hungarian matching
        # "giou_cost": 2,               # GIoU weight in Hungarian matching
        # "bbox_loss_coefficient": 5,   # L1 bbox weight in final loss
        # "giou_loss_coefficient": 2,   # GIoU weight in final loss
        # "eos_coefficient": 0.1,       # "No-object" class weight
        # Total Loss = Classification Loss + λ₁ × L1 Loss + λ₂ × GIoU Loss
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        wandb.log(
            {
                "train_loss_step": loss.item(),
                **{f"train_{k}": v.item() for k, v in loss_dict.items()},
            }
        )

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(
    model: Union[DetrForObjectDetection, ConditionalDetrForObjectDetection],
    dataloader: DataLoader[Any],
    device: torch.device,
    epoch: int,
    processor: DetrImageProcessor,
):
    """Evaluate on the validation set and compute COCO-style mAP."""
    model.eval()

    total_loss = 0.0
    metric = MeanAveragePrecision()

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1} [VAL]")
    for batch in progress:
        # Move tensor inputs to device
        inputs: dict[str, Any] = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        # Labels need special handling (list[dict])
        inputs["labels"] = [
            {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
        ]

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        # ------------------------------------------------------------------
        # Prepare predictions & targets for mAP computation
        # ------------------------------------------------------------------
        # Determine original image sizes (h, w) for post-processing
        # The processor adds 'orig_size' to the labels
        target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]]).to(
            device
        )

        processed_outputs = processor.post_process_object_detection(
            # no threshold, metric handles scores
            outputs,
            target_sizes=target_sizes.tolist(),
            threshold=0.0,
        )

        preds_for_metric = []
        for pred in processed_outputs:
            preds_for_metric.append(
                {
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu(),
                }
            )

        targets_for_metric = []
        for tgt in batch["labels"]:
            # Get original image size to scale the target boxes to absolute pixel coords
            h, w = tgt["orig_size"].cpu().tolist()
            scaler = torch.tensor([w, h, w, h])

            # Convert boxes from relative (cx, cy, w, h) to absolute (x1, y1, x2, y2)
            boxes_cxcywh_abs = tgt["boxes"].cpu() * scaler
            cx, cy, width, height = boxes_cxcywh_abs.unbind(-1)
            x1 = cx - 0.5 * width
            y1 = cy - 0.5 * height
            x2 = cx + 0.5 * width
            y2 = cy + 0.5 * height
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            targets_for_metric.append(
                {
                    "boxes": boxes_xyxy,
                    "labels": tgt["class_labels"].cpu(),
                }
            )

        metric.update(preds_for_metric, targets_for_metric)

        progress.set_postfix(loss=loss.item())

    # Aggregate metrics
    avg_loss = total_loss / len(dataloader)
    metric_results = metric.compute()
    map_score = metric_results["map"].item()

    print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f} | mAP: {map_score:.4f}")
    return avg_loss, map_score


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def validate_model_name(model_name: str) -> None:
    """Validate that the model name is supported and provide helpful error messages."""
    supported_models = [
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",
        "microsoft/conditional-detr-resnet-50",
    ]

    if model_name not in supported_models:
        # Check if it's a similar name that might work
        model_name_lower = model_name.lower()
        if "conditional-detr" in model_name_lower:
            raise ValueError(
                f"Model '{model_name}' requires ConditionalDetrForObjectDetection, "
                f"but it's not available in your transformers version. "
                f"Please update transformers: pip install transformers>=4.21.0"
            )
        elif "detr" not in model_name_lower:
            raise ValueError(
                f"Model '{model_name}' doesn't appear to be a DETR model. "
                f"Supported models: {', '.join(supported_models)}"
            )
        else:
            print(f"Warning: '{model_name}' is not in the tested model list.")
            print(f"Tested models: {', '.join(supported_models)}")
            print(
                "Proceeding anyway - this may work if it's a compatible DETR variant."
            )


def main(args: argparse.Namespace) -> None:
    """Main function to run the training and evaluation."""
    # Validate model name
    validate_model_name(args.model_name)

    # Setup
    # Select GPU/CPU device based on the provided gpu_id
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.use_bdd_direct_mapping:
        class_mapping, bdd_classes = create_bdd_direct_mapping()
        num_classes = 12  # BDD100K has 12 classes
    else:
        class_mapping = create_bdd_to_detr_mapping()
        bdd_classes = None
        num_classes = 91  # Keep original DETR model's 91 classes

    # Determine model class and info based on model name
    if "conditional-detr" in args.model_name:
        base_model_class = ConditionalDetrForObjectDetection
        model_info = {"name": "Conditional DETR", "type": "conditional"}
    else:
        base_model_class = DetrForObjectDetection
        model_info = {"name": "DETR", "type": "standard"}
    print(f"Using {model_info['name']} ({model_info['type']}) model: {args.model_name}")

    # Model and Processor
    print("Loading DETR model and processor...")
    # Conditional DETR doesn't have the no_timm revision
    if "conditional-detr" in args.model_name:
        processor = DetrImageProcessor.from_pretrained(args.model_name)
    else:
        processor = DetrImageProcessor.from_pretrained(
            args.model_name, revision="no_timm"
        )

    # Check if we should load a pre-trained model or train from scratch
    # Use a GPU-specific directory so parallel runs don't overwrite each other
    output_dir = Path(f"detr_finetuned_model_gpu{args.gpu_id}")
    best_model_path = output_dir / "best_model"

    model: Union[DetrForObjectDetection, ConditionalDetrForObjectDetection]
    if args.load_model and best_model_path.exists():
        print(f"Loading pre-trained model from {best_model_path}")
        model = base_model_class.from_pretrained(best_model_path)
        assert model is not None
        model = model.to(device)
        processor = DetrImageProcessor.from_pretrained(best_model_path)
        print("✓ Pre-trained model loaded.")
        skip_training = True
    else:
        print("Loading base model for training...")

        # Create custom id2label and label2id if using direct BDD mapping
        id2label, label2id = None, None
        if args.use_bdd_direct_mapping and bdd_classes is not None:
            print(f"Using direct BDD100K mapping with {num_classes} classes")
            id2label = {i: bdd_classes[i] for i in range(num_classes)}
            label2id = {v: k for k, v in id2label.items()}

        model_class = base_model_class

        model_kwargs: dict[str, Any] = {
            "num_labels": num_classes,
            "id2label": id2label,
            "label2id": label2id,
            "ignore_mismatched_sizes": True,
        }
        # Regular DETR models use no_timm revision, conditional DETR doesn't
        if "conditional-detr" not in args.model_name:
            model_kwargs["revision"] = "no_timm"

        model = model_class.from_pretrained(args.model_name, **model_kwargs)

        # ------------------------------------------------------------------
        # Freeze parameters except heads + last k layers if requested
        # ------------------------------------------------------------------
        trainable_params = None
        if args.train_last_k >= 0:
            for p in model.parameters():
                p.requires_grad = False

            # Unfreeze heads
            for p in model.class_labels_classifier.parameters():
                p.requires_grad = True
            for p in model.bbox_predictor.parameters():
                p.requires_grad = True

            k = args.train_last_k
            if k > 0:
                # Unfreeze last layer only for the transformer
                transformer_k = 1
                if hasattr(model.model.encoder, "layers"):
                    for layer in model.model.encoder.layers[-transformer_k:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                if hasattr(model.model.decoder, "layers"):
                    for layer in model.model.decoder.layers[-transformer_k:]:
                        for p in layer.parameters():
                            p.requires_grad = True

                # Unfreeze last k ResNet layers if backbone exists
                if hasattr(model.model.backbone, "conv_encoder"):
                    conv_encoder = model.model.backbone.conv_encoder
                    # Flatten all bottleneck layers from layer1 through layer4
                    all_resnet_layers = []
                    for i in range(1, 5):
                        stage_name = f"layer{i}"
                        if hasattr(conv_encoder, stage_name):
                            stage = getattr(conv_encoder, stage_name)
                            all_resnet_layers.extend(list(stage))

                    # Unfreeze the last k bottleneck layers
                    for layer in all_resnet_layers[-k:]:
                        for p in layer.parameters():
                            p.requires_grad = True

            # After setting requires_grad, filter for trainable parameters for the optimizer
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Set loss coefficients from args
        model.config.eos_coefficient = args.eos_coefficient
        if args.bbox_loss_coef is not None:
            model.config.bbox_loss_coefficient = args.bbox_loss_coef
        if args.giou_loss_coef is not None:
            model.config.giou_loss_coefficient = args.giou_loss_coef

        model = model.to(device)
        print("✓ Model and processor loaded.")
        skip_training = False

    # Datasets and Dataloaders
    print("Loading BDD100K datasets...")
    train_dataset = Bdd100kDataset(
        split="train",
        # Use BDD categories for direct mapping, COCO for COCO mapping
        use_original_categories=args.use_bdd_direct_mapping,
        use_time_filtered=True,
    )
    val_dataset = Bdd100kDataset(
        split="val",
        use_original_categories=args.use_bdd_direct_mapping,
        use_time_filtered=True,
    )
    print(
        f"✓ Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images."
    )

    # -------------------------------------------------------------------
    # Optional: apply class-rebalancing when using direct BDD mapping
    # -------------------------------------------------------------------
    class_weights = None
    if (
        args.use_bdd_direct_mapping
        and not skip_training
        and not args.no_class_weighting
    ):
        print("Computing class-balancing weights (median-frequency)...")
        weights_fg = compute_median_freq_weights(train_dataset, num_classes=num_classes)
        min_weight = 1e-6
        weights_fg = torch.clamp(weights_fg, min=min_weight)
        class_weights = torch.cat([weights_fg, torch.tensor([args.eos_coefficient])])
        print("Class-weight vector:", class_weights.cpu().numpy())
        if bdd_classes:
            print("Per-class weights (BDD order):")
            for cls_id in range(num_classes):
                cls_name = bdd_classes.get(cls_id, str(cls_id))
                print(f"  {cls_name}: {weights_fg[cls_id].item():.3f}")

    # Apply all custom loss modifications after model is loaded
    apply_custom_losses(model, args.area_loss_power, class_weights)

    # -------------------------------------------------------------------
    # DataLoaders (validation always needed)
    # -------------------------------------------------------------------
    collator = DetrDataCollator(processor, class_mapping)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
    )

    if not skip_training:
        # Init wandb only if training
        import uuid

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"detr-finetune-bdd-{str(uuid.uuid4())[:8]}",
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            trainable_params if trainable_params is not None else model.parameters(),
            lr=args.lr,
        )

        # ------------------------------------------------------------------
        # Learning-rate scheduler (optional)
        # ------------------------------------------------------------------
        if args.lr_schedule == "step":
            scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
            print(f"✓ Using StepLR: step={args.lr_step} epochs, gamma={args.lr_gamma}")
        else:
            scheduler = None

        # Training loop
        print("\nStarting training...")
        best_val_loss = float("inf")
        output_dir.mkdir(exist_ok=True)

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model, train_dataloader, optimizer, device, epoch
            )
            val_loss, val_map = evaluate(
                model, val_dataloader, device, epoch, processor
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss_epoch": train_loss,
                    "val_loss_epoch": val_loss,
                    "val_map_epoch": val_map,
                }
            )

            if scheduler is not None:
                scheduler.step()

            # Log current LR (scheduler or static)
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(output_dir / "best_model")
                processor.save_pretrained(output_dir / "best_model")
                print(f"✓ New best model saved to {output_dir / 'best_model'}")

        print("\n✓ Training finished.")
        wandb.finish()
    else:
        print("\nSkipping training - using pre-trained model.")

        # ------------------------------------------------------------------
        # Evaluate loaded model on validation set to compute fresh mAP
        # ------------------------------------------------------------------
        print("\nEvaluating loaded model on validation set …")
        val_loss, val_map = evaluate(
            model, val_dataloader, device, epoch=0, processor=processor
        )
        print(f"Validation Loss: {val_loss:.4f} | mAP: {val_map:.4f}")

    # Visualization
    print("\nStarting visualization on validation images...")
    # GPU-specific visualization directory to avoid overwriting between parallel runs
    vis_dir = Path(f"detr_finetune_results_gpu{args.gpu_id}")
    vis_dir.mkdir(exist_ok=True)

    model.eval()
    for i in range(6):
        item = val_dataset[i]
        # The dataset might return different image formats, ensure it's a PIL Image
        image = item["image"]

        # Convert to PIL Image if it's not already
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)  # Convert to HWC

            # FIX: Handle different tensor value ranges
            if image.max() > 1.0:
                # Image is in [0, 255] range, convert to uint8
                image = image.clamp(0, 255).byte().cpu().numpy()
            else:
                # Image is in [0, 1] range, scale to [0, 255]
                image = (image * 255).clamp(0, 255).byte().cpu().numpy()

            pil_image = Image.fromarray(image)
        elif hasattr(image, "convert"):  # Already a PIL Image
            pil_image = image
        else:
            # Try to convert array-like to PIL Image
            import numpy as np

            if isinstance(image, np.ndarray):
                # Handle different numpy array value ranges
                if image.max() > 1.0:
                    # Array is in [0, 255] range
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                else:
                    # Array is in [0, 1] range
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                print(
                    f"Warning: Unexpected image type {type(image)}, skipping visualization"
                )
                continue

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # The target size for post-processing should be based on the original PIL image size
        image_size = pil_image.size  # This should be a tuple (width, height)
        # Convert to [[height, width]]
        target_sizes = [image_size[::-1]]

        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]

        vis_image = _draw_boxes(
            pil_image.copy(),
            results["boxes"].cpu().tolist(),
            results["scores"].cpu().tolist(),
            results["labels"].cpu().tolist(),
            model,
            bdd_classes if args.use_bdd_direct_mapping else None,
        )

        save_path = vis_dir / f"vis_{item['name'].replace('/', '_')}.png"
        Image.fromarray(vis_image).save(save_path)
        print(f"✓ Saved visualization to {save_path}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DETR on BDD100K.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/detr-resnet-50",
        help="HuggingFace model name. Supported models: "
        "facebook/detr-resnet-50, facebook/detr-resnet-101, "
        "microsoft/conditional-detr-resnet-50",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--load_model",
        action="store_true",
        default=False,
        help="Load pre-trained model instead of training from scratch.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="graid-detr-finetune",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (username or team).",
    )
    parser.add_argument(
        "--use_bdd_direct_mapping",
        action="store_true",
        default=False,
        help="Use direct BDD100K class mapping (0-11) instead of COCO mapping.",
    )
    parser.add_argument(
        "--eos_coefficient",
        type=float,
        default=0.1,
        help="Adjust the 'No-object' class weight (eos_coefficient) during fine-tuning.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=7,
        help="CUDA device id to run the training on (e.g. 0, 1, 2 …).",
    )
    parser.add_argument(
        "--no_class_weighting",
        action="store_true",
        default=False,
        help="Disable class weighting when using direct BDD mapping.",
    )
    parser.add_argument(
        "--bbox_loss_coef", type=float, default=None, help="Weight for the L1 box loss."
    )
    parser.add_argument(
        "--giou_loss_coef", type=float, default=None, help="Weight for the GIoU loss."
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="none",
        choices=["none", "step"],
        help="Type of LR scheduler to use.",
    )
    parser.add_argument(
        "--lr_step",
        type=int,
        default=3,
        help="StepLR: number of epochs between LR decays.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor for LR decay (gamma).",
    )
    parser.add_argument(
        "--train_last_k",
        type=int,
        default=-1,
        help="If >0, unfreeze heads plus last k transformer & ResNet layers. -1 trains all layers.",
    )
    parser.add_argument(
        "--area_loss_power",
        type=float,
        default=0.0,
        help="If >0, enable area weighting with this power (0.5=sqrt, 1=linear, etc.). Use 0 to disable.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
