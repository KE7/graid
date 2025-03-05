from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

targets = [{'boxes': torch.Tensor([[ 997.0701,  188.0018, 1280.0000,  382.1104], [ 357.6905,  150.9485,  785.0663,  345.8993]]), 'labels': torch.Tensor([1, 1]), 'scores': torch.Tensor([1., 1.])}]
preds = [{'boxes': torch.Tensor([]), 'labels': torch.Tensor([]), 'scores': torch.Tensor([])}]

print(targets[0]['labels'].shape)
metric = MeanAveragePrecision(
            class_metrics=False,
            extended_summary=True,
            box_format="xyxy",
            iou_thresholds=[0.25],
            iou_type="bbox",
            backend="faster_coco_eval"
        )

metric.update(target=targets, preds=preds)

print(metric.compute())