"""
Minimal MMDetection 3.x config for BDD100K Faster R-CNN R50 FPN (inference only).

This is a port of the bdd100k-models 2.x config to mmdet 3.x format.
Only the model architecture and test pipeline are included — no training
or data loader configs are needed since we only run inference via
init_detector + inference_detector.

The model architecture dict is identical between mmdet 2.x and 3.x.
The key changes for 3.x compatibility are:
  - data_preprocessor replaces Normalize/Pad in the pipeline
  - PackDetInputs replaces DefaultFormatBundle + Collect
  - test_pipeline uses the 3.x format

Source checkpoint:
  https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth

BDD100K classes (10):
  pedestrian, rider, car, truck, bus, train, motorcycle, bicycle,
  traffic light, traffic sign
"""

# Required for mmdet 3.x registry lookups
default_scope = "mmdet"

# -- Model definition (identical to 2.x) ------------------------------------
model = dict(
    type="FasterRCNN",
    train_cfg=None,  # inference-only config; suppresses init_detector warning
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    # Test-time settings
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)
