#!/usr/bin/env python
"""
Instance segmentation mAP evaluation (COCO mask AP) on BDD100K.

Example:
$ python evaluate_seg.py -d nuimage -m mask2former_lsj -bs 2 --device-id 0
"""
import argparse, os
import torch, numpy as np
import json, pickle
from datetime import datetime
from pathlib import Path
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from graid.data.ImageLoader import Bdd10kDataset
from graid.models.MMDetection import MMdetection_seg

import mmdet

def bitmask_to_rle(bitmask):
    mask = bitmask.squeeze(0).cpu().numpy().astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle, mask

def to_coco(out, img_id, start_id=0, truth=False):
    coco_list = []
    next_id = start_id
    for inst in out:
        rle, mask = bitmask_to_rle(inst.bitmask.tensor)
        ann = {
            'id': int(next_id),
            'image_id': int(img_id),
            'category_id': int(inst.cls),
            'segmentation': rle
        }
        if not truth:
            ann['score'] = float(inst.score)
        else:
            ann['area'] = int(mask.sum())
            ann['iscrowd'] = 0
        coco_list.append(ann)
        next_id += 1
    return coco_list, next_id

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset",
                    choices=['bdd'],
                    default='bdd')
    ap.add_argument("-m","--model",
                    choices=['mask2former', 'rtmdet', 'co-detr', 'grounded-sam2'],
                    default='mask2former')
    ap.add_argument("-c","--checkpoint", type=str)
    ap.add_argument("-s","--split", type=str, default='val')
    ap.add_argument("--batch-size","-bs", type=int, default=1)
    ap.add_argument("--device-id","-did", type=int, default=0)
    ap.add_argument("--limit", type=int, default=2000,
                    help='max images to evaluate')
    ap.add_argument(
        "-o", "--outdir", type=Path,
        help="Ouput directory (default: outputs/eval-seg_{dataset}_{model}_YYYYMMDD)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available()
                          else 'cpu')

    print(f'Running on device: {device}')

    today = datetime.now().strftime("%Y%m%d_%H%M")
    if args.outdir is None:
        args.outdir = Path(f"outputs/eval-seg_{args.dataset}_{args.model}_{today}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    pred_file  = args.outdir / "predicted.jsonl"
    truth_file = args.outdir / "truth.jsonl"
    imgs_file  = args.outdir / "images.jsonl"

    # MARK: Dataset
    if args.dataset == 'bdd':
        ds = Bdd10kDataset(args.split)
        classes = Bdd10kDataset._CATEGORIES_TO_COCO
    else:
        raise ValueError('Unknown dataset.')

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True,
                        collate_fn=lambda x: x)

    # MARK: Model
    if args.model == 'mask2former':
        pkg_dir = os.path.dirname(mmdet.__file__)
        cfg_file = os.path.join(
            pkg_dir, '.mim', 'configs',
            'mask2former',
            'mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py'
        )
        ckpt_file = args.checkpoint
        model = MMdetection_seg(str(cfg_file), str(ckpt_file), device=device)
    elif args.model == 'rtmdet':
        pkg_dir = os.path.dirname(mmdet.__file__)
        cfg_file = os.path.join(
            pkg_dir, '.mim', 'configs',
            'rtmdet',
            'rtmdet-ins_x_8xb16-300e_coco.py'
        )
        ckpt_file = args.checkpoint
        model = MMdetection_seg(str(cfg_file), str(ckpt_file), device=device)
    elif args.model == 'co-detr':
        cfg_file = './externals/Co-DETR/projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py'
        ckpt_file = args.checkpoint
        model = MMdetection_seg(str(cfg_file), str(ckpt_file), device=device)
    elif args.model == 'grounded-sam2':
        from graid.models.grounded_sam import GroundedSAM2
        sam2_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        sam2_ckpt = args.checkpoint
        gnd_model_id = 'IDEA-Research/grounding-dino-tiny'
        model = GroundedSAM2(
            sam2_cfg=sam2_cfg,
            sam2_ckpt=sam2_ckpt,
            gnd_model_id=gnd_model_id,
            classes=(classes if args.dataset in ['bdd'] else None),
            box_threshold=0.2,
            text_threshold=0.1
        )
    else:
        raise ValueError('Unknown model.')

    model.to(device)

    img_id = 0
    ann_id = 0

    pbar = tqdm(enumerate(loader), total=min(args.limit,len(loader)), desc='Inference')
    for idx, batch in pbar:
        if idx * args.batch_size >= args.limit: break

        inputs = [sample['path'] for sample in batch]
        targets = [sample['labels'] for sample in batch]
        out = model.identify_for_image_batch(inputs, batch_size=args.batch_size)

        for i, (pred, tgt) in enumerate(zip(out, targets)):
            H, W = batch[i]['image'].shape[-2:]
            path = batch[i]['path']

            name = str(path.split('/')[-1])
            with pred_file.open('a') as f:
                coco_dt, ann_id = to_coco(pred, img_id, start_id=ann_id, truth=False)
                entry = {name: coco_dt}
                f.write(json.dumps(entry) + "\n")
            with truth_file.open('a') as f:
                coco_gt, ann_id = to_coco(tgt, img_id, start_id=ann_id, truth=True)
                entry = {name: coco_gt}
                f.write(json.dumps(entry) + "\n")
            with imgs_file.open('a') as f:
                entry = {
                    'id': img_id,
                    'height': int(H),
                    'width': int(W),
                    'file_name': str(path)
                }
                f.write(json.dumps(entry) + "\n")

            img_id += 1

    # MARK: Evaluation
    def load_jsonl(path):
        items = []
        with open(path, 'r') as f:
            for line in f:
                d = json.loads(line)
                for v in d.values():
                    items.extend(v)
        return items

    def load_images_jsonl(path):
        items = []
        with open(path, 'r') as f:
            for line in f:
                items.append(json.loads(line))
        return items

    images = load_images_jsonl(imgs_file)
    gts = load_jsonl(truth_file)
    dts = load_jsonl(pred_file)

    target_labels = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "train": 6,
        "truck": 7,
        "traffic light": 9,
        "traffic sign": 11
    }
    categories = [{"id": id, "name": name} for name, id in target_labels.items()]

    coco_gt = COCO()
    coco_gt.dataset = {
        'images': images,
        'annotations': gts,
        'categories': categories,
        'info': {}
    }
    
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dts)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    # MARK: Save
    stat_names = [
        "AP", "AP50", "AP75",
        "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100",
        "AR_small", "AR_medium", "AR_large",
    ]

    stats_dict = {k: float(v) for k, v in zip(stat_names, coco_eval.stats)}

    stats_file = args.outdir / "coco_stats.json"
    with stats_file.open("w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[✓] COCO stats written to {stats_file}")

    full_file = args.outdir / "coco_eval_full.pkl"
    with full_file.open("wb") as f:
        pickle.dump({"eval": coco_eval.eval, "stats": coco_eval.stats}, f)

if __name__ == "__main__":
    main()