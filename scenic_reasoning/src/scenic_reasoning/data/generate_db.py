from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.utilities.common import yolo_nuscene_transform

dataset = ObjDectDatasetBuilder(split="val", dataset="nuimages", db_name="NuImage_val_yolo", transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(768, 1280)))
model = Yolo(model="yolov8n.pt")
dataset.build(model=model)
