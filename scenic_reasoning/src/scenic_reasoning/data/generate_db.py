from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder

dataset = ObjDectDatasetBuilder(split="train", dataset="nuimages", db_name="NuImage_train_gt")
dataset.build()
