from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder

dataset = ObjDectDatasetBuilder(split="train", dataset="nuimages")
dataset.build()
