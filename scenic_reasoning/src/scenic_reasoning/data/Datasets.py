import os
from typing import Optional

import torch
import torchvision.transforms as transforms
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionModelI
from scenic_reasoning.questions.ObjectDetectionQ import ALL_QUESTIONS, Quadrants
from sqlitedict import SqliteDict
from torch.utils.data import Dataset
from tqdm import tqdm


class ObjDectDatasetBuilder(Dataset):
    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "data",
        "databases",
    )

    def __init__(
        self,
        split: list[str] = ["train", "val", "test"],
        db_name: str = "object_detection_questions_from_ground_truth",
    ):
        self.split = split
        self.questions = ALL_QUESTIONS
        self.questions.append(Quadrants(2, 2))
        self.questions.append(Quadrants(2, 3))
        self.questions.append(Quadrants(3, 2))
        self.questions.append(Quadrants(3, 3))

        self.dataset = {}
        db_path = os.path.join(self.DEFAULT_DB_PATH, db_name, ".db")
        if not os.path.exists(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        for question in self.questions:
            table_name = str(question)
            self.dataset[table_name] = SqliteDict(
                str(db_path), tablename=table_name, autocommit=False
            )

        self.bdd = Bdd100kDataset(split=self.split)
        # self.nu_images = NuImagesDataset(split=self.split, size="full")
        if self.split == "val":
            self.waymo = WaymoDataset(split="validation")
        else:
            self.waymo = WaymoDataset(split=self.split + "ing")

        self.all_sets = [
            self.bdd, 
            # self.nu_images, 
            self.waymo
        ]

        db_total = sum(len(self.dataset[str(q)]) for q in self.questions)
        expected_total = sum(len(d) for d in self.all_sets)
        if db_total == expected_total:
            self.has_been_built = True

        self.has_been_built = False

    def __len__(self):
        if not self.has_been_built:
            raise ValueError("Dataset has not been built yet.")

        count = 0
        for d in self.dataset:
            count += len(self.dataset[d])
        return count

    def __getitem__(self, idx):
        if not self.has_been_built:
            raise ValueError("Dataset has not been built yet.")

        for d in self.dataset:
            if idx < len(self.dataset[d]):
                return self.dataset[d][idx]
            idx -= len(self.dataset[d])
        raise IndexError("Index out of range")

    def build(self, model: Optional[ObjectDetectionModelI] = None, batch_size: int = 1):
        if self.has_been_built:
            print("Dataset has already been built.")
            return

        for dataset in tqdm(self.all_sets, desc="Datasets"):
            for i in tqdm(
                range(0, len(dataset), batch_size),
                desc=f"Processing {dataset.__class__.__name__}",
                leave=False,
            ):
                batch_images = dataset[i : i + batch_size]["image"]
                batch_images = [transforms.ToTensor()(img) for img in batch_images]
                batch_names = dataset[i : i + batch_size]["name"]

                if model is not None:
                    labels = model.identify_for_image_as_tensor(
                        torch.stack(batch_images)
                    )
                else:
                    labels = dataset[i : i + batch_size]["labels"]

                for j, lbl in enumerate(labels):
                    for question in self.questions:
                        table_name = str(question)

                        if question.is_applicable(batch_images[j], lbl):
                            qa_list = question.apply(batch_images[j], lbl)
                            # because of Python semantics, sqlitedict cannot 
                            # know when a mutable SqliteDict-backed entry 
                            # was modified in RAM. You'll need to explicitly 
                            # assign the mutated object back to SqliteDict:
                            # https://github.com/piskvorky/sqlitedict
                            self.dataset[table_name][batch_names[j]] = {
                                "questions": qa_list,
                                "split": self.split,
                                "num of labels": len(lbl),
                            }
                        else:
                            self.dataset[table_name][batch_names[j]] = {
                                "questions": "Question not applicable",
                                "split": self.split,
                                "num of labels": len(lbl),
                            }

                    for table_name in self.dataset:
                        self.dataset[table_name].commit()

        self.has_been_built = True
