import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionModelI
from scenic_reasoning.questions.ObjectDetectionQ import ALL_QUESTIONS, Quadrants
from sqlitedict import SqliteDict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ObjDectDatasetBuilder(Dataset):
    DEFAULT_DB_PATH = Path(__file__).resolve().parent / "databases"

    def __init__(
        self,
        split: list[str] = ["train", "val", "test"],
        dataset: list[str] = ["waymo", "nuimage", "bdd", "all"],
        db_name: str = "object_detection_questions_from_gt",
        transform=None,
    ):
        self.split = split
        self.questions = ALL_QUESTIONS
        self.questions.append(Quadrants(2, 2))
        self.questions.append(Quadrants(2, 3))
        self.questions.append(Quadrants(3, 2))
        self.questions.append(Quadrants(3, 3))

        self.dataset = {}
        db_path = self.DEFAULT_DB_PATH / f"{db_name}.sqlite"
        print("DB path: ", db_path)
        if not os.path.exists(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        for question in self.questions:
            table_name = str(question)
            self.dataset[table_name] = SqliteDict(
                str(db_path),
                tablename=table_name,
                autocommit=True,
                encode=json.dumps,
                decode=json.loads,
            )
            self.dataset[table_name].commit()

        self.all_sets = []
        if dataset == "bdd":
            self.bdd = Bdd100kDataset(split=self.split, transform=transform)
            self.all_sets.append(self.bdd)
        elif dataset == "nuimage":
            self.nu_images = NuImagesDataset(
                split=self.split, size="all", transform=transform
            )
            self.all_sets.append(self.nu_images)
        elif dataset == "waymo":
            if self.split == "val":
                self.waymo = WaymoDataset(split="validation", transform=transform)
            else:
                self.waymo = WaymoDataset(split=self.split + "ing", transform=transform)

            self.all_sets.append(self.waymo)
        else:
            print("invalid dataset combination")

        db_total = sum(len(self.dataset[str(q)]) for q in self.questions)
        expected_total = sum(len(d) for d in self.all_sets)
        if db_total == expected_total:
            self.has_been_built = True

        self.has_been_built = False
    
    def is_built(self):
        return self.has_been_built

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

        for dataset in self.all_sets:
            print("Generating dataset...")

            
            data_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
            )
            for batch in tqdm(data_loader, desc="generating dataset..."):

                batch_images = torch.stack([sample["image"] for sample in batch])
                batch_names = [sample["path"] for sample in batch]   # using path would simpler

                if model is not None:
                    # labels = model.identify_for_image_as_tensor(batch_images)
                    labels = model.identify_for_image(batch_images)
                    if labels == [None]:
                        continue
                else:
                    labels = [sample["labels"] for sample in batch]

                for j, lbl in enumerate(labels):
                    image = batch_images[j].permute(1, 2, 0).numpy()
                    image = Image.fromarray(image.astype(np.uint8))
                    name = batch_names[j]
                    for question in self.questions:
                        table_name = str(question)

                        if question.is_applicable(image, lbl):
                            qa_list = question.apply(image, lbl)
                            # because of Python semantics, sqlitedict cannot
                            # know when a mutable SqliteDict-backed entry
                            # was modified in RAM. You'll need to explicitly
                            # assign the mutated object back to SqliteDict:
                            # https://github.com/piskvorky/sqlitedict
                            self.dataset[table_name][name] = {
                                "qa_list": qa_list,
                                "split": self.split,
                                "num of labels": len(lbl),
                            }
                        else:
                            self.dataset[table_name][name] = {
                                "qa_list": "Question not applicable",
                                "split": self.split,
                                "num of labels": len(lbl),
                            }

        for table_name in self.dataset:
            if not self.dataset[table_name]:
                continue
            self.dataset[table_name].close()

        self.has_been_built = True
