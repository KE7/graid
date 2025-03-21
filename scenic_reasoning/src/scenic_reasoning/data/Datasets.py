import json
import os
from pathlib import Path
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import numpy as np
from scenic_reasoning.utilities.common import project_root_dir
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


lock = threading.Lock()

class ObjDectDatasetBuilder(Dataset):
    DEFAULT_DB_PATH = project_root_dir() / "data" / "databases2"

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

        def process_batch(base_idx, batch):
            batch_images = torch.stack([sample["image"] for sample in batch])
            batch_names = [sample["path"] for sample in batch]

            if model is not None:
                labels = model.identify_for_image(batch_images)
                if labels == [None]:
                    return
            else:
                labels = [sample["labels"] for sample in batch]

            tables_accessed = set()

            for j, lbl in enumerate(labels):
                image = batch_images[j].permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))
                # image.show()  # for debugging
                name = f"{int(base_idx + j)}.pkl"
                print(f"Processing {name}...")
                for question in self.questions:
                    table_name = str(question)
                    tables_accessed.add(table_name)
                    if question.is_applicable(image, lbl):
                        qa_list = question.apply(image, lbl)
                    else:
                        qa_list = []

                    entry = {
                        "qa_list": qa_list,
                        "split": self.split,
                        "num of labels": len(lbl),
                    }

                    
                    # with self.lock:
                    self.dataset[table_name][name] = entry
            
            print("Batch done!!!")
            for table_name in tables_accessed:
                self.dataset[table_name].commit()  # commit after processing each batch

                    
        if self.has_been_built:
            print("Dataset has already been built.")
            return

        for dataset in self.all_sets:
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=lambda x: x, 
                num_workers=8
            )

            max_workers = 2
            inflight_futures = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                data_iter = iter(tqdm(enumerate(data_loader)))

                # Start with an initial batch of tasks (fill the pool)
                for _ in range(max_workers):
                    try:
                        base_idx, batch = next(data_iter)
                        inflight_futures.append(
                            executor.submit(process_batch, base_idx * batch_size, batch)
                        )
                    except StopIteration:
                        break

                # As futures complete, submit new tasks
                while inflight_futures:
                    for future in as_completed(inflight_futures):
                        inflight_futures.remove(future)
                        future.result()
                        try:
                            base_idx, batch = next(data_iter)
                            inflight_futures.append(
                                executor.submit(process_batch, base_idx * batch_size, batch)
                            )
                        except StopIteration:
                            continue

            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     executor.map(process_batch, tqdm(data_loader))



            # for batch in tqdm(data_loader, desc="generating dataset..."):

            #     batch_images = torch.stack([sample["image"] for sample in batch])
            #     batch_names = [sample["path"] for sample in batch]   # using path would simpler

            #     if model is not None:
            #         # labels = model.identify_for_image_batch(batch_images)
            #         labels = model.identify_for_image(batch_images)
            #         if labels == [None]:
            #             continue
            #     else:
            #         labels = [sample["labels"] for sample in batch]

            #     for j, lbl in enumerate(labels):
            #         image = batch_images[j].permute(1, 2, 0).cpu().numpy()
            #         image = Image.fromarray(image.astype(np.uint8))
            #         name = batch_names[j]
            #         for question in self.questions:
            #             table_name = str(question)

            #             if question.is_applicable(image, lbl):
            #                 qa_list = question.apply(image, lbl)
            #                 # because of Python semantics, sqlitedict cannot
            #                 # know when a mutable SqliteDict-backed entry
            #                 # was modified in RAM. You'll need to explicitly
            #                 # assign the mutated object back to SqliteDict:
            #                 # https://github.com/piskvorky/sqlitedict
            #                 self.dataset[table_name][name] = {
            #                     "qa_list": qa_list,
            #                     "split": self.split,
            #                     "num of labels": len(lbl),
            #                 }
            #             else:
            #                 self.dataset[table_name][name] = {
            #                     "qa_list": [],
            #                     "split": self.split,
            #                     "num of labels": len(lbl),
            #                 }

        for table_name in self.dataset:
            if not self.dataset[table_name]:
                continue
            self.dataset[table_name].close()

        self.has_been_built = True
