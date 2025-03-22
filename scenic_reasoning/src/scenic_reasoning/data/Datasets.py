import gc
import json
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from scenic_reasoning.utilities.common import project_root_dir
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

        self.writer_queue = queue.Queue()

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
                autocommit=False,
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

        # def build_for_question(question, data_iter):
        #     table_name = str(question)
        #     data_iter = enumerate(data_iter)
        #     while True:
        #         try:
        #             i, sample_batch = next(data_iter)
        #             print(f"Current index: {i} inside {table_name}")
        #         except StopIteration:
        #             break

        #         items = dict()
        #         for sample in sample_batch:
        #             if model is not None:
        #                 labels = model.identify_for_image(sample["image"])
        #                 if labels == [None]:
        #                     continue
        #             else:
        #                 labels = sample["labels"]

        #             image = sample["image"].permute(1, 2, 0).cpu().numpy()
        #             image = Image.fromarray(image.astype(np.uint8))
        #             name = sample["path"]
        #             lbl = sample["labels"]

        #             if question.is_applicable(image, lbl):
        #                 qa_list = question.apply(image, lbl)
        #             else:
        #                 qa_list = []

        #             entry = {
        #                 "qa_list": qa_list,
        #                 "split": self.split,
        #                 "num of labels": len(lbl),
        #             }

        #             items[name] = entry

        #         self.dataset[table_name].update(items)  # update the table with the new entries
        #         print(f"Processed {len(items)} items for question: {question}")
        #         self.dataset[table_name].commit()
        #         del sample_batch
        #         del items
        #         gc.collect()  # free up memory

        def buffered_writer():
            buffer = defaultdict(dict)
            count = 0
            while True:
                items = self.writer_queue.get()
                if items is None:
                    break
                for question in self.questions:
                    table_name = str(question)
                    buffer[table_name].update(items[table_name])
                    buffer[table_name].update(items)

                count += len(items)
                if count >= 1000 or self.writer_queue.qsize() == 0:
                    print(f"Writing {count} items to the database...")
                    for question in self.questions:
                        table_name = str(question)
                        self.dataset[table_name].update(buffer[table_name])
                        self.dataset[table_name].commit()
                        buffer[table_name].clear()
                    count = 0

            for table_name in self.dataset:
                if buffer[table_name]:
                    self.dataset[table_name].update(buffer[table_name])
                    buffer[table_name].clear()
                    self.dataset[table_name].commit()

            self.writer_queue.task_done()
            print("Writer thread exiting...")

        def writer():
            while True:
                items = self.writer_queue.get()
                if items is None:
                    break
                print("Writing items to the database...")
                for question in self.questions:
                    table_name = str(question)
                    self.dataset[table_name].update(items[table_name])
                    self.dataset[table_name].commit()
                self.writer_queue.task_done()
            print("Writer thread exiting...")

        def process_batch(base_idx, batch):
            batch_images = torch.stack([sample["image"] for sample in batch])

            if model is not None:
                labels = model.identify_for_image(batch_images)
                if labels == [None]:
                    return
            else:
                labels = [sample["labels"] for sample in batch]

            items = defaultdict(dict)
            for j, lbl in enumerate(labels):
                image = batch_images[j].permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))
                # image.show()  # for debugging
                name = f"{int(base_idx + j)}.pkl"
                print(f"Processing {name}...")
                for question in self.questions:
                    table_name = str(question)
                    if question.is_applicable(image, lbl):
                        qa_list = question.apply(image, lbl)
                    else:
                        qa_list = []

                    entry = {
                        "qa_list": qa_list,
                        "split": self.split,
                        "num of labels": len(lbl),
                    }

                    items[table_name][name] = entry

            self.writer_queue.put(items)
            # for question in self.questions:
            #     table_name = str(question)
            #     self.dataset[table_name].update(items)
            #     self.dataset[table_name].commit()
            print("Batch done!!!")

        if self.has_been_built:
            print("Dataset has already been built.")
            return

        # for dataset in self.all_sets:
        #     data_loader = DataLoader(
        #         dataset,
        #         batch_size=batch_size,
        #         shuffle=False,
        #         collate_fn=lambda x: x,
        #         num_workers=4
        #     )

        #     max_workers = len(self.questions)
        #     inflight_futures = []
        #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #         data_iter = iter(data_loader)
        #         for w in range(max_workers):
        #             inflight_futures.append(
        #                 executor.submit(build_for_question, self.questions[w], data_iter)
        #             )
        #         for future in as_completed(inflight_futures):
        #             inflight_futures.remove(future)
        #             future.result()

        for dataset in self.all_sets:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: x,
                num_workers=8,
            )

            max_workers = 50
            inflight_futures = []

            writer_thread = threading.Thread(target=buffered_writer, daemon=True)
            writer_thread.start()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                data_iter = iter(tqdm(enumerate(data_loader)))

                for _ in range(max_workers):
                    try:
                        base_idx, batch = next(data_iter)
                        inflight_futures.append(
                            executor.submit(process_batch, base_idx * batch_size, batch)
                        )
                    except StopIteration:
                        break

                while inflight_futures:
                    for future in as_completed(inflight_futures):
                        inflight_futures.remove(future)
                        future.result()
                        try:
                            base_idx, batch = next(data_iter)
                            inflight_futures.append(
                                executor.submit(
                                    process_batch, base_idx * batch_size, batch
                                )
                            )
                        except StopIteration:
                            continue

            self.writer_queue.put(None)
            writer_thread.join()

        #     # with ThreadPoolExecutor(max_workers=10) as executor:
        #     #     executor.map(process_batch, tqdm(data_loader))

        #     # for batch in tqdm(data_loader, desc="generating dataset..."):

        #     #     batch_images = torch.stack([sample["image"] for sample in batch])
        #     #     batch_names = [sample["path"] for sample in batch]   # using path would simpler

        #     #     if model is not None:
        #     #         # labels = model.identify_for_image_batch(batch_images)
        #     #         labels = model.identify_for_image(batch_images)
        #     #         if labels == [None]:
        #     #             continue
        #     #     else:
        #     #         labels = [sample["labels"] for sample in batch]

        #     #     for j, lbl in enumerate(labels):
        #     #         image = batch_images[j].permute(1, 2, 0).cpu().numpy()
        #     #         image = Image.fromarray(image.astype(np.uint8))
        #     #         name = batch_names[j]
        #     #         for question in self.questions:
        #     #             table_name = str(question)

        #     #             if question.is_applicable(image, lbl):
        #     #                 qa_list = question.apply(image, lbl)
        #     #                 # because of Python semantics, sqlitedict cannot
        #     #                 # know when a mutable SqliteDict-backed entry
        #     #                 # was modified in RAM. You'll need to explicitly
        #     #                 # assign the mutated object back to SqliteDict:
        #     #                 # https://github.com/piskvorky/sqlitedict
        #     #                 self.dataset[table_name][name] = {
        #     #                     "qa_list": qa_list,
        #     #                     "split": self.split,
        #     #                     "num of labels": len(lbl),
        #     #                 }
        #     #             else:
        #     #                 self.dataset[table_name][name] = {
        #     #                     "qa_list": [],
        #     #                     "split": self.split,
        #     #                     "num of labels": len(lbl),
        #     #                 }

        for table_name in self.dataset:
            if not self.dataset[table_name]:
                continue
            self.dataset[table_name].commit()
            self.dataset[table_name].close()

        self.has_been_built = True
