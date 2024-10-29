from itertools import islice
from typing import Iterator, List, Union

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from scenic_reasoning.src.scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format, ObjectDetectionResult)


class Yolo:
    def __init__(self, download_path, task : str = None, verbose : bool = False):
        self._model = YOLO(download_path, task=task, verbose=verbose)

    def identify_for_image(self, image):
        results = self._model(image)

        if len(results) == 0:
            return None

        results = []
        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                odr = ObjectDetectionResult(
                    score=box.conf.item(),
                    cls=int(box.cls.item()),
                    label=names[int(box.cls.item())],
                    bbox=box,
                    image_hw=box.orig_shape,
                    bbox_format=BBox_Format.Ultralytics,
                )

                results.append(odr)

        return results

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[List[ObjectDetectionResult]]]:
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        # If video is a list, convert it to an iterator of batches
        if isinstance(video, list):
            video_iterator = batch_iterator(video, batch_size)
        else:
            # If video is already an iterator, create batches from it
            video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break

            images = torch.stack([torch.tensor(np.array(img)) for img in batch])
            batch_results = self._model(images)

            boxes_across_frames = []

            if len(batch_results) == 0:
                boxes_across_frames = [[None] * len(batch)]
            else:
                for frame_result in batch_results:
                    per_frame_results = []

                    boxes = frame_result.boxes
                    names = frame_result.names

                    for box in boxes:
                        odr = ObjectDetectionResult(
                            score=box.conf.item(),
                            cls=int(box.cls.item()),
                            label=names[int(box.cls.item())],
                            bbox=box,
                            image_hw=box.orig_shape,
                            bbox_format=BBox_Format.Ultralytics,
                        )

                        per_frame_results.append(odr)

                    boxes_across_frames.append(per_frame_results)

            yield boxes_across_frames
