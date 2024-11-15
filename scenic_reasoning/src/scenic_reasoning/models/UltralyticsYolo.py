from itertools import islice
from typing import Iterator, List, Union

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)


class Yolo(ObjectDetectionModelI):
    def __init__(self, model, task: str = None, verbose: bool = False):
        self._model = YOLO(model, task=task, verbose=verbose)

    def identify_for_image(self, image, debug = False, **kwargs) -> List[List[ObjectDetectionResultI]]:
        """
        Run object detection on an image or a batch of images.

        Args:
            image: either a PIL image or a tensor of shape (B, C, H, W)
                where B is the batch size, C is the channel size, H is the
                height, and W is the width.

        Returns:
            A list of list of ObjectDetectionResultI, where the outer list
            represents the batch of images, and the inner list represents the
            detections in a particular image.
        """
        predictions = self._model.predict(image, **kwargs)

        if len(predictions) == 0:
            return None

        formatted_results = []
        for y_hat in predictions:
            result_for_image = []
            boxes = y_hat.boxes
            names = y_hat.names

            if debug:
                y_hat.show()

            for box in boxes:
                odr = ObjectDetectionResultI(
                    score=box.conf.item(),
                    cls=int(box.cls.item()),
                    label=names[int(box.cls.item())],
                    bbox=box.cpu(),
                    image_hw=box.orig_shape,
                    bbox_format=BBox_Format.UltralyticsBox,
                )

                result_for_image.append(odr)

            formatted_results.append(result_for_image)

        return formatted_results

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[List[ObjectDetectionResultI]]]:
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
                        odr = ObjectDetectionResultI(
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
