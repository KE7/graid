from itertools import islice
from typing import Iterator, List, Union, override

import depth_pro
import torch
from PIL import Image, ImageSequence
from scenic_reasoning.interfaces.DepthPerceptionI import (
    DepthPerceptionI,
    DepthPerceptionResult,
)
from scenic_reasoning.utilities.common import get_default_device, project_root_dir


class DepthPro(DepthPerceptionI):
    def __init__(self, **kwargs):
        model_path = kwargs.get(
            "model_path", project_root_dir() / "checkpoints" / "depth_pro.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}",
                f"Please follow the project's readme to install all components.",
            )

        depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = model_path

        self.device = kwargs.get("device", get_default_device())
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.model.eval()

        self._prediction = None
        self._depth_prediction = None
        self._focallength_px = None
        self._depth_map = None

    @override
    def predict_depth(self, image):
        image, _, f_px = depth_pro.load_rgb(image)
        image = self.transform(image)
        prediction = self.model.infer(image, f_px=f_px)
        depth_prediction = prediction["depth"]
        focallength_px = prediction["focallength_px"]

        result = DepthPerceptionResult(
            depth_prediction=depth_prediction,
            focallength_px=focallength_px,
        )
        return result

    @override
    def predict_depths(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[DepthPerceptionResult]:
        """
        Predicts the depth of each frame in the input video.
        Note: The video must be a list of PIL images
            In this way, we force the callers to do any preprocessing they need.
            For example, skipping frames to reduce computation time.

        Args:
            video: An iterator or list of PIL images
            batch_size: The number of frames to predict in one forward pass

        Yields:
            An iterator of batches of DepthPerceptionResult objects
        """

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
            images, f_px_list = [], []
            for img in batch:
                img, _, f_px = depth_pro.load_rgb(img)
                img = self.transform(img)
                images.append(img)
                f_px_list.append(f_px)

            images = torch.stack(images)
            f_px_list = torch.stack(f_px_list)

            predictions = self.model.infer(
                images, f_px=f_px_list
            )  # tensor of shape (batch_size, 1, H, W)
            batch_results = []

            for j in range(predictions.shape[0]):
                depth_perception = DepthPerceptionI(
                    image=batch[j],
                    prediction=predictions[j],
                    focallength_px=f_px_list[j],
                )
                batch_results.append(depth_perception)
            yield batch_results


class DepthProV:
    # TODO: ImageSequence is the wrong type. Should be list of PIL images but requires
    #      fixing the for loop as well
    def __init__(self, video: ImageSequence, batch_size: int, **kwargs):
        model_path = kwargs.get(
            "model_path", project_root_dir() / "checkpoints" / "depth_pro.pt"
        )
        depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = model_path

        self.device = kwargs.get("device", get_default_device())
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.model.eval()

        self._depth_map: List[DepthPerceptionI] = []

        # split the video into batches
        for i in range(0, len(video), batch_size):
            batch = video[i : i + batch_size]
            images, f_px_list = [], []
            for img in batch:
                img, _, f_px = depth_pro.load_rgb(img)
                img = self.transform(img)
                images.append(img)
                f_px_list.append(f_px)

            images = torch.stack(images)
            f_px_list = torch.stack(f_px_list)

            predictions = self.model.infer(images, f_px=f_px_list)

            for j in range(predictions.shape[0]):
                depth_perception = DepthPerceptionI(
                    image=batch[j],
                    depth_prediction=predictions[j]["depth"],
                    focallength_px=predictions[j]["focallength_px"],
                )
                self._depth_map.append(depth_perception)
