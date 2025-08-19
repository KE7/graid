from itertools import islice
from typing import Iterator, List, Union

import depth_pro
import torch
from PIL import Image
from graid.interfaces.DepthPerceptionI import (
    DepthPerceptionI,
    DepthPerceptionResult,
)
from graid.utilities.common import get_default_device, project_root_dir


class DepthPro(DepthPerceptionI):
    def __init__(self, **kwargs) -> None:
        model_path = kwargs.get(
            "model_path", project_root_dir() / "checkpoints" / "depth_pro.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}. "
                "Please follow the project's readme to install all components."
            )

        depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = model_path

        self.device = kwargs.get("device", get_default_device())
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.model.eval()

    def predict_depth(self, image: Image.Image) -> DepthPerceptionResult:
        """
        Predict depth for a single image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            DepthPerceptionResult containing depth prediction and focal length
        """
        # Convert PIL Image to numpy array for direct processing
        # (bypassing depth_pro.load_rgb which expects file paths)
        import numpy as np
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply transform directly to the image array
        # depth_pro.load_rgb normally returns (image_array, icc_profile, f_px)
        # We'll set f_px to None and let the model estimate it
        image_tensor = self.transform(image_array)
        f_px = None  # Let the model estimate focal length
        
        prediction = self.model.infer(image_tensor, f_px=f_px)
        depth_prediction = prediction["depth"]
        focallength_px = prediction["focallength_px"]

        result = DepthPerceptionResult(
            depth_prediction=depth_prediction,
            focallength_px=focallength_px,
        )
        return result

    def predict_depths(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[DepthPerceptionResult]:
        """
        Predicts the depth of each frame in the input video.
        
        Args:
            video: An iterator or list of PIL images
            batch_size: The number of frames to predict in one forward pass

        Yields:
            Iterator of DepthPerceptionResult objects (one per frame)
        """

        def _batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        # Convert to batch iterator regardless of input type
        video_iterator = _batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break
                
            images, f_px_list = [], []
            for img in batch:
                img_tensor, _, f_px = depth_pro.load_rgb(img)
                img_tensor = self.transform(img_tensor)
                images.append(img_tensor)
                f_px_list.append(f_px)

            images_batch = torch.stack(images)
            f_px_batch = torch.stack(f_px_list)

            predictions = self.model.infer(images_batch, f_px=f_px_batch)
            
            # Extract individual results from batch
            depth_batch = predictions["depth"]  # shape: (batch_size, H, W)
            focallength_batch = predictions["focallength_px"]  # shape: (batch_size,)

            for j in range(depth_batch.shape[0]):
                result = DepthPerceptionResult(
                    depth_prediction=depth_batch[j],
                    focallength_px=focallength_batch[j],
                )
                yield result

    def to(self, device: Union[str, torch.device]) -> "DepthPro":
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self


