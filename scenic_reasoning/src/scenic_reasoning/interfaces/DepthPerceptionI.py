import copy
from abc import abstractmethod
from typing import Iterator

import numpy as np
import PIL
import PIL.Image
from matplotlib import pyplot as plt
from PIL.Image import Image
from torch import Tensor


class DepthPerceptionResult:

    def __init__(self, depth_prediction: Tensor, focallength_px):
        self.depth_prediction = depth_prediction
        self.focallength_px = focallength_px


class DepthPerceptionI:

    @abstractmethod
    def __init__(self):
        """
        Initialize the depth perception model.
        """

    @abstractmethod
    def predict_depth(self, image) -> DepthPerceptionResult:
        """
        Predict the depth of the input image.
        """

    @abstractmethod
    def predict_depths(self, video) -> Iterator[DepthPerceptionResult]:
        """
        Predict the depth of each frame in the input video.
        """

    @staticmethod
    def visualize_inverse_depth(dpr: DepthPerceptionResult) -> Image:
        """
        The following code is copied from Apple's ML Depth Pro
        """
        depth = dpr.depth_prediction

        if depth.get_device() != "cpu":
            original_device = depth.get_device()
            depth = copy.deepcopy(depth.cpu())  # avoid cuda oom errors
            depth.to(original_device)
        else:
            depth = copy.deepcopy(depth)

        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth,
        #   clipped to [0.1m;250m] range for better visualization
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

        return PIL.Image.fromarray(color_depth)
