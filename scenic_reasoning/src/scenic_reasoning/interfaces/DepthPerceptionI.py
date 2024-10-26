from abc import ABC, abstractmethod
import PIL
from PIL.Image import Image
from matplotlib import pyplot as plt
import numpy as np
import torch

class DepthPerceptionI(ABC):

    def __init__(self, image):
        self.image = image
        self.channels, self.height, self.width = self.image.shape
        if not self.channels <= 3:
            raise ValueError("Image provided in wrong format. Please ensure channels are first and are in RGB or gray scale.")
        self._depth_prediction = None
        self._focallength_px = None
        self.__init_depth_prediction__()

    @abstractmethod
    def __init_depth_prediction__(self):
        """
        Run the depth prediction model and store the results.
        """

    def get_depth_prediction(self):
        return self._depth_prediction
    
    def get_focallength_px(self):
        return self._focallength_px

    @staticmethod
    def visualize_inverse_depth(input_depth: torch.Tensor) -> Image:
        """
        The following code is copied from Apple's ML Depth Pro
        """
        if input_depth.get_device() != "cpu":
            original_device = input_depth.get_device()
            depth = np.copy.deepcopy(input_depth.cpu()) # avoid cuda oom errors
            input_depth.to(original_device)
        else:
            depth = np.copy.deepcopy(input_depth)
        
        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
            np.uint8
        )

        return PIL.Image.fromarray(color_depth)