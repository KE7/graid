from abc import abstractmethod
import PIL
from PIL.Image import Image
from matplotlib import pyplot as plt
import numpy as np

class DepthPerceptionI():

    def __init__(self, image, depth_prediction=None, focallength_px=None):
        self.image = image
        self.channels, self.height, self.width = self.image.shape
        if not self.channels <= 3:
            raise ValueError("Image provided in wrong format. Please ensure channels are first and are in RGB or gray scale.")
        self._depth_prediction = depth_prediction
        self._focallength_px = focallength_px

        compute_preds = depth_prediction is None and focallength_px is None
        if compute_preds:
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


    def visualize_inverse_depth(self) -> Image:
        """
        The following code is copied from Apple's ML Depth Pro
        """
        if self._depth_prediction.get_device() != "cpu":
            original_device = self._depth_prediction.get_device()
            depth = np.copy.deepcopy(self._depth_prediction.cpu()) # avoid cuda oom errors
            self._depth_prediction.to(original_device)
        else:
            depth = np.copy.deepcopy(self._depth_prediction)
        
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