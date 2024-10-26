from typing import override
from scenic_reasoning.src.scenic_reasoning.interfaces import DepthPerceptionI
import depth_pro
from scenic_reasoning.src.scenic_reasoning.utilities.common import get_default_device, project_root_dir


class DepthPro(DepthPerceptionI):
    def __init__(self, image, **kwargs):
        model_path = kwargs.get('model_path', project_root_dir() / 'checkpoints' / 'depth_pro.pt')
        depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = model_path

        self.device = kwargs.get('device', get_default_device())
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.model.eval()

        super().__init__(image)

    @override
    def __init_depth_prediction__(self):
        image, _, f_px = depth_pro.load_rgb(self.image)
        image = self.transform(image)
        self._prediction = self.model.infer(image, f_px=f_px)
        self._depth_prediction = self._prediction["depth"]
        self._focallength_px = self._prediction["focallength_px"]
