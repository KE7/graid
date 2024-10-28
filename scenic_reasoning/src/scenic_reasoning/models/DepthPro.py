
import torch
import depth_pro
from PIL import ImageSequence
from scenic_reasoning.src.scenic_reasoning.interfaces.DepthPerceptionI import DepthPerceptionI
from scenic_reasoning.src.scenic_reasoning.utilities.common import get_default_device, project_root_dir
from typing import List, override


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


class DepthProV():

    def __init__(self, video : ImageSequence, batch_size : int, **kwargs):
        model_path = kwargs.get('model_path', project_root_dir() / 'checkpoints' / 'depth_pro.pt')
        depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = model_path

        self.device = kwargs.get('device', get_default_device())
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.model.eval()

        self._depth_map : List[DepthPerceptionI] = []

        # split the video into batches
        for i in range(0, len(video), batch_size):
            batch = video[i:i+batch_size]
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
                    compute_preds=False
                )
                self._depth_map.append(depth_perception)
            
    def get_depth_map(self):
        return self._depth_map
    
    def get_depth_map_at_index(self, index : int):
        return self._depth_map[index] if index < len(self._depth_map) else None          
            