from scenic_reasoning.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from scenic_reasoning.utilities.common import get_default_device, yolo_transform
from scenic_reasoning.questions.ObjectDetectionQ import MostAppearance, IsObjectCentered, WidthVsHeight, Quadrants, LargestAppearance, LeastAppearance, LeftOf, RightOf, LeftMost, RightMost, HowMany
import torchvision.transforms as transforms
NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1


bdd = Bdd100kDataset(
    split="val", 
    transform=yolo_transform,  
    use_original_categories=False,
    use_extended_annotations=False,
)

niu = NuImagesDataset(split='test', transform=yolo_transform)

waymo = WaymoDataset(split="validation", transform=yolo_transform)

my_dataset = waymo

q_list = [Quadrants(2, 2), MostAppearance(), IsObjectCentered(), WidthVsHeight(), LargestAppearance(), LeastAppearance(), LeftOf(), RightOf(), LeftMost(), RightMost(), HowMany()]
for i in range(10):
    print(i)
    data = my_dataset[i]
    image = data['image']
    image = transforms.ToPILImage()(image)
    labels = data['labels']
    for q in q_list:    
        if q.is_applicable(image, labels):
            qa_list = q.apply(image, labels)
            print(q, qa_list, "Passed")
        else:
            print(q, "Not applicable")

    print("==================================================================")

