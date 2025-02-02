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

# data = bdd[2]
# image = data['image']
# image = transforms.ToPILImage()(image)
# labels = data['labels']
# timestamp = data['timestamp']

q_list = [Quadrants(2, 2), MostAppearance(), IsObjectCentered(), WidthVsHeight(), LargestAppearance(), LeastAppearance(), LeftOf(), RightOf(), LeftMost(), RightMost(), HowMany()]
for i in range(10):
    data = bdd[i]
    image = data['image']
    image = transforms.ToPILImage()(image)
    labels = data['labels']
    for q in q_list:    
        if q.is_applicable(image, labels):
            qa_list = q.apply(image, labels)
            print(q, "Passed", qa_list)
        else:
            print(q, "Not applicable")

    print("==================================================================")

