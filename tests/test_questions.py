import torchvision.transforms as transforms
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.questions.ObjectDetectionQ import (
    HowMany,
    IsObjectCentered,
    LargestAppearance,
    LeastAppearance,
    LeftMost,
    LeftOf,
    MostAppearance,
    Quadrants,
    RightMost,
    RightOf,
    WidthVsHeight,
)
from scenic_reasoning.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1


bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)

niu = NuImagesDataset(
    split="test", transform=lambda i, l: yolo_nuscene_transform(i, l, (768, 1280))
)

waymo = WaymoDataset(
    split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, stride=32)
)

my_dataset = bdd

q_list = [
    Quadrants(2, 2),
    MostAppearance(),
    IsObjectCentered(),
    WidthVsHeight(),
    LargestAppearance(),
    LeastAppearance(),
    LeftOf(),
    RightOf(),
    LeftMost(),
    RightMost(),
    HowMany(),
]
for i in range(10):
    print(i)
    data = my_dataset[i]
    image = data["image"]
    image = transforms.ToPILImage()(image)
    labels = data["labels"]
    at_least_one_was_applicable = False
    for q in q_list:
        if q.is_applicable(image, labels):
            qa_list = q.apply(image, labels)
            if len(qa_list) == 0:

                print(str(q) + "\tIs applicable but no questions\n")
                continue

            print(str(q) + "\t" + "Passed")
            print("[\n" + "\n".join(["\t" + str(item) for item in qa_list]) + "\n]\n")
            at_least_one_was_applicable = True
        else:
            print(q, "Not applicable")
    if at_least_one_was_applicable:
        ObjectDetectionUtils.show_image_with_detections(image, labels)

    print("==================================================================")
