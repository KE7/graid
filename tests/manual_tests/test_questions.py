import logging

import torchvision.transforms as transforms

from graid.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from graid.interfaces.ObjectDetectionI import ObjectDetectionUtils
from graid.questions.ObjectDetectionQ import (
    AreMore,
    HowMany,
    IsObjectCentered,
    LargestAppearance,
    LeastAppearance,
    LeftMost,
    LeftMostWidthVsHeight,
    LeftOf,
    MostAppearance,
    MostClusteredObjects,
    ObjectsInLine,
    ObjectsInRow,
    Quadrants,
    RightMost,
    RightMostWidthVsHeight,
    RightOf,
    WhichMore,
    WidthVsHeight,
)
from graid.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)

# Configure logging
logging.basicConfig(
    filename="test_questions.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1


bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)
bdd_original = Bdd100kDataset(
    split="val",
    use_original_categories=True,
    use_extended_annotations=False,
)

niu = NuImagesDataset(
    split="val",
    transform=lambda i, l: yolo_nuscene_transform(i, l, (768, 1280)),
    size="mini",
)
niu_original = NuImagesDataset(
    split="val",
    size="mini",
)

# waymo = WaymoDataset(
#     split="validation",
#     # TODO: I think yolo_waymo_transform is broken now
#     # transform=lambda i, l: yolo_waymo_transform(i, l, (640, 1333))
# )
# waymo_original = WaymoDataset(
#     split="validation",
# )

my_dataset = bdd
original_dataset = bdd_original

q_list = [WhichMore()]

for i in range(100):
    print(i)
    data = my_dataset[i]
    image = data["image"]
    image = transforms.ToPILImage()(image)
    labels = data["labels"]
    path = data["path"]
    print(path)
    # let's filter out labels that are really small.
    # say anything with area less than 1000 pixels
    threshold = 500
    print("Num labels before filtering: ", len(labels))
    labels = list(filter(lambda x: x.get_area().item() > threshold, labels))
    print("Num labels after filtering: ", len(labels))
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
        og_data = original_dataset[i]
        og_image = og_data["image"]
        og_image = transforms.ToPILImage()(og_image)
        og_labels = og_data["labels"]

        ObjectDetectionUtils.show_image_with_detections(
            og_image,
            og_labels,
        )
        ObjectDetectionUtils.show_image_with_detections(
            image,
            labels,
        )

    print("==================================================================")
