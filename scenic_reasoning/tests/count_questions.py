import time

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
    # transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)

niu = NuImagesDataset(
    split="val",
    # transform=lambda i, l: yolo_nuscene_transform(i, l, (768, 1280))
)

waymo = WaymoDataset(
    split="validation",
    # transform=lambda i, l: yolo_waymo_transform(i, l, (640, 1333))
)

datasets = [bdd, niu, waymo]

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

total_count_across_all_datasets = 0
for my_dataset in datasets:
    dataset_name = my_dataset.__class__.__name__
    total_num_questions_for_dataset = 0
    print("Dataset:", dataset_name + my_dataset.split)
    print("\tNumber of images in there:", len(my_dataset))
    start_time = time.time()

    for i in range(len(my_dataset)):
        data = my_dataset[i]
        image = data["image"]
        image = transforms.ToPILImage()(image)
        labels = data["labels"]

        for q in q_list:
            if q.is_applicable(image, labels):
                qa_list = q.apply(image, labels)
                total_num_questions_for_dataset += len(qa_list)

    end_time = time.time()
    print(
        "Total number of questions for",
        dataset_name + ":",
        total_num_questions_for_dataset,
    )
    print("\tTime taken (in seconds):", end_time - start_time)
    total_count_across_all_datasets += total_num_questions_for_dataset

print(
    "Total number of questions across all datasets: ", total_count_across_all_datasets
)
# Total number of questions across all datasets:  193150
