import argparse
import os
import platform
import subprocess

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def install_depth_pro():
    root_dir = PROJECT_DIR

    os.chdir(root_dir)
    if not os.path.exists("install"):
        os.makedirs("install")

    os.chdir("install")

    # Check if the repository already exists in the root
    if not os.path.exists("ml-depth-pro"):
        subprocess.run(["git", "clone", "https://github.com/apple/ml-depth-pro"])

    # Change to the ml-depth-pro directory
    os.chdir("ml-depth-pro")

    # Install the package in editable mode
    subprocess.run(["pip", "install", "-e", "."])

    # Change back to the original directory
    os.chdir("..")

    # Check if the checkpoints folder exists and contains depth_pro.pt
    if not (
        os.path.exists("checkpoints") and os.path.isfile("checkpoints/depth_pro.pt")
    ):
        # Run the get_pretrained_models.sh script
        subprocess.run(["bash", "ml-depth-pro/get_pretrained_models.sh"])


def clean_depth_pro():
    root_dir = PROJECT_DIR

    os.chdir(root_dir)
    if not os.path.exists("install"):
        os.makedirs("install")

    os.chdir("install")

    # Check if the repository already exists in the root
    if os.path.exists("ml-depth-pro"):
        subprocess.run(["rm", "-rf", "ml-depth-pro"])

    # Change back to the original directory
    os.chdir("..")


def install_detectron2():
    # https://detectron2.readthedocs.io/en/latest/tutorials/install.html

    root_dir = PROJECT_DIR

    os.chdir(root_dir)
    if not os.path.exists("install"):
        os.makedirs("install")

    os.chdir("install")

    # Check if the repository already exists in the root
    if not os.path.exists("detectron2"):
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/detectron2.git"]
        )

    # Change to the detectron2 directory
    os.chdir("detectron2")

    # clean up the build directory in case the cloned repo was already there
    subprocess.run(["rm", "-rf", "build/", "**/*.so"])

    # if on macOS
    if platform.system() == "Darwin":
        subprocess.run(
            [
                'CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python',
                "-m",
                "pip",
                "install",
                "-e",
                ".",
            ]
        )
    else:
        subprocess.run(["python", "-m", "pip", "install", "-e", "."])

    # Change back to the original directory
    os.chdir("..")


def clean_detectron2():
    root_dir = PROJECT_DIR

    os.chdir(root_dir)
    if not os.path.exists("install"):
        return

    os.chdir("install")

    # Check if the repository already exists in the root
    if os.path.exists("detectron2"):
        subprocess.run(["rm", "-rf", "detectron2"])

    # Change back to the original directory
    os.chdir("..")


def install_mmdetection():
    subprocess.run(["pip", "install", "--upgrade", "openmim"])
    subprocess.run(["mim", "install", "mmengine"])
    subprocess.run(["mim", "install", "mmcv==2.1.0"])

    root_dir = PROJECT_DIR

    os.chdir(root_dir)

    if not os.path.exists("install"):
        os.makedirs("install")

    os.chdir("install")

    # Check if the repository already exists in the root
    if not os.path.exists("mmdetection"):
        subprocess.run(
            ["git", "clone", "https://github.com/open-mmlab/mmdetection.git"]
        )

    # Change to the mmdetection directory
    os.chdir("mmdetection")

    subprocess.run(["pip", "install", "-e", "."])

    # Change back to the original directory
    os.chdir("..")


def clean_mmdetection():
    root_dir = PROJECT_DIR

    os.chdir(root_dir)
    if not os.path.exists("install"):
        return

    os.chdir("install")

    # Check if the repository already exists in the root
    if os.path.exists("mmdetection"):
        subprocess.run(["rm", "-rf", "mmdetection"])

    # Change back to the original directory
    os.chdir("..")


def download_nuscenes():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=str, default="mini", help="mini or full", required=True
    )

    args = parser.parse_args()
    size = args.size
    
    root_dir = PROJECT_DIR

    os.chdir(root_dir)

    subprocess.run(["mkdir", "-p", "data/nuscenes"])

    if size == "mini":
            # !wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.

            # !tar -xf v1.0-mini.tgz -C /data/sets/nuscenes  # Uncompress the nuScenes mini split.

            # !pip install nuscenes-devkit &> /dev/null  # Install nuScenes.
        location = "https://www.nuscenes.org/data/v1.0-mini.tgz"
        if not subprocess.run(["wget", location]).returncode == 0:
            raise OSError("wget is not installed on your system.")

        subprocess.run(["tar", "-xf", "v1.0-mini.tgz", "-C", "data/nuscenes"])

    elif size == "full":
        raise NotImplementedError
    

    subprocess.run(["pip", "install", "nuscenes-devkit"])


def install_all():
    install_depth_pro()
    install_detectron2()
    install_mmdetection()


def clean_all():
    clean_depth_pro()
    clean_detectron2()
    clean_mmdetection()