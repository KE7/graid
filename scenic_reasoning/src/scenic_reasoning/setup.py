import os
import platform
import subprocess

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def install_depth_pro():
    root_dir = PROJECT_DIR

    os.chdir(root_dir)

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


def install_detectron2():
    # https://detectron2.readthedocs.io/en/latest/tutorials/install.html

    root_dir = PROJECT_DIR

    os.chdir(root_dir)

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


def install_all():
    install_depth_pro()
    install_detectron2()
