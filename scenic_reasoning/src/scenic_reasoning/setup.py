import argparse
import concurrent.futures
import hashlib
import os
import platform
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

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


def _download_chunk(url, start, end, chunk_index, temp_dir):
    headers = {"Range": f"bytes={start}-{end}"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
    chunk_size = end - start + 1

    with open(chunk_path, "wb") as f, tqdm(
        total=chunk_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading chunk {chunk_index + 1}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    downloaded_chunk_size = os.path.getsize(chunk_path)
    if downloaded_chunk_size != chunk_size:
        raise ValueError(
            f"Chunk {chunk_index} was not downloaded correctly. Expected {chunk_size} bytes, got {downloaded_chunk_size} bytes."
        )

    return chunk_path


def _merge_chunks(temp_dir, dest_path, num_chunks, file_size):
    with open(dest_path, "wb") as dest_file, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Writing"
    ) as pbar:
        for i in range(num_chunks):
            chunk_path = os.path.join(temp_dir, f"chunk_{i}")
            with open(chunk_path, "rb") as chunk_file:
                while True:
                    data = chunk_file.read(8192)
                    if not data:
                        break
                    dest_file.write(data)
                    pbar.update(len(data))
            os.remove(chunk_path)


def _download_file(url, dest_path, num_threads=10):
    response = requests.head(url)
    file_size = int(response.headers["Content-Length"])

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == file_size:
        return True

    chunk_size = (file_size + num_threads - 1) // num_threads

    print(
        f"Downloading {url} to {os.path.abspath(dest_path)} using {num_threads} threads."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    _download_chunk,
                    url,
                    i * chunk_size,
                    min((i + 1) * chunk_size - 1, file_size - 1),
                    i,
                    temp_dir,
                )
                for i in range(num_threads)
            ]
            for future in futures:
                future.result()

        _merge_chunks(temp_dir, dest_path, num_threads, file_size)

    print(f"Download completed: {dest_path}")
    return True


def _check_md5(file_path, expected_md5):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5


def cleanup(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def download_and_check_md5(file_url, file_name, md5_url, md5_name):
    if not _download_file(file_url, file_name):
        raise RuntimeError(f"Failed to download file {file_url}.")
    if not _download_file(md5_url, md5_name, num_threads=1):
        cleanup([file_name])
        raise RuntimeError(f"Failed to download MD5 file {md5_url}.")

    with open(md5_name, "r") as f:
        expected_md5 = (
            f.read().strip().split(" ")[0] # Only take the hash, ignore the filename
        )

    if not _check_md5(file_name, expected_md5):
        print(f"MD5 mismatch for {file_name}.")
        cleanup([file_name, md5_name])
        raise RuntimeError("MD5 mismatch for downloaded file.")


def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {os.path.abspath(extract_to)}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def download_bdd(task=None, split=None):
    if task is None or split is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--task",
            "-t",
            type=str,
            choices=["detection", "segmentation"],
            help="Are you interested in object 'detection' or instance 'segmenation' ?",
        )
        parser.add_argument(
            "--split",
            "-s",
            type=str,
            choices=["train", "val", "test", "all"],
            help="Which data split to download",
        )

        args = parser.parse_args()

        task = args.task
        split = args.split

    assert task in ["detection", "segmentation"]
    assert split in ["train", "val", "test", "all"]

    root_dir = PROJECT_DIR
    data_dir = "/data/bdd100k"
    os.chdir(root_dir)
    os.makedirs(root_dir + data_dir, exist_ok=True)
    os.chdir(root_dir + data_dir)

    source_url = "https://dl.cv.ethz.ch/bdd100k/data/"

    # Mapping task/split to specific file URLs and MD5 files
    task_map = {
        "detection": ("100k", "bdd100k_det_20_labels_trainval.zip"),
        "segmentation": ("10k", "bdd100k_ins_seg_labels_trainval.zip"),
    }

    split_prefix, label_file = task_map[task]
    files_to_download = []
    file_splits = ["train", "val", "test"] if split == "all" else [split]

    for split in file_splits:
        zip_file = f"{split_prefix}_images_{split}.zip"
        files_to_download.append(
            (source_url + zip_file, root_dir + data_dir + "/" + zip_file)
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(files_to_download)
    ) as executor:
        futures = [
            executor.submit(
                download_and_check_md5,
                file_url,
                os.path.abspath(file_name),
                file_url + ".md5",
                os.path.abspath(file_name) + ".md5",
            )
            for file_url, file_name in files_to_download
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # label files don't have an md5 so we download them separately
    if split != "test":
        label_file = task_map[task][1]
        if not _download_file(
            source_url + label_file, root_dir + data_dir + "/" + label_file
        ):
            raise RuntimeError(f"Failed to download file {label_file}.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(unzip_file, file_name, root_dir + data_dir)
            for _, file_name in files_to_download
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(cleanup, [file_name, file_name + ".md5"])
            for file_name in files_to_download
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    os.chdir(PROJECT_DIR)
    print("Download and extraction complete.")


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

    subprocess.run(["rmdir", "install"])


def main():
    parser = argparse.ArgumentParser(description="Dataset download utilities")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for download_bdd
    parser_bdd = subparsers.add_parser("download_bdd", help="Download BDD dataset")
    parser_bdd.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["detection", "segmentation"],
        help="Are you interested in object 'detection' or instance 'segmentation'?",
    )
    parser_bdd.add_argument(
        "--split",
        "-s",
        type=str,
        choices=["train", "val", "test", "all"],
        help="Which data split to download",
    )

    # Subcommand for download_nuscenes
    parser_nuscenes = subparsers.add_parser(
        "download_nuscenes", help="Download NuScenes dataset"
    )
    # Add arguments specific to download_nuscenes if needed
    parser_nuscenes.add_argument(
        "--size",
        "-s",
        type=str,
        choices=["mini", "full"],
        help="Which data split to download",
    )

    args = parser.parse_args()

    if args.command == "download_bdd":
        download_bdd(args.task, args.split)
    elif args.command == "download_nuscenes":
        download_nuscenes()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
