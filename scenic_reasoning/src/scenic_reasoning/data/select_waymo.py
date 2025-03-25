import json
import os
from scenic_reasoning.utilities.common import project_root_dir
import subprocess
from tqdm import tqdm
import argparse

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy interesting files based on validation json.")
    parser.add_argument('--split', type=str, choices=['validation', 'training'], required=True,
                        help='Choose between validation or training.')
    args = parser.parse_args()
    split = args.split

    file_path = f'{split}_best_frames.json'
    data = load_json(file_path)
    interesting_indices = set([d['index'] for d in data.values()])
    
    directory_path = str(project_root_dir() / f'data/waymo_{split}')
    interesting_directory_path = str(project_root_dir() / f'data/waymo_{split}_interesting')


    for file_name in tqdm(os.listdir(directory_path), desc="Processing files"):
        file_id = int(file_name.split('.')[0])
        if file_id in interesting_indices:
            print(f"Found: {file_name}")
            file_path = os.path.join(directory_path, file_name)
            # new_file_name += 1
            subprocess.run(['cp', file_path, interesting_directory_path])


    print("renaming...")
    files = [f for f in os.listdir(interesting_directory_path)]
    files.sort()

    for i, file_name in enumerate(files):
        old_file = os.path.join(interesting_directory_path, file_name)
        temp_file = os.path.join(interesting_directory_path, f"temp_{i}.pkl")
        os.rename(old_file, temp_file)


    temp_files = sorted(f for f in os.listdir(interesting_directory_path) if f.startswith('temp_'))

    for i, file_name in enumerate(temp_files):
        old_file = os.path.join(interesting_directory_path, file_name)
        new_file = os.path.join(interesting_directory_path, f"{i}.pkl")
        os.rename(old_file, new_file)
    