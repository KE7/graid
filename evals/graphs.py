import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from graid.utilities.common import project_root_dir


def extract_ap_lines_from_txts(directory: str):
    pattern = r"(Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50:0\.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+[0-9.]+)"
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as f:
                text = f.read()
                match = re.search(pattern, text)
                if match:
                    results.append((filename, match.group(1).split(" ")[-1]))

    return results


directory_path = project_root_dir() / "data/outputs/"
matches = extract_ap_lines_from_txts(directory_path)
print(matches)


model_data = defaultdict(list)

for filename, score_str in matches:
    # Extract model name and confidence level
    match = re.match(r"output_bdd_([a-zA-Z0-9_]+)_(0\.\d+)_gpu", filename)
    if match:
        model_name = match.group(1)
        confidence = float(match.group(2))
        score = float(score_str)
        model_data[model_name].append((confidence, score))

# Step 2: Plot
plt.figure(figsize=(12, 6))

for model, points in model_data.items():
    # Sort by confidence
    points.sort()
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y, marker="o", label=model)

plt.title("Confidence vs. Score per Model")
plt.xlabel("Confidence Level")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"map_vs_confidence_bdd.png", dpi=300)
