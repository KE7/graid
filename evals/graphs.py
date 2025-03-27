import json
import matplotlib.pyplot as plt
from scenic_reasoning.utilities.common import project_root_dir
from scenic_reasoning.utilities.coco import coco_label
import os
from collections import defaultdict


# def load_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# d = 'bdd_new'

# data_dir = project_root_dir() / 'evals'

# files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
# files = [filename for filename in files if filename.endswith('.json')]

# # Initialize the plot
# plt.figure(figsize=(10, 6))

# # Loop over each model file
# for file_name in files:
#     file_path = data_dir / file_name
#     data = load_json(file_path)

#     # Get the model key (e.g., 'yolo_v10x-BDD100K...')
#     top_key = list(data.keys())[0]
#     model_name = top_key.split('-')[0]  # Extract just the model name, like 'yolo_v10x'

#     section = data[top_key]['metrics_no_pen']

#     confidences = []
#     map_values = []

#     best_metrics = dict()

#     for conf_str, metrics in section.items():
#         confidence = float(conf_str)
#         map_score = metrics['map']
#         print(f"map score for {model_name} at confidence {confidence:.1f}: {map_score}")
#         confidences.append(confidence)
#         map_values.append(map_score)
#         if map_score > best_metrics.get('map', 0):
#             best_metrics = metrics

#     sorted_pairs = sorted(zip(confidences, map_values))
#     sorted_confidences, sorted_map_values = zip(*sorted_pairs)

#     # Plot one line per model
#     plt.plot(sorted_confidences, sorted_map_values, marker='o', label=model_name)
#     print(f"Plotted {model_name}")

#     # If detailed per-class metrics exist, print the top 5 classes by mAP
#     if 'map_per_class' in best_metrics and 'mar_100_per_class' in best_metrics:
#         map_per_class = best_metrics['map_per_class']
#         mar_100_per_class = best_metrics['mar_100_per_class']
        
#         # Sort class indices by mAP in descending order and select the top 5
#         top_classes = sorted(range(len(map_per_class)), key=lambda i: map_per_class[i], reverse=True)[:5]
#         print("  Top 5 classes:")
#         for idx in top_classes:
#             label = coco_label.get(idx, f'class_{idx}')
#             print(f"    {label}: mAP = {map_per_class[idx]}")
        
#         # Create a single horizontal bar chart for the top 5 classes
#         fig, ax = plt.subplots(figsize=(10, 4))
#         class_names = [coco_label.get(idx, f'class_{idx}') for idx in top_classes]
#         y_pos = range(len(class_names))
        
#         map_values = [map_per_class[idx] for idx in top_classes]
#         mar_values = [mar_100_per_class[idx] for idx in top_classes]
        
#         ax.barh(y_pos, map_values, height=0.4, color='blue', alpha=0.7, label='mAP')
#         ax.barh([y + 0.4 for y in y_pos], mar_values, height=0.4, color='red', alpha=0.7, label='mAR')
        
#         ax.set_yticks([y + 0.2 for y in y_pos])
#         ax.set_yticklabels(class_names)
#         ax.set_xlabel('Score')
#         ax.set_title(f'Top 5 Classes for {model_name}')
#         ax.legend()
#         ax.grid(axis='x', linestyle='--', alpha=0.7)
        
#         fig.tight_layout()
#         fig.savefig(f'top_classes_{model_name}_{d}.png', dpi=300)
#         plt.close(fig)

# # Final plot settings
# plt.xlabel('Confidence Threshold')
# plt.ylabel('mAP')
# plt.title(f'mAP vs Confidence Threshold ({d})')
# plt.grid(True)
# plt.legend(title='Model')
# plt.tight_layout()

# # Save and show the plot
# plt.savefig(f'map_vs_confidence_{d}.png', dpi=300)




import re

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
    match = re.match(r'output_bdd_([a-zA-Z0-9_]+)_(0\.\d+)_gpu', filename)
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
    plt.plot(x, y, marker='o', label=model)

plt.title("Confidence vs. Score per Model")
plt.xlabel("Confidence Level")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'map_vs_confidence_bdd.png', dpi=300)
