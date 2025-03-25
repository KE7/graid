import json
import matplotlib.pyplot as plt
from scenic_reasoning.utilities.common import project_root_dir
import os


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

d = 'bdd'

data_dir = project_root_dir() / 'data/eval_results' / d

files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Initialize the plot
plt.figure(figsize=(10, 6))

# Loop over each model file
for file_name in files:
    file_path = data_dir / file_name
    data = load_json(file_path)

    # Get the model key (e.g., 'yolo_v10x-BDD100K...')
    top_key = list(data.keys())[0]
    model_name = top_key.split('-')[0]  # Extract just the model name, like 'yolo_v10x'

    section = data[top_key]['metrics_pen']

    confidences = []
    map_values = []

    for conf_str, metrics in section.items():
        confidence = float(conf_str)
        map_score = metrics['map']
        print("map score", map_score)
        confidences.append(confidence)
        map_values.append(map_score)

    sorted_pairs = sorted(zip(confidences, map_values))
    sorted_confidences, sorted_map_values = zip(*sorted_pairs)

    # Plot one line per model
    plt.plot(sorted_confidences, sorted_map_values, marker='o', label=model_name)
    print(f"Plotted {model_name}")

# Final plot settings
plt.xlabel('Confidence Threshold')
plt.ylabel('mAP')
plt.title(f'mAP vs Confidence Threshold ({d})')
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# Save and show the plot
plt.savefig(f'map_vs_confidence_{d}.png', dpi=300)
