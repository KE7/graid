import os
import re
import matplotlib.pyplot as plt
from scenic_reasoning.utilities.common import project_root_dir

# Define the data directory where the files are stored.
data_dir = project_root_dir() /  "data" / "coco_outputs_bdd_nonfiltered" 
data_dir = str(data_dir)

# Dictionary to hold model data in the form: { model_name: [(conf, map), ...] }
model_data = {}

# Regex pattern to extract model name and confidence from file names.
# Example filename: output_bdd_faster_rcnn_R_50_FPN_3x_0.6_gpu4.txt
pattern_filename = re.compile(r"output_bdd_(.*?)_([\d.]+)_gpu\d+\.txt")

# Regex pattern to extract the mAP value from the file contents.
# This pattern matches: Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.138
pattern_map = re.compile(
    r"Average Precision\s*\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d.]+)"
)
pattern_mar_1 = re.compile(
    r"Average Recall\s*\(AR\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*1\s*\]\s*=\s*([\d.]+)"
)
pattern_mar_10 = re.compile(
    r"Average Recall\s*\(AR\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*10\s*\]\s*=\s*([\d.]+)"
)

# Process each file in the specified directory.
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        match_filename = pattern_filename.search(filename)
        if match_filename:
            model_name = match_filename.group(1)
            conf = float(match_filename.group(2))
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r") as f:
                content = f.read()
                match_map = pattern_map.search(content)
                if match_map:
                    map_value = float(match_map.group(1))
                    # Initialize the list for the model if it doesn't exist.
                    if model_name not in model_data:
                        model_data[model_name] = []

                    match_mar_1 = pattern_mar_1.search(content)
                    match_mar_10 = pattern_mar_10.search(content)
                    
                    mar_1_value = float(match_mar_1.group(1)) if match_mar_1 else None
                    mar_10_value = float(match_mar_10.group(1)) if match_mar_10 else None
                    
                    # Append all metrics to the model data
                    model_data[model_name].append((conf, map_value, mar_1_value, mar_10_value))
                else:
                    print(f"Map value not found in {filename}")
        else:
            print(f"Filename pattern not matched for {filename}")

# Plotting the data.
plt.figure(figsize=(10, 6))
# plt.style.use('seaborn-v0_8-whitegrid')  # For a clean and publication-quality style.
color_iter = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # For consistent coloring.
# For each model, sort the data by confidence and plot the corresponding line.
for model, data_points in model_data.items():
    data_points.sort(key=lambda x: x[0])
    confs = [point[0] for point in data_points]
    maps = [point[1] for point in data_points]
    mar_1s = [point[2] for point in data_points]
    mar_10s = [point[3] for point in data_points]
    
    color = next(color_iter)
    plt.plot(confs, maps, marker='o', color=color, label=f"{model} mAP")
    plt.plot(confs, mar_1s, marker='x', linestyle='--', color=color, label=f"{model} AR@1")
    plt.plot(confs, mar_10s, marker='s', linestyle=':', color=color, label=f"{model} AR@10")

# Add axis labels and title.
plt.xlabel("Confidence Threshold")
plt.ylabel("mAP (Average Precision) & AR (Average Recall)")
plt.title("Model Performance Across Confidence Thresholds")
plt.legend(title="Model", loc='best')
plt.tight_layout()

# Save the figure in high resolution and display it.
plt.savefig("model_performance_chart.png", dpi=300)
plt.show()
