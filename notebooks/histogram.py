#!/usr/bin/env python3
import os
import pickle
import sys
import re
import matplotlib.pyplot as plt
from datetime import time
from tqdm import tqdm
from datetime import datetime
from brokenaxes import brokenaxes
import numpy as np
import json

def extract_hour_nuimage(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}-(\d{2})-(\d{2})-", filename)
    if not match:
        raise ValueError(f"Time not found in filename: {filename}")
    
    hour = int(match.group(1))
    minute = int(match.group(2))
    t = time(hour, minute)
    return t.hour

def extract_hour_waymo(timestamp_micro: str) -> bool:
    timestamp_sec = int(timestamp_micro) / 1e6
    dt = datetime.utcfromtimestamp(timestamp_sec)
    return dt.hour



def histogram_bdd():
    f = "/home/eecs/liheng/scenic-reasoning/data/bdd100k/labels/det_20/det_train.json"
    time_set = set()
    weather_set = set()
    with open(f, "r") as file:
        data = json.load(file)

    # Initialize counters for time and weather
    time_counts = [0] * 24
    weather_counts = {}

    # Iterate through the dataset
    for item in data:
        attributes = item.get("attributes", {})
        time_of_day = attributes.get("timeofday")
        weather = attributes.get("weather")

        # Update time counts
        if time_of_day == "daytime":
            time_counts[12] += 1
        elif time_of_day == "night":
            time_counts[0] += 1
        elif time_of_day == "dawn/dusk":
            time_counts[6] += 1

        # Update weather counts
        if weather:
            weather_counts[weather] = weather_counts.get(weather, 0) + 1

    # Plot time histogram
    plt.figure(figsize=(10, 5))
    plt.bar(["Night", "Dawn/Dusk", "Daytime"], [time_counts[0], time_counts[6], time_counts[12]])
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Images")
    plt.title("BDD100K Images by Time of Day")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("bdd_time_histogram.png")

    # Plot weather histogram
    plt.figure(figsize=(10, 5))
    plt.bar(weather_counts.keys(), weather_counts.values())
    plt.xlabel("Weather")
    plt.ylabel("Number of Images")
    plt.title("BDD100K Images by Weather")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("bdd_weather_histogram.png")
    

def histogram_waymo(directory):
    hour_counts = [0] * 24 
    # Validate the provided directory exists
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
        
    # Iterate over items in the directory
    for filename in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            hour = extract_hour_waymo(data["timestamp"])
            hour_counts[hour] += 1
    
    # Save the hour_counts to a file
    output_file = "waymo_hour_counts.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(hour_counts, f)
    print(f"Hour counts saved to {output_file}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), hour_counts)
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Images")
    plt.title("Waymo by Hour")
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("waymo_histogram.png")
    # hours = np.arange(24)
    # counts = np.array(hour_counts)

    # fig = plt.figure(figsize=(10, 5))
    # bax = brokenaxes(xlims=((0, 23),), hspace=.05, fig=fig)

    # # Plot the bars on the broken x-axis
    # bax.bar(hours, counts, color='#377eb8', edgecolor='white')

    # # Set labels and title
    # bax.set_xlabel('Hour')
    # bax.set_ylabel('Number of Images')
    # plt.suptitle('Waymo Images by Hour (Broken X-Axis)', fontsize=14, fontweight='bold')

    # # Save the figure
    # plt.savefig("waymo_histogram_broken.png")
    # print("Saved plot to waymo_histogram_broken.png")

def histogram_nuimage(directory):
    hour_counts = [0] * 24 
    # Validate the provided directory exists
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
        
    # Iterate over items in the directory
    for filename in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            hour = extract_hour_nuimage(data["filename"])
            hour_counts[hour] += 1
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), hour_counts)
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Images")
    plt.title("Nuimage by Hour")
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("nuimage_histogram.png")



# histogram("/home/eecs/liheng/scenic-reasoning/data/nuimages_train")
# histogram_waymo("/home/eecs/liheng/scenic-reasoning/data/waymo_training_interesting")
histogram_bdd()




# import matplotlib.pyplot as plt
# import numpy as np
# from brokenaxes import brokenaxes

# # Example data: similar to the earlier example
# hours = np.arange(24)
# counts = np.array([0, 0, 0, 0, 0, 15, 30, 45, 60, 75, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 5, 5, 5])

# # Define the broken portions of the x-axis.
# # In this case, we want to break the axis to exclude 0-7.
# bax = brokenaxes(xlims=((0, 23),), hspace=.05, fig=fig)

# # Plot the data on the broken axis
# # We use a bar plot with full x data (the library will only show the specified ranges)
# bax.bar(hours, counts, color='#377eb8', edgecolor='white')

# # Set labels on the common axes
# bax.set_xlabel('Hour')
# bax.set_ylabel('Count')
# plt.suptitle('Histogram with Broken X-Axis (Excluding Zero Count Hours)', fontsize=14, fontweight='bold')

# plt.show()