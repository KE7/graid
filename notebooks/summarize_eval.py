# import os
# import re
# import ast



# # Directory containing the .txt files
# directory = '/home/eecs/liheng/scenic-reasoning/data/databases_final/'

# # Iterate through every .txt file in the directory
# correctness = []
# for filename in os.listdir(directory):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(directory, filename)
#         with open(file_path, 'r') as f:
#             text = f.read()
#             match = re.search(r"Correctness:\s*\n(.*?)\n", text)
#             score = match.group(1)
#             score = ast.literal_eval(score)
#             correctness.extend(score)

# print(sum(correctness) / len(correctness))



import os
import re
import ast
from collections import defaultdict

# Root directory
root_directory = '/home/eecs/liheng/scenic-reasoning/data/databases_final/'

# Dictionary to store correctness scores per directory
directory_scores = defaultdict(list)

# Walk through each subdirectory and file
for dirpath, _, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(dirpath, filename)
            with open(file_path, 'r') as f:
                text = f.read()
                match = re.search(r"Correctness:\s*\n(.*?)\n", text)
                if match:
                    score_str = match.group(1)
                    try:
                        score = ast.literal_eval(score_str)
                        directory_scores[os.path.basename(dirpath)].extend([int(s) for s in score])
                    except Exception as e:
                        print(f"Error parsing score in {file_path}: {e}")

# Compute average correctness per directory
directory_correctness = {}
for dirpath, scores in directory_scores.items():
    if scores:
        average = sum(scores) / len(scores)
        directory_correctness[dirpath] = average
    else:
        directory_correctness[dirpath] = None  # or float('nan')

# Print results
for directory, avg_score in directory_correctness.items():
    print(f"{directory}: {avg_score}")
