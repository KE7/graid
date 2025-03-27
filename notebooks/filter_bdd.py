import json

# Path to the JSON file
json_file_path = '/Users/harry/Desktop/Nothing/sky/scenic-reasoning/data/bdd100k/labels/det_20/det_val.json'

# Load the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# import pdb
# pdb.set_trace()

weather_set = set(['partly cloudy', 'clear'])
timeofday_set = set(['daytime'])
print(weather_set)
print(timeofday_set)
print('total:', len(data))
count = 0
empty_count = 0
for d in data:
    if len(d['labels']) == 0:
        empty_count += 1
    if d['attributes']['weather'] not in weather_set and d['attributes']['timeofday'] not in timeofday_set:
        continue
    count += 1
    # weather_set.add(d['attributes']['weather'])
    # timeofday_set.add(d['attributes']['timeofday'])

print('filtered', count)
print('empty', empty_count)
