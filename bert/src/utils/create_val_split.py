import json
import math
from tqdm import tqdm
from collections import defaultdict

directory = # Enter path to your data directory here
new_directory = # Enter output path here
val_pct = 0.0005 # Percentage of data to use for validation

index = f"{directory}/train/index.json"
with open(index, "r") as f:
    index = json.load(f)

train_index = {}
val_index = {}

# Version
train_index["version"] = index["version"]
val_index["version"] = index["version"]

# Shards
num_shards = len(index["shards"])
num_train_shards = math.floor((1 - val_pct) * num_shards)
train_index['shards'] = []
val_index['shards'] = []
train_basenames = []
val_basenames = []

print(f"Splitting into {num_train_shards} train shards and {num_shards - num_train_shards} val shards")
for item in tqdm(index['shards'], desc="Splitting shards"):
    shard_basename = item['raw_data']['basename']
    shard = shard_basename.split('.')[1]
    shard = int(shard)

    if shard < num_train_shards:
        train_index['shards'].append(item)
        train_basenames.append(shard_basename)

    else:
        val_index['shards'].append(item)
        val_basenames.append(shard_basename)

# Save down the new indices
import os
train_directory = f"{new_directory}/train/"
val_directory = f"{new_directory}/val/"
os.makedirs(train_directory, exist_ok=True)
os.makedirs(val_directory, exist_ok=True)

with open(f"{train_directory}/index.json", "w") as f:
    json.dump(train_index, f)
with open(f"{val_directory}/index.json", "w") as f:
    json.dump(val_index, f)

# Copy the shards from the old directory to the new directories
import shutil
for basename in tqdm(train_basenames, desc="Copying train shards"):
    shutil.copy(f"{directory}/train/{basename}", f"{train_directory}/{basename}")
for basename in tqdm(val_basenames, desc="Copying val shards"):
    shutil.copy(f"{directory}/train/{basename}", f"{val_directory}/{basename}")