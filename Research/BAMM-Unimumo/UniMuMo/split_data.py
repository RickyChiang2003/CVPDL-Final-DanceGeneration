import os
import glob
import random
import shutil
from collections import defaultdict
import re

data_dir = '/tmp2/b11705045/CVPDL/Final-project/data/processed_house/processed_codes'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all files
files = glob.glob(os.path.join(data_dir, '*.pth'))
print(f"Total files found: {len(files)}")

# Group by prefix
pattern = re.compile(r"(.*)_beat(\d+)_beat(\d+)\.pth")
prefix_map = defaultdict(list)

for f in files:
    basename = os.path.basename(f)
    match = pattern.match(basename)
    if match:
        prefix = match.group(1)
        prefix_map[prefix].append(f)

prefixes = list(prefix_map.keys())
random.shuffle(prefixes)

n_total = len(prefixes)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val

train_prefixes = prefixes[:n_train]
val_prefixes = prefixes[n_train:n_train+n_val]
test_prefixes = prefixes[n_train+n_val:]

print(f"Split: Train={len(train_prefixes)}, Val={len(val_prefixes)}, Test={len(test_prefixes)}")

def move_files(prefix_list, target_dir):
    count = 0
    for p in prefix_list:
        file_list = prefix_map[p]
        for f in file_list:
            shutil.move(f, os.path.join(target_dir, os.path.basename(f)))
            count += 1
    return count

n_train_files = move_files(train_prefixes, train_dir)
n_val_files = move_files(val_prefixes, val_dir)
n_test_files = move_files(test_prefixes, test_dir)

print(f"Moved files: Train={n_train_files}, Val={n_val_files}, Test={n_test_files}")

















