from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import shutil
import os

# Clear any existing cache
cache_dirs = [
    "~/.cache/huggingface/datasets/epiceric666___folding",
    "~/.cache/huggingface/hub/datasets--epiceric666--folding",
    "./data/epiceric666/folding"
]

for cache_dir in cache_dirs:
    full_path = os.path.expanduser(cache_dir)
    if os.path.exists(full_path):
        print(f"Removing {full_path}")
        shutil.rmtree(full_path)

# Force download with cache_dir=None to avoid any caching
print("Loading fresh dataset...")
dataset = LeRobotDataset("epiceric666/folding", root="/tmp/fresh_folding")

print(f"Number of samples: {dataset.num_samples}")  # Should be 20372
print(f"Number of episodes: {dataset.num_episodes}")  # Should be 49

# Check max episode index
episode_indices = list(dataset.hf_dataset['episode_index'])
print(f"Max episode index: {max(episode_indices)}")  # Should be 48

if dataset.num_samples == 20372:
    print("✅ Dataset is correct!")
else:
    print("❌ Still using old cached data")