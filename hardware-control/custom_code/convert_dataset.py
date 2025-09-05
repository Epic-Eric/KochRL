from pathlib import Path
from lerobot.common.datasets.populate_dataset import create_lerobot_dataset

# Load your existing dataset
root = "data"
repo_id = "epiceric666/folding"
local_dir = Path(root) / repo_id

# This should contain your episodes and videos directories
dataset = {
    "repo_id": repo_id,
    "local_dir": local_dir,
    "videos_dir": local_dir / "videos",
    "episodes_dir": local_dir / "episodes", 
    "fps": 30,  # adjust based on your collection
    "video": True,
    "num_episodes": len(list((local_dir / "episodes").glob("episode_*.pth")))  # count episodes
}

if __name__ == "__main__":
    # Convert to HuggingFace format
    lerobot_dataset = create_lerobot_dataset(
        dataset=dataset,
        run_compute_stats=True,
        push_to_hub=True,  # set to True if you want to upload
    tags=["koch", "folding"],
    play_sounds=False
)