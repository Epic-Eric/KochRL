#!/usr/bin/env python3
"""
Script to load RSL-RL policy weights and perform inference on observations.

This script demonstrates how to:
1. Load a trained policy from rsl_rl checkpoints
2. Set up observation normalization
3. Process observations and output actions

Usage:
    python load_policy_inference.py --checkpoint path/to/model.pt
    python load_policy_inference.py --checkpoint path/to/model.pt --obs "1.0,2.0,3.0,..."
"""

import argparse
import os
import sys
import torch
import numpy as np
import pickle
from typing import Optional, Union

# Add the source directory to the path to import KochRL modules
script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(os.path.dirname(script_dir), "source", "KochRL")
sys.path.insert(0, source_dir)

# Note: We avoid importing rsl_rl here to prevent API/version mismatches.


class PolicyLoader:
    """Class to load and use RSL-RL policies for inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the policy loader.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Load the policy and normalization parameters
        self._load_policy()
        self._load_normalization()
        
    def _load_policy(self):
        """Load exported TorchScript policy (exported/policy.pt) and avoid rsl_rl API issues."""
        print(f"Loading policy from: {self.checkpoint_path}")

        # Expected dims from env
        self.obs_dim = 52
        self.action_dim = 6

        run_dir = self.checkpoint_dir
        exported_pt = os.path.join(run_dir, "exported", "policy.pt")

        # Prefer TorchScript export next to the checkpoint
        if os.path.exists(exported_pt):
            self.backend = "jit"
            self.policy = torch.jit.load(exported_pt, map_location=self.device)
            self.policy.eval()
            print("✓ Loaded exported TorchScript policy.pt")
            return

        # If the checkpoint itself is a TorchScript file, try that
        try:
            self.backend = "jit"
            self.policy = torch.jit.load(self.checkpoint_path, map_location=self.device)
            self.policy.eval()
            print("✓ Loaded TorchScript policy from checkpoint path")
            return
        except Exception:
            pass

        raise RuntimeError(
            f"No TorchScript policy found. Expected '{exported_pt}' or a TorchScript checkpoint."
        )
    
    def _load_normalization(self):
        """Load observation normalization parameters."""
        # Try to load obs normalizer from the checkpoint directory
        normalizer_path = os.path.join(self.checkpoint_dir, "obs_normalizer.pkl")
        params_dir = os.path.join(self.checkpoint_dir, "params")
        
        self.obs_normalizer = None
        
        # Try to load from params directory (newer versions)
        if os.path.exists(params_dir):
            agent_pkl_path = os.path.join(params_dir, "agent.pkl")
            if os.path.exists(agent_pkl_path):
                try:
                    with open(agent_pkl_path, 'rb') as f:
                        agent_params = pickle.load(f)
                    
                    # Extract normalizer parameters if available
                    if hasattr(agent_params, 'obs_normalizer'):
                        self.obs_normalizer = agent_params.obs_normalizer
                        print("✓ Observation normalizer loaded from agent.pkl")
                    elif 'obs_normalizer' in agent_params:
                        self.obs_normalizer = agent_params['obs_normalizer']
                        print("✓ Observation normalizer loaded from agent.pkl")
                except Exception as e:
                    print(f"Could not load normalizer from agent.pkl: {e}")
        
        # Try direct normalizer file
        elif os.path.exists(normalizer_path):
            try:
                with open(normalizer_path, 'rb') as f:
                    self.obs_normalizer = pickle.load(f)
                print("✓ Observation normalizer loaded from obs_normalizer.pkl")
            except Exception as e:
                print(f"Could not load normalizer: {e}")
        
        if self.obs_normalizer is None:
            print("⚠ No observation normalizer found. Using identity normalization.")
            # Create a dummy normalizer that doesn't change the observations
            class IdentityNormalizer:
                def __call__(self, obs):
                    return obs
                def normalize(self, obs):
                    return obs
            self.obs_normalizer = IdentityNormalizer()
    
    def get_action(self, observation: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """
        Get action from observation.
        
        Args:
            observation: Input observation (shape: [53] or [batch_size, 53])
            
        Returns:
            Action array (shape: [6] or [batch_size, 6])
        """
        # Convert to tensor if needed
        if isinstance(observation, (list, np.ndarray)):
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = observation.to(self.device)
        
        # Ensure proper shape
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Validate observation dimension
        if obs_tensor.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observation dimension {self.obs_dim}, got {obs_tensor.shape[-1]}")
        
        # Normalize observation
        if hasattr(self.obs_normalizer, 'normalize'):
            normalized_obs = self.obs_normalizer.normalize(obs_tensor)
        else:
            normalized_obs = self.obs_normalizer(obs_tensor)
        
        # Get action from policy
        with torch.inference_mode():
            if hasattr(self.policy, 'act_inference'):
                action = self.policy.act_inference(normalized_obs)
            else:
                out = self.policy(normalized_obs)
                action = out if isinstance(out, torch.Tensor) else out[0]
        
        # Convert to numpy and remove batch dimension if needed
        action_np = action.cpu().numpy()
        if squeeze_output:
            action_np = action_np.squeeze(0)
        
        return action_np
    
    def print_observation_structure(self):
        """Print the expected observation structure for the KochRL environment."""
        print("\n" + "="*60)
        print("KOCHRL ENVIRONMENT OBSERVATION STRUCTURE (52 dimensions)")
        print("="*60)
        print("Joint space (12 dimensions):")
        print("  - Joint positions [0:6]     : Current joint angles")
        print("  - Joint velocities [6:12]   : Joint angular velocities")
        print("")
        print("Cartesian space (22 dimensions):")
        print("  - End-effector pose [12:19] : Position (3) + Quaternion (4)")
        print("  - EE linear velocity [19:22]: Linear velocity of end-effector")
        print("  - EE angular velocity [22:25]: Angular velocity of end-effector")
        print("  - Keypoints [25:34]         : 9 keypoint coordinates")
        print("")
        print("Task parameters (12 dimensions):")
        print("  - Target error [34:37]      : Position error to target")
        print("  - Target keypoint error [37:46]: Keypoint error to target")
        print("")
        print("Other (6 dimensions):")
        print("  - Previous action [46:52]   : Last executed action")
        print("="*60)


def parse_observation_string(obs_str: str) -> np.ndarray:
    """Parse observation string into numpy array."""
    try:
        # Remove brackets and split by comma
        obs_str = obs_str.strip("[]")
        values = [float(x.strip()) for x in obs_str.split(",")]
        return np.array(values)
    except Exception as e:
        raise ValueError(f"Could not parse observation string: {e}")


def create_random_observation(seed: Optional[int] = None) -> np.ndarray:
    """Create a random observation for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create random observation with reasonable ranges
    obs = np.zeros(53)
    
    # Joint positions (6) - within typical joint limits
    obs[0:6] = np.random.uniform(-3.14, 3.14, 6)
    
    # Joint velocities (6) - small values
    obs[6:12] = np.random.uniform(-1.0, 1.0, 6)
    
    # End-effector pose (7) - position + quaternion
    obs[12:15] = np.random.uniform(-1.0, 1.0, 3)  # position
    obs[15:19] = np.array([0, 0, 0, 1])  # identity quaternion
    
    # EE velocities (6)
    obs[19:25] = np.random.uniform(-0.5, 0.5, 6)
    
    # Keypoints (9)
    obs[25:34] = np.random.uniform(-1.0, 1.0, 9)
    
    # Target errors (12)
    obs[34:46] = np.random.uniform(-0.5, 0.5, 12)
    
    # Stiffness (1)
    obs[46:47] = np.random.uniform(0.1, 1.0, 1)
    
    # Previous action (6)
    obs[47:53] = np.random.uniform(-0.1, 0.1, 6)
    
    return obs


def main():
    parser = argparse.ArgumentParser(description="Load RSL-RL policy and perform inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--obs", type=str, default=None,
                       help="Observation string (comma-separated values) or 'random'")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for generating random observations")
    parser.add_argument("--print-structure", action="store_true",
                       help="Print observation structure and exit")
    
    args = parser.parse_args()
    
    if args.print_structure:
        PolicyLoader.print_observation_structure(None)
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"✗ Checkpoint file not found: {args.checkpoint}")
        print("\nAvailable checkpoints in KochRL logs:")
        logs_dir = "/home/asblab/ericxie/KochRL/source/KochRL/logs/rsl_rl/kochrl"
        if os.path.exists(logs_dir):
            for run_dir in os.listdir(logs_dir):
                run_path = os.path.join(logs_dir, run_dir)
                if os.path.isdir(run_path):
                    model_files = [f for f in os.listdir(run_path) if f.startswith("model_") and f.endswith(".pt")]
                    if model_files:
                        print(f"  {run_dir}/:")
                        for model_file in sorted(model_files):
                            print(f"    {os.path.join(run_path, model_file)}")
        return
    
    try:
        # Load the policy
        print("Initializing policy loader...")
        policy_loader = PolicyLoader(args.checkpoint, args.device)
        
        # Print observation structure
        policy_loader.print_observation_structure()
        
        # Get observation
        if args.obs is None or args.obs.lower() == "random":
            print(f"\nGenerating random observation (seed={args.random_seed})...")
            observation = create_random_observation(args.random_seed)
        else:
            print(f"\nParsing provided observation...")
            observation = parse_observation_string(args.obs)
        
        print(f"Observation shape: {observation.shape}")
        print(f"Observation (first 10 values): {observation[:10]}")
        
        # Get action
        print("\nComputing action...")
        action = policy_loader.get_action(observation)
        
        print(f"Action shape: {action.shape}")
        print(f"Action: {action}")
        print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
        
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
