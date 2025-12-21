## KochRL

KochRL is an Isaac Lab-based reinforcement learning project for a 6-DoF Koch manipulator. It includes a custom Direct RL environment, PPO training/evaluation scripts, visualization helpers, and an optional hardware teleoperation runner to deploy trained policies on real robots.

* **Env ID**: `Template-Kochrl-Direct-v0`
* **Stack**: Isaac Lab (Isaac Sim), PyTorch, RSL-RL (PPO)
* **Package**: `KochRL` (installed from `source/KochRL`)

### Features

* **Custom Isaac Lab task** under `KochRL/tasks/direct/kochrl` with:

  * Target sampling within a hemispherical workspace
  * Optional external force injection with spring-equilibrium clamping
  * Keypoint-based observation of the end-effector
  * Visualization markers for targets and forces
* **Training/Eval scripts** via RSL-RL (`scripts/rsl_rl/`)
* **Hardware teleop runner** to drive a follower arm from a trained policy (`hardware-control/custom_code/teleop_rl_runner.py`)

---
### Current branching
main --- tested, reliable position matching in simulation
actual-force --- dev branch for force position matching, simulation in progress, inference in progress

## Quick Commands

### Train an agent with RSL-RL

```bash
conda activate ericisaaclab
cd ~/ericxie/KochRL/source/KochRL
source ~/IsaacLab/_isaac_sim/setup_conda_env.sh
python ../../scripts/rsl_rl/train.py --num_envs=4096 --task Template-Kochrl-Direct-v0 --headless --log_project_name kochrl-force
```

### Play with a trained agent

```bash
python ../../scripts/rsl_rl/play.py --task Template-Kochrl-Direct-v0 --num_envs=1
```

### Resume training from a previous run

```bash
python ../../scripts/rsl_rl/train.py --num_envs=4096 --task Template-Kochrl-Direct-v0 --headless --log_project_name kochrl --load_run 2025-08-13_23-57-43 --resume
```

### Load a checkpoint and continue training with a new project name

```bash
python ../../scripts/rsl_rl/train.py --num_envs=4096 --task Template-Kochrl-Direct-v0 --headless --log_project_name kochrl2 --load_run 2025-08-14_12-54-26 --checkpoint model_598.pt
```

### Open Isaac Sim to edit USD files

```bash
cd ~/IsaacLab/_isaac_sim
bash isaac-sim.sh
```

### Inference (position only)

```bash
conda activate erickoch
cd hardware-control
python custom_code/teleop_rl_runner.py --checkpoint /home/asblab/ericxie/KochRL/source/KochRL/logs/rsl_rl/kochrl/2025-09-03_14-34-46/model_999.pt --device cuda --rate 60
```

### Inference (force-based)

```bash
python /home/asblab/ericxie/KochRL/hardware-control/custom_code/teleop_force_runner.py --init-align --rate 60 --k 1000 --device cuda --predictor /home/asblab/ericxie/KochRL/force-predictor/xgb_force_model2.pkl
```

---

## 1) Prerequisites

* Python 3.10
* Isaac Lab installed and working (follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))

  * Conda install is recommended for easy CLI use
* NVIDIA GPU recommended

## 2) Install KochRL

From the repo root:

```bash
# Use Isaac Lab Python if it's not your default interpreter
python -m pip install -e source/KochRL
```

Verify the task is registered:

```bash
# Lists available tasks (looks for "Template-")
python scripts/list_envs.py
```

You should see `Template-Kochrl-Direct-v0`.

## 3) Train with PPO (RSL-RL)

```bash
# Minimal example
python scripts/rsl_rl/train.py --task=Template-Kochrl-Direct-v0
```

* Training configs: `source/KochRL/KochRL/tasks/direct/kochrl/agents/rsl_rl_ppo_cfg.py`
* Robot config: `source/KochRL/KochRL/tasks/direct/kochrl/koch.py`
* Env config: `source/KochRL/KochRL/tasks/direct/kochrl/kochrl_env_cfg.py`
* Logs and Hydra outputs: `source/KochRL/outputs/<DATE>/<TIME>/`
* RSL-RL logs/checkpoints: `source/KochRL/logs/rsl_rl/kochrl/`

Play/evaluate a checkpoint:

```bash
python scripts/rsl_rl/play.py --task=Template-Kochrl-Direct-v0 --checkpoint /path/to/checkpoint.pt
```

## 4) Dummy agents

Quick functional tests without training:

```bash
# Zero-action agent
python scripts/zero_agent.py --task=Template-Kochrl-Direct-v0

# Random-action agent
python scripts/random_agent.py --task=Template-Kochrl-Direct-v0
```

## 5) Environment overview

* Implementation: `KochRL/tasks/direct/kochrl/kochrl_env.py`
* Registration: `KochRL/tasks/direct/kochrl/__init__.py`
* Helpers: `KochRL/tasks/direct/kochrl/helper.py`

### Observations (high level)

* Joint positions/velocities (6 + 6)
* End-effector pose `[x, y, z, qx, qy, qz, qw]`
* End-effector linear and angular velocities
* Three EE keypoints (9 dims) derived from orientation to avoid discontinuities
* Target errors (position and keypoints)
* Stiffness parameter, previous action, applied torque

Note: Exact layout and sizes are defined in `kochrl_env.py` and `kochrl_env_cfg.py`.

### Actions

* 6-dim delta joint positions, clamped to per-joint limits (`kochrl_env_cfg.py`).

### Rewards (core components)

* Position error penalty (L2 of keypoint error)
* Position tracking reward (tanh kernel)
* Action rate penalty (L2 of delta in actions)
* End-effector acceleration penalty

### Target sampling and forces

* Targets sampled in a hemisphere around `sampling_origin` with `sampling_radius`
* Optional external forces sampled randomly; equilibrium position computed and clamped within workspace

## 6) Hardware teleoperation runner (optional)

Use a trained policy to control a follower arm:

```bash
python hardware-control/custom_code/teleop_rl_runner.py \
  --checkpoint /path/to/exported_or_training_checkpoint.pt \
  --device cuda \
  --rate 50 \
  --init-align
```

* Requires hardware access and `lerobot` device stack
* Builds KochRL-style observations, runs policy inference, and sends position targets to the follower
* See file for serial ports and bus config: `hardware-control/custom_code/teleop_rl_runner.py`
* Joint-limit mirroring and real↔sim angle conversions are handled inside the runner

## 7) Development

* Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit run --all-files
```

* Optional VSCode extension setup is described in the template README; you can mirror that by generating `.vscode/.python.env` to index Isaac modules.

## 8) Project structure

```text
source/KochRL/
  KochRL/
    tasks/
      direct/kochrl/
        agents/rsl_rl_ppo_cfg.py  # PPO config
        helper.py                 # reward fns, sampling, quat utils
        koch.py                   # robot/actuator config
        kochrl_env.py             # Direct RL env (observations, rewards, resampling)
        kochrl_env_cfg.py         # env config & limits
    ui_extension_example.py
scripts/
  rsl_rl/train.py | play.py | cli_args.py
  list_envs.py
  zero_agent.py
  random_agent.py
hardware-control/custom_code/teleop_rl_runner.py
```

## 9) Troubleshooting

* Ensure Isaac Lab is importable in the interpreter you use for `pip install -e`.
* If indexing is slow or missing in IDEs, add `source/KochRL` to Python analysis extra paths.
* For GPU rendering or physics issues, consult the official Isaac Lab docs and your Isaac Sim version compatibility.

## License

This project is distributed under the MIT License (see `setup.py`).
