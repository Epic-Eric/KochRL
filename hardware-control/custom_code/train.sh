#!/bin/bash
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=epiceric666/folding \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/act_koch_folding \
  hydra.job.name=act_koch_folding \
  device=cuda \
  wandb.enable=true \