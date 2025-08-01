import torch

def clamp_actions(l1:list, limits:list):
    assert len(l1) == len(limits)
    min_limits = torch.tensor(limits[0][i] for i in range(len(limits)))
    max_limits = torch.tensor(limits[1][i] for i in range(len(limits)))
    return torch.clamp(l1, min_limits, max_limits)