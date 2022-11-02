import math
import torch
import numpy as np


def mixup_for_outlier(
    X: torch.tensor,
    Y: torch.tensor,
    section: torch.tensor,
    alpha=0.2,
    use_neg_section_as_zero=False,
):
    """MixUp for ASD."""
    batch_size = X.size(0)
    lam = torch.tensor(
        np.random.beta(alpha, alpha, batch_size), dtype=torch.float32
    ).to(X.device)[:, None]
    perm = torch.randperm(batch_size).to(X.device)
    mixed_X = lam * X + (1 - lam) * X[perm]
    mixed_Y = lam * Y + (1 - lam) * Y[perm]
    section[0 == Y.squeeze(1)] = 0
    mixed_section = lam * section + (1 - lam) * section[perm]
    if use_neg_section_as_zero:
        section_idx = torch.ones(batch_size, dtype=torch.bool).to(X.device)
    else:
        section_idx = (0 < mixed_Y).squeeze(1)

    return mixed_X, mixed_Y, mixed_section, section_idx


def schedule_cos_phases(max_step, step):
    return 0.5 * (1.0 - math.cos(min(math.pi, 2 * math.pi * step / max_step)))


def count_params(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params
