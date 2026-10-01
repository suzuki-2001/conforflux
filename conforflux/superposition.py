"""Kabsch superposition."""

from __future__ import annotations

import torch
from torch import Tensor


def rigid_align(a: Tensor, b: Tensor) -> Tensor:
    """Superpose `a` onto `b` and return the moved `a`. Differentiable, [N, 3] each."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        a32, b32 = a.float(), b.float()
        a_c = a32 - a32.mean(dim=0, keepdim=True)
        b_c = b32 - b32.mean(dim=0, keepdim=True)
        u, _s, vh = torch.linalg.svd(a_c.transpose(0, 1) @ b_c)
        d = torch.det(u @ vh).detach()
        f = torch.eye(3, device=a.device, dtype=torch.float32).clone()
        f[2, 2] = torch.sign(d)
        return a_c @ (u @ f @ vh) + b32.mean(dim=0, keepdim=True)


def rmsd(a: Tensor, b: Tensor) -> Tensor:
    """Calpha-RMSD after optimal superposition."""
    moved = rigid_align(a, b)
    return ((moved - b.float()) ** 2).sum(dim=-1).mean().sqrt()
