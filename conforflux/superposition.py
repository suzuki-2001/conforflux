"""Optimal superposition, one implementation.

With row vectors the deviation |a_c M - b_c| is minimised by M = U F Vh, where
cov = a_c^T b_c = U S Vh and F = diag(1, 1, det(U Vh)) removes the reflection. Applying M^T
instead superposes by the inverse rotation; the guidance did that until 2026-08-07, so its
pairwise distance was a Calpha-RMSD only while two structures already shared a frame.
"""
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
