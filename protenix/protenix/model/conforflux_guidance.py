"""ConforFlux (trunk-level particle-guided repulsion) for Protenix.

Ported from the Boltz-2 implementation (`conforflux/state.py`). The mechanism is
identical: M diffusion particles are coupled through a pairwise Calpha-RMSD
repulsion whose gradient is back-propagated to the trunk single/pair embeddings
(s_trunk, z_trunk), NOT to the diffusion coordinates. Protenix shares the AF3
conditioning property: the trunk conditions every block of the diffusion module,
so the same trunk-level update transfers.

Protenix-specific wiring:
  * denoiser: DiffusionModule.forward(..., s_trunk=s_i, z_trunk=z_i, pair_z=None)
    -> passing pair_z=None forces DiffusionConditioning.prepare_cache to recompute
    the pair conditioning from the *updated* z_i (using input_feature_dict["relp"]),
    and the single conditioning is recomputed from s_i internally. So updating
    s_i/z_i propagates to the whole diffusion module with no external cache logic.
  * Calpha indices: input_feature_dict["distogram_rep_atom_mask"] (token
    representative atom == Calpha for protein tokens).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor


@dataclass
class ConforFluxConfig:
    alpha_s: float = 0.02          # RMS-normalized step size for s_trunk
    alpha_z: float = 0.02          # RMS-normalized step size for z_trunk
    sigma: float = 2.0             # RBF kernel bandwidth (Angstrom)
    update_interval: int = 3       # guide every k diffusion steps
    start_frac: float = 0.0
    stop_frac: float = 0.8
    max_update_ratio: float = 100.0  # cap |update| as fraction of embedding norm (100 = off)
    rms_eps: float = 1e-30
    noise_scale: bool = True       # scale update by noise level (bell factor)
    max_offdiag_scale: bool = True # scale by max off-diagonal kernel value (vanishes when spread)
    kernel_saturation_threshold: float = 0.01  # skip the update when max off-diagonal < threshold


def rms_normalize(grad: Tensor, eps: float = 1e-30) -> Tensor:
    return grad / torch.sqrt(torch.mean(grad**2) + eps)


def _kabsch_align(a: Tensor, b: Tensor) -> Tensor:
    """Rigid-align a onto b (differentiable Kabsch via SVD). a,b: [N,3].
    SVD needs float32 and no autocast (model runs bf16)."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        a = a.float(); b = b.float()
        a_c = a - a.mean(dim=0, keepdim=True)
        b_c = b - b.mean(dim=0, keepdim=True)
        h = a_c.transpose(0, 1) @ b_c
        u, _, vt = torch.linalg.svd(h)
        d = torch.sign(torch.det(vt.transpose(0, 1) @ u.transpose(0, 1)))
        diag = torch.eye(3, device=a.device, dtype=torch.float32)
        diag[2, 2] = d
        r = vt.transpose(0, 1) @ diag @ u.transpose(0, 1)
        return a_c @ r.transpose(0, 1) + b.mean(dim=0, keepdim=True)


def _differentiable_rmsd(a: Tensor, b: Tensor) -> Tensor:
    a_aligned = _kabsch_align(a, b)
    return torch.sqrt(((a_aligned - b) ** 2).sum(dim=-1).mean() + 1e-8)


def ca_indices_from_features(input_feature_dict: dict) -> Tensor:
    mask = input_feature_dict.get("distogram_rep_atom_mask", None)
    if mask is None:
        raise KeyError("distogram_rep_atom_mask not in input_feature_dict")
    m = mask
    while m.dim() > 1:
        m = m[0]
    return torch.nonzero(m > 0.5, as_tuple=False).squeeze(-1).long()


def is_guided_step(step_i: int, total_steps: int, cfg: ConforFluxConfig) -> bool:
    frac = step_i / max(total_steps, 1)
    if not (cfg.start_frac <= frac < cfg.stop_frac):
        return False
    return (step_i % cfg.update_interval) == 0


def conforflux_update(
    s_particles: Tensor,          # [M, N_token, c_s]
    z_particles: Tensor,          # [M, N_token, N_token, c_z]
    x_noisy: Tensor,              # [M, N_atom, 3]
    t_hat: Tensor,                # [M]
    denoiser_fn: Callable,        # (x_noisy_i[1,N,3], t_hat_i[1], s_i[1,..], z_i[1,..]) -> x0_i[1,N_atom,3]
    ca_indices: Tensor,           # [N_ca]
    cfg: ConforFluxConfig,
    step_i: int,
    total_steps: int,
) -> tuple[Tensor, Tensor]:
    """One guided embedding update. Returns updated (s_particles, z_particles)."""
    M = s_particles.shape[0]
    th = float(t_hat.reshape(-1)[0].item())
    scale = (1.0 + math.log1p(max(0.0, th - 1.0))) if cfg.noise_scale else 1.0

    with torch.enable_grad():
        s_list, z_list, cas = [], [], []
        for i in range(M):
            s_i = s_particles[i:i + 1].detach().clone().requires_grad_(True)
            z_i = z_particles[i:i + 1].detach().clone().requires_grad_(True)
            x0_i = denoiser_fn(x_noisy[i:i + 1], t_hat[i:i + 1], s_i, z_i)
            ca_i = x0_i[:, ca_indices, :].float().squeeze(0)   # [N_ca, 3]
            s_list.append(s_i); z_list.append(z_i); cas.append(ca_i)

        N = M
        L = torch.zeros(N, N, device=cas[0].device, dtype=torch.float32)
        for i in range(M):
            for j in range(i + 1, M):
                r = torch.clamp(_differentiable_rmsd(cas[i], cas[j]), min=0.1)
                k = torch.exp(-r**2 / (2 * cfg.sigma**2))
                L[i, j] = k; L[j, i] = k
        mask = torch.triu(torch.ones(N, N, device=L.device), diagonal=1).bool()
        loss = L[mask].sum()

        # Gate on the closest pair's kernel value, so the update fades as the particles separate.
        L_off = L.detach().clone()
        L_off.fill_diagonal_(0.0)
        max_offdiag = float(L_off.max())
        if cfg.max_offdiag_scale:
            scale = scale * max_offdiag
        if max_offdiag < cfg.kernel_saturation_threshold:
            return s_particles, z_particles

        targets = []
        for i in range(M):
            targets.append(s_list[i]); targets.append(z_list[i])
        grads = torch.autograd.grad(loss, targets, allow_unused=True)

    with torch.no_grad():
        idx = 0
        for i in range(M):
            gs = grads[idx]; idx += 1
            gz = grads[idx]; idx += 1
            if cfg.alpha_s > 0 and gs is not None:
                ds = cfg.alpha_s * scale * rms_normalize(gs.squeeze(0), cfg.rms_eps)
                mx = s_particles[i].norm() * cfg.max_update_ratio
                if ds.norm() > mx:
                    ds = ds * (mx / ds.norm())
                s_particles[i] = s_particles[i] - ds
            if cfg.alpha_z > 0 and gz is not None:
                dz = cfg.alpha_z * scale * rms_normalize(gz.squeeze(0), cfg.rms_eps)
                mx = z_particles[i].norm() * cfg.max_update_ratio
                if dz.norm() > mx:
                    dz = dz * (mx / dz.norm())
                z_particles[i] = z_particles[i] - dz

    return s_particles, z_particles
