"""ConforFlux (trunk-level particle-guided repulsion) for OpenFold3-preview2.

Mirrors the Protenix port (`protenix/protenix/model/conforflux_guidance.py`) and the
Boltz-2 origin (`conforflux/state.py`). Mechanism is identical:
M diffusion particles are coupled through a pairwise Calpha-RMSD RBF repulsion whose
gradient is back-propagated to the trunk single/pair embeddings (si_trunk, zij_trunk),
NOT to the diffusion coordinates. OF3 shares the AF3 conditioning property: the trunk
conditions every block of the diffusion module (DiffusionConditioning recomputes si/zij
from si_trunk/zij_trunk every step), so updating the trunk propagates everywhere.

OF3-specific wiring (vs Protenix):
  * tensors carry a [batch, sample] leading pair of dims; we keep M particles as Python
    lists of per-particle tensors (si_i [B,1,N_tok,c_s], zij_i [B,1,N_tok,N_tok,c_z]).
  * denoiser_fn(s_i, z_i, x_i, t) -> x0 [B,1,N_atom,3] calls model.diffusion_module(...).
  * Calpha positions via get_token_center_atoms(batch, x0, atom_mask) (Ca for protein).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor


@dataclass
class ConforFluxConfig:
    alpha_s: float = 0.02
    alpha_z: float = 0.02
    sigma: float = 2.0
    update_interval: int = 3
    start_frac: float = 0.0
    stop_frac: float = 0.8
    max_update_ratio: float = 100.0
    rms_eps: float = 1e-30
    noise_scale: bool = True
    max_offdiag_scale: bool = True
    kernel_saturation_threshold: float = 0.01


def rms_normalize(grad: Tensor, eps: float = 1e-30) -> Tensor:
    return grad / torch.sqrt(torch.mean(grad**2) + eps)


def _kabsch_align(a: Tensor, b: Tensor) -> Tensor:
    """Rigid-align a onto b (differentiable Kabsch via SVD). a,b: [N,3]. float32, no autocast."""
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


def is_guided_step(step_i: int, total_steps: int, cfg: ConforFluxConfig) -> bool:
    frac = step_i / max(total_steps, 1)
    if not (cfg.start_frac <= frac < cfg.stop_frac):
        return False
    return (step_i % cfg.update_interval) == 0


def conforflux_update(
    s_particles: list,            # list of M tensors [B,1,N_tok,c_s]
    z_particles: list,            # list of M tensors [B,1,N_tok,N_tok,c_z]
    x_noisy: list,                # list of M tensors [B,1,N_atom,3]
    t: Tensor,                    # scalar tensor (noise level)
    denoiser_fn: Callable,        # (s_i, z_i, x_i, t) -> x0 [B,1,N_atom,3]
    ca_fn: Callable,              # (x0 [B,1,N_atom,3]) -> ca coords [N_ca,3] (differentiable)
    cfg: ConforFluxConfig,
    step_i: int,
    total_steps: int,
) -> tuple[list, list]:
    """One guided embedding update (in place on the list elements). Returns updated lists."""
    M = len(s_particles)
    th = float(t.reshape(-1)[0].item())
    scale = (1.0 + math.log1p(max(0.0, th - 1.0))) if cfg.noise_scale else 1.0

    # inference_mode is not lifted by enable_grad, and prediction runs under it.
    with torch.inference_mode(mode=False), torch.enable_grad():
        s_list, z_list, cas = [], [], []
        for i in range(M):
            s_i = s_particles[i].detach().clone().requires_grad_(True)
            z_i = z_particles[i].detach().clone().requires_grad_(True)
            x0_i = denoiser_fn(s_i, z_i, x_noisy[i], t)
            ca_i = ca_fn(x0_i).float()                     # [N_ca, 3]
            s_list.append(s_i); z_list.append(z_i); cas.append(ca_i)

        L = torch.zeros(M, M, device=cas[0].device, dtype=torch.float32)
        for i in range(M):
            for j in range(i + 1, M):
                r = torch.clamp(_differentiable_rmsd(cas[i], cas[j]), min=0.1)
                k = torch.exp(-r**2 / (2 * cfg.sigma**2))
                L[i, j] = k; L[j, i] = k
        mask = torch.triu(torch.ones(M, M, device=L.device), diagonal=1).bool()
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
                ds = cfg.alpha_s * scale * rms_normalize(gs, cfg.rms_eps)
                mx = s_particles[i].norm() * cfg.max_update_ratio
                if ds.norm() > mx:
                    ds = ds * (mx / ds.norm())
                s_particles[i] = s_particles[i] - ds
            if cfg.alpha_z > 0 and gz is not None:
                dz = cfg.alpha_z * scale * rms_normalize(gz, cfg.rms_eps)
                mx = z_particles[i].norm() * cfg.max_update_ratio
                if dz.norm() > mx:
                    dz = dz * (mx / dz.norm())
                z_particles[i] = z_particles[i] - dz

    return s_particles, z_particles


def sample_conforflux(
    sample_diffusion,             # SampleDiffusion module (gamma/noise/step scales, denoiser)
    batch: dict,
    si_input: Tensor,
    si_trunk: Tensor,
    zij_trunk: Tensor,
    noise_schedule: Tensor,
    no_rollout_samples: int,      # M
    cfg: ConforFluxConfig,
    **denoise_kwargs,
) -> Tensor:
    """Coupled rollout: M particles, each carrying its own (si_trunk, zij_trunk).

    Replaces SampleDiffusion.forward's independent rollout. The schedule, augmentation and
    step scaling are the module's own, so with the guidance inactive this reduces to stock
    sampling. Returns [B, M, N_atom, 3], the shape the unguided path returns.
    """
    from openfold3.core.model.structure.diffusion_module import centre_random_augmentation
    from openfold3.core.utils.atomize_utils import get_token_center_atoms

    sd = sample_diffusion
    M = no_rollout_samples

    # Prediction runs under inference_mode, and tensors created there can never carry an
    # autograd graph. Clone the inputs out of it and run the whole rollout outside, so the
    # per-step coordinates the guidance differentiates through are ordinary tensors.
    with torch.inference_mode(mode=False), torch.no_grad():
        batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        si_input = si_input.clone()
        si_trunk = si_trunk.clone()
        zij_trunk = zij_trunk.clone()
        noise_schedule = noise_schedule.clone()

    atom_mask = batch["atom_mask"]
    batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]
    total = len(noise_schedule) - 1
    inference_mode_off = torch.inference_mode(mode=False)

    def denoise(s_i, z_i, xn, t):
        return sd.diffusion_module(
            batch=batch, xl_noisy=xn, token_mask=batch["token_mask"], atom_mask=atom_mask,
            t=t.to(xn.device), si_input=si_input, si_trunk=s_i, zij_trunk=z_i,
            **denoise_kwargs)

    def ca_fn(x0):  # [B, 1, N_atom, 3] -> [N_ca, 3]
        center_x, center_mask = get_token_center_atoms(batch, x0, atom_mask)
        return center_x[0, 0][center_mask[0, 0].bool()]

    with inference_mode_off:
        s_particles = [si_trunk.detach().clone() for _ in range(M)]
        z_particles = [zij_trunk.detach().clone() for _ in range(M)]
        xl_list = [
            noise_schedule[0] * torch.randn(
                (batch_dim, 1, num_atoms, 3), device=atom_mask.device, dtype=atom_mask.dtype)
            for _ in range(M)
        ]

        for tau, c_tau in enumerate(noise_schedule[1:]):
            gamma = sd.gamma_0 if c_tau > sd.gamma_min else 0
            t = noise_schedule[tau] * (gamma + 1)
            x_noisy = []
            for m in range(M):
                xlm = centre_random_augmentation(xl=xl_list[m], atom_mask=atom_mask)
                noise = sd.noise_scale * torch.sqrt(t ** 2 - noise_schedule[tau] ** 2) * torch.randn_like(xlm)
                x_noisy.append(xlm + noise)

            if is_guided_step(tau, total, cfg):
                s_particles, z_particles = conforflux_update(
                    s_particles, z_particles, x_noisy, t, denoise, ca_fn, cfg, tau, total)

            with torch.no_grad():
                x_den = [denoise(s_particles[m], z_particles[m], x_noisy[m], t) for m in range(M)]
            for m in range(M):
                delta = (x_noisy[m] - x_den[m]) / t
                xl_list[m] = x_noisy[m] + sd.step_scale * (c_tau - t) * delta

        return torch.cat(xl_list, dim=1)
