"""ConforFlux for OpenFold3-preview2: the shared core wired to OF3's denoiser."""

from __future__ import annotations

import os
from typing import Callable

import torch
from torch import Tensor

from conforflux.repulsion import (
    ConforFluxConfig,
    ConforFluxState,
    is_guided_step,
    rms_normalize,
)
from conforflux.superposition import rigid_align, rmsd

__all__ = [
    "ConforFluxConfig",
    "conforflux_update",
    "is_guided_step",
    "sample_conforflux",
    "rigid_align",
    "rmsd",
    "rms_normalize",
]


class _ListState(ConforFluxState):
    """Particles as a list of tensors that carry OF3's [batch, sample] dims."""

    def _take(self, arr, i):
        return arr[i]

    def _put(self, arr, i, delta):
        arr[i] = arr[i] - delta

    def _copy(self, arr, dst, src):
        arr[dst] = arr[src].clone()


def conforflux_update(
    s_particles: list,
    z_particles: list,
    x_noisy: list,
    t: Tensor,
    denoiser_fn: Callable,
    ca_fn: Callable,
    cfg: ConforFluxConfig,
    step_i: int,
    total_steps: int,
    state: _ListState | None = None,
) -> tuple[list, list]:
    """One guided embedding update, in OF3's list-of-particles layout."""
    st = state if state is not None else _ListState.from_particles(s_particles, z_particles, cfg)
    st.s_particles = s_particles
    st.z_particles = z_particles

    def denoise_ca(s_i: Tensor, z_i: Tensor, i: int) -> Tensor:
        return ca_fn(denoiser_fn(s_i, z_i, x_noisy[i], t))

    th = float(t.reshape(-1)[0].item())
    st.step(denoise_ca, t_hat=th, step_idx=step_i)

    if os.environ.get("CONFORFLUX_LOG_NORMS") == "1":
        print(f"[ConforFlux] backbone=of3 {st.last}", flush=True)
    return st.s_particles, st.z_particles


def sample_conforflux(
    sample_diffusion,
    batch: dict,
    si_input: Tensor,
    si_trunk: Tensor,
    zij_trunk: Tensor,
    noise_schedule: Tensor,
    no_rollout_samples: int,
    cfg: ConforFluxConfig,
    **denoise_kwargs,
) -> Tensor:
    """SampleDiffusion.forward with the M rollouts coupled. Returns [B, M, N_atom, 3]."""
    from openfold3.core.model.structure.diffusion_module import (
        centre_random_augmentation,
    )
    from openfold3.core.utils.atomize_utils import get_token_center_atoms

    sd = sample_diffusion
    M = no_rollout_samples

    # Tensors made under inference_mode cannot enter autograd, so the rollout runs outside it.
    with torch.inference_mode(mode=False), torch.no_grad():
        batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        si_input = si_input.clone()
        si_trunk = si_trunk.clone()
        zij_trunk = zij_trunk.clone()
        noise_schedule = noise_schedule.clone()

    atom_mask = batch["atom_mask"]
    batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]
    total = len(noise_schedule) - 1

    def denoise(s_i, z_i, xn, t):
        return sd.diffusion_module(
            batch=batch,
            xl_noisy=xn,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask,
            t=t.to(xn.device),
            si_input=si_input,
            si_trunk=s_i,
            zij_trunk=z_i,
            **denoise_kwargs,
        )

    def ca_fn(x0):
        center_x, center_mask = get_token_center_atoms(batch, x0, atom_mask)
        return center_x[0, 0][center_mask[0, 0].bool()]

    with torch.inference_mode(mode=False):
        s_particles = [si_trunk.detach().clone() for _ in range(M)]
        z_particles = [zij_trunk.detach().clone() for _ in range(M)]
        state = _ListState.from_particles(s_particles, z_particles, cfg)
        xl_list = [
            noise_schedule[0]
            * torch.randn(
                (batch_dim, 1, num_atoms, 3),
                device=atom_mask.device,
                dtype=atom_mask.dtype,
            )
            for _ in range(M)
        ]

        for tau, c_tau in enumerate(noise_schedule[1:]):
            gamma = sd.gamma_0 if c_tau > sd.gamma_min else 0
            t = noise_schedule[tau] * (gamma + 1)
            x_noisy = []
            for m in range(M):
                xlm = centre_random_augmentation(xl=xl_list[m], atom_mask=atom_mask)
                noise = (
                    sd.noise_scale
                    * torch.sqrt(t**2 - noise_schedule[tau] ** 2)
                    * torch.randn_like(xlm)
                )
                x_noisy.append(xlm + noise)

            if is_guided_step(tau, cfg):
                s_particles, z_particles = conforflux_update(
                    s_particles,
                    z_particles,
                    x_noisy,
                    t,
                    denoise,
                    ca_fn,
                    cfg,
                    tau,
                    total,
                    state=state,
                )

            with torch.no_grad():
                x_den = [denoise(s_particles[m], z_particles[m], x_noisy[m], t) for m in range(M)]
            for m in range(M):
                delta = (x_noisy[m] - x_den[m]) / t
                xl_list[m] = x_noisy[m] + sd.step_scale * (c_tau - t) * delta

        return torch.cat(xl_list, dim=1)
