"""ConforFlux for Protenix: the shared core wired to Protenix's denoiser."""

from __future__ import annotations

import os

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
    "ConforFluxState",
    "conforflux_update",
    "is_guided_step",
    "ca_indices_from_features",
    "rigid_align",
    "rmsd",
    "rms_normalize",
]


def ca_indices_from_features(input_feature_dict: dict) -> Tensor:
    mask = input_feature_dict.get("distogram_rep_atom_mask")
    if mask is None:
        raise KeyError("distogram_rep_atom_mask not in input_feature_dict")
    while mask.dim() > 1:
        mask = mask[0]
    return torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1).long()


def conforflux_update(
    s_particles: Tensor,
    z_particles: Tensor,
    x_noisy: Tensor,
    t_hat: Tensor,
    denoiser_fn,
    ca_indices: Tensor,
    cfg: ConforFluxConfig,
    step_i: int,
    total_steps: int,
    state: ConforFluxState | None = None,
) -> tuple[Tensor, Tensor]:
    """One guided update of s_particles [M, N_tok, c_s] and z_particles [M, N_tok, N_tok, c_z]."""
    st = (
        state
        if state is not None
        else ConforFluxState.from_particles(s_particles, z_particles, cfg)
    )
    st.s_particles = s_particles
    st.z_particles = z_particles

    def denoise_ca(s_i: Tensor, z_i: Tensor, i: int) -> Tensor:
        x0 = denoiser_fn(x_noisy[i : i + 1], t_hat[i : i + 1], s_i, z_i)
        return x0[:, ca_indices, :].squeeze(0)

    t = float(t_hat.reshape(-1)[0].item())
    st.step(denoise_ca, t_hat=t, step_idx=step_i)

    if os.environ.get("CONFORFLUX_LOG_NORMS") == "1":
        print(f"[ConforFlux] backbone=protenix {st.last}", flush=True)
    return st.s_particles, st.z_particles
