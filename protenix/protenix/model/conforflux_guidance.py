"""ConforFlux for Protenix: the shared core, wired to Protenix's denoiser.

The method lives in `conforflux.repulsion` and is the same object every backbone runs.
This file is the adapter: it finds the Calpha atoms in Protenix's feature
dict and calls the denoiser with the per-particle trunk embeddings.

Protenix-specific wiring:
  * denoiser: DiffusionModule.forward(..., s_trunk=s_i, z_trunk=z_i, pair_z=None). Passing
    pair_z=None makes DiffusionConditioning.prepare_cache recompute the pair conditioning
    from the updated z_i, and the single conditioning is recomputed from s_i, so an update
    to s_i/z_i reaches the whole diffusion module with no external cache handling.
  * Calpha indices: input_feature_dict["distogram_rep_atom_mask"], whose token
    representative atom is the Calpha for protein tokens.
"""
from __future__ import annotations

import os

import torch
from torch import Tensor


from conforflux.repulsion import (
    ConforFluxConfig, ConforFluxState, is_guided_step, rigid_align, rmsd, rms_normalize,
)

__all__ = ["ConforFluxConfig", "ConforFluxState", "conforflux_update", "is_guided_step",
           "ca_indices_from_features", "rigid_align", "rmsd", "rms_normalize"]


def ca_indices_from_features(input_feature_dict: dict) -> Tensor:
    mask = input_feature_dict.get("distogram_rep_atom_mask")
    if mask is None:
        raise KeyError("distogram_rep_atom_mask not in input_feature_dict")
    while mask.dim() > 1:
        mask = mask[0]
    return torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1).long()


def conforflux_update(s_particles: Tensor, z_particles: Tensor, x_noisy: Tensor,
                      t_hat: Tensor, denoiser_fn, ca_indices: Tensor,
                      cfg: ConforFluxConfig, step_i: int, total_steps: int,
                      state: ConforFluxState | None = None) -> tuple[Tensor, Tensor]:
    """One guided embedding update, in Protenix's tensor layout.

    `s_particles` is [M, N_tok, c_s] and `z_particles` [M, N_tok, N_tok, c_z]; the core keeps
    the same layout, so the two share storage rather than copying per step. `state` carries
    the resampling bookkeeping across steps; without it each call starts a fresh state and
    resampling can only fire at step 0.
    """
    st = state if state is not None else ConforFluxState.from_particles(
        s_particles, z_particles, cfg)
    st.s_particles = s_particles
    st.z_particles = z_particles

    def denoise_ca(s_i: Tensor, z_i: Tensor, i: int) -> Tensor:
        x0 = denoiser_fn(x_noisy[i:i + 1], t_hat[i:i + 1], s_i, z_i)
        return x0[:, ca_indices, :].squeeze(0)

    t = float(t_hat.reshape(-1)[0].item())
    st.step(denoise_ca, t_hat=t, step_idx=step_i)

    if os.environ.get("CONFORFLUX_LOG_NORMS") == "1":
        print(f"[ConforFlux] backbone=protenix {st.last}", flush=True)
    return st.s_particles, st.z_particles
