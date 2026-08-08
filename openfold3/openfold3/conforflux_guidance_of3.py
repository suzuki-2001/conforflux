"""ConforFlux for OpenFold3-preview2: the shared core, wired to OF3's denoiser.

The method lives in `conforflux.repulsion` and is the same object every backbone runs.
This file is the adapter.

OF3-specific wiring:
  * tensors carry a [batch, sample] leading pair of dims, so the M particles are held as
    Python lists of per-particle tensors (s_i [B,1,N_tok,c_s], z_i [B,1,N_tok,N_tok,c_z])
    rather than as one stacked tensor.
  * denoiser_fn(s_i, z_i, x_i, t) -> x0 [B,1,N_atom,3] calls model.diffusion_module.
  * Calpha positions via get_token_center_atoms(batch, x0, atom_mask).
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable

import torch
from torch import Tensor


from conforflux.repulsion import (
    ConforFluxConfig, ConforFluxState, is_guided_step, rigid_align, rmsd, rms_normalize,
)

__all__ = ["ConforFluxConfig", "conforflux_update", "is_guided_step",
           "rigid_align", "rmsd", "rms_normalize"]


class _ListState(ConforFluxState):
    """OF3 keeps the M particles as a Python list of tensors that already carry their own
    [batch, sample] leading dims, so a particle is one list element rather than a slice."""

    def _take(self, arr, i):
        return arr[i]

    def _put(self, arr, i, delta):
        arr[i] = arr[i] - delta

    def _copy(self, arr, dst, src):
        arr[dst] = arr[src].clone()

    @classmethod
    def from_lists(cls, s_list: list, z_list: list, cfg: ConforFluxConfig) -> "_ListState":
        st = cls.__new__(cls)
        st.cfg = cfg
        st.M = len(s_list)
        st.s_particles = s_list
        st.z_particles = z_list
        st.bond = 3.8
        st.last = {}
        return st


def conforflux_update(s_particles: list, z_particles: list, x_noisy: list, t: Tensor,
                      denoiser_fn: Callable, ca_fn: Callable, cfg: ConforFluxConfig,
                      step_i: int, total_steps: int,
                      state: _ListState | None = None) -> tuple[list, list]:
    """One guided embedding update, in OF3's list-of-particles layout."""
    st = state if state is not None else _ListState.from_lists(s_particles, z_particles, cfg)
    st.s_particles = s_particles
    st.z_particles = z_particles

    def denoise_ca(s_i: Tensor, z_i: Tensor, i: int) -> Tensor:
        return ca_fn(denoiser_fn(s_i, z_i, x_noisy[i], t))

    th = float(t.reshape(-1)[0].item())
    st.step(denoise_ca, t_hat=th, step_idx=step_i)

    if os.environ.get("CONFORFLUX_LOG_NORMS") == "1":
        print(f"[ConforFlux] backbone=of3 {st.last}", flush=True)
    return st.s_particles, st.z_particles


