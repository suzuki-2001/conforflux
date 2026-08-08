"""The ConforFlux update: repulsion between diffusion particles on the trunk embeddings."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from conforflux.config import ConforFluxConfig
from conforflux.superposition import rigid_align, rmsd


def rms_normalize(grad: Tensor, eps: float = 1e-30) -> Tensor:
    return grad / torch.sqrt(torch.mean(grad ** 2) + eps)


def is_guided_step(step_idx: int, cfg: ConforFluxConfig) -> bool:
    """Every step in the trajectory is a candidate; the interval thins them."""
    return cfg.update_interval <= 1 or step_idx % cfg.update_interval == 0


def broken_particles(cas: list[Tensor], bond_tol: float, bond: float) -> tuple[list[int], list[int], list[float]]:
    """Split particles by whether their Calpha trace still has physical bond lengths."""
    healthy: list[int] = []
    broken: list[int] = []
    scores: list[float] = []
    for i, ca in enumerate(cas):
        d = ((ca[1:] - ca[:-1]) ** 2).sum(dim=-1).sqrt()
        dev = (d - bond).abs()
        scores.append(float(dev.mean()))
        (broken if int((dev > bond_tol).sum()) > len(d) * 0.1 else healthy).append(i)
    return healthy, broken, scores


class ConforFluxState:
    """M copies of the trunk embeddings, pushed apart in Calpha space.

    `denoise_ca(s_i, z_i, i)` runs the backbone's denoiser for particle `i` on the current
    trunk copies and returns its Calpha coordinates, differentiably.
    """

    def __init__(self, s_trunk: Tensor, z_trunk: Tensor, num_particles: int,
                 cfg: ConforFluxConfig) -> None:
        self.cfg = cfg
        self.M = num_particles
        self.s_particles = s_trunk.expand(num_particles, *s_trunk.shape[1:]).clone().float()
        self.z_particles = z_trunk.expand(num_particles, *z_trunk.shape[1:]).clone().float()
        self.bond = 3.8                      # protein Calpha; RNA C1' is set on the first call
        self.last: dict = {}                 # diagnostics for the caller to log

    @classmethod
    def from_particles(cls, s_particles: Tensor, z_particles: Tensor,
                       cfg: ConforFluxConfig) -> "ConforFluxState":
        """Attach to particle tensors a backbone already owns, updating them in place.

        Protenix and OpenFold3 carry the M copies themselves through their samplers, so the
        state has to write into those tensors rather than into copies of its own.
        """
        st = cls.__new__(cls)
        st.cfg = cfg
        st.M = int(s_particles.shape[0])
        st.s_particles = s_particles
        st.z_particles = z_particles
        st.bond = 3.8
        st.last = {}
        return st

    # How a single particle is taken out of the container and put back. Boltz-2 and Protenix
    # stack the M particles in a tensor, where a particle is a slice that keeps the leading
    # dimension the denoiser expects; OpenFold3 keeps a Python list of tensors that already
    # carry their own leading dims. `_ListState` overrides these two.
    def _take(self, arr, i: int):
        return arr[i:i + 1]

    def _put(self, arr, i: int, delta: Tensor) -> None:
        arr[i] = arr[i] - delta.squeeze(0)

    def _copy(self, arr, dst: int, src: int) -> None:
        arr[dst] = arr[src].clone()

    def _resample(self, cas: list[Tensor], step_idx: int) -> None:
        cfg = self.cfg
        if not cfg.resample or step_idx % cfg.resample_interval != 0:
            return
        with torch.no_grad():
            d = ((cas[0][1:] - cas[0][:-1]) ** 2).sum(dim=-1).sqrt()
            if step_idx == 0 and float(d.median()) > 4.5:
                self.bond = 5.9
            healthy, broken, scores = broken_particles(
                [c.detach() for c in cas], cfg.bond_tol, self.bond)
            if not broken or not healthy:
                return
            best = min(healthy, key=lambda i: scores[i])
            for b in broken:
                self._copy(self.s_particles, b, best)
                self._copy(self.z_particles, b, best)
            self.last["resampled"] = broken

    def step(self, denoise_ca: Callable[[Tensor, Tensor, int], Tensor],
             t_hat: float, step_idx: int) -> bool:
        """One update. Returns whether the embeddings moved."""
        cfg = self.cfg
        M = self.M
        update_z = cfg.alpha_z > 0
        noise = (1.0 + math.log1p(max(0.0, t_hat - 1.0))) if cfg.noise_scale else 1.0
        self.last = {"step": step_idx}

        if cfg.objective == "noise":
            with torch.no_grad():
                for i in range(M):
                    self.s_particles[i] -= cfg.alpha_s * noise * rms_normalize(
                        torch.randn_like(self.s_particles[i]), cfg.rms_eps)
                    if update_z:
                        self.z_particles[i] -= cfg.alpha_z * noise * rms_normalize(
                            torch.randn_like(self.z_particles[i]), cfg.rms_eps)
            self.last.update(scale=noise, updated=True)
            return True

        with torch.enable_grad():
            s_list, z_list, cas = [], [], []
            for i in range(M):
                s_i = self._take(self.s_particles, i).detach().clone().requires_grad_(True)
                z_i = (self._take(self.z_particles, i).detach().clone().requires_grad_(True)
                       if update_z else self._take(self.z_particles, i))
                if cfg.gradient_checkpointing:
                    ca = torch.utils.checkpoint.checkpoint(
                        denoise_ca, s_i, z_i, i, use_reentrant=False)
                else:
                    ca = denoise_ca(s_i, z_i, i)
                s_list.append(s_i)
                z_list.append(z_i)
                cas.append(ca.float())

            self._resample(cas, step_idx)

            dev = cas[0].device
            r = torch.zeros(M, M, device=dev)
            for i in range(M):
                for j in range(i + 1, M):
                    v = torch.clamp(rmsd(cas[i], cas[j]), min=0.1)
                    r[i, j] = v
                    r[j, i] = v
            kernel = torch.exp(-r ** 2 / (2 * cfg.sigma ** 2))
            upper = torch.triu(torch.ones_like(kernel), diagonal=1).bool()
            loss = kernel[upper].sum()

            off = kernel.detach().clone()
            off.fill_diagonal_(0.0)
            max_offdiag = float(off.max())
            scale = max_offdiag * noise if cfg.max_offdiag_scale else noise
            self.last.update(max_offdiag=max_offdiag, scale=scale,
                             rmsd_mean=float(r[upper].detach().mean()),
                             rmsd_min=float(r[upper].detach().min()))

            # Far apart relative to sigma: the ensemble is already spread and pushing further
            # only buys off-manifold structures.
            if max_offdiag < cfg.kernel_saturation_threshold:
                self.last["updated"] = False
                return False

            targets: list[Tensor] = []
            for i in range(M):
                targets.append(s_list[i])
                if update_z:
                    targets.append(z_list[i])
            grads = torch.autograd.grad(loss, targets)

        with torch.no_grad():
            k = 0
            for i in range(M):
                self._put(self.s_particles, i,
                          cfg.alpha_s * scale * rms_normalize(grads[k], cfg.rms_eps))
                k += 1
                if update_z:
                    self._put(self.z_particles, i,
                              cfg.alpha_z * scale * rms_normalize(grads[k], cfg.rms_eps))
                    k += 1
        self.last["updated"] = True
        return True
