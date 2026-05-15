from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
from torch import Tensor

from conforflux.config import ConforFluxConfig


def differentiable_rigid_align(coords_a: Tensor, coords_b: Tensor) -> Tensor:
    centroid_a = coords_a.mean(dim=0, keepdim=True)
    centroid_b = coords_b.mean(dim=0, keepdim=True)
    a_centered = coords_a - centroid_a
    b_centered = coords_b - centroid_b
    cov = a_centered.T @ b_centered
    U, _S, Vh = torch.linalg.svd(cov.to(torch.float32))
    d = torch.det((U @ Vh).float()).detach()
    F = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=coords_a.device, dtype=torch.float32))
    R = (U @ F @ Vh).to(coords_a.dtype)
    return a_centered @ R.T + centroid_b


def differentiable_rmsd(coords_a: Tensor, coords_b: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    aligned_a = differentiable_rigid_align(coords_a, coords_b)
    sq_dev = ((aligned_a - coords_b) ** 2).sum(dim=-1)
    if weights is not None:
        return (sq_dev * weights).sum().div(weights.sum()).sqrt()
    return sq_dev.mean().sqrt()


def rms_normalize(grad: Tensor, eps: float = 1e-30) -> Tensor:
    rms = torch.sqrt(torch.mean(grad ** 2) + eps)
    return grad / rms


class ConforFluxGuidanceState:

    def __init__(
        self,
        s_trunk_base: Tensor,  # [1, N_tokens, d_s]
        z_trunk_base: Tensor,  # [1, N_tokens, N_tokens, d_z]
        num_particles: int,
        ca_indices: Tensor,
        config: ConforFluxConfig,
        diffusion_cond_module,
        rel_pos_enc,
        feats: dict,
        s_inputs: Tensor,
    ) -> None:
        self.config = config
        self.M = num_particles
        self.ca_indices = ca_indices
        self.s_particles = s_trunk_base.expand(num_particles, -1, -1).clone().float()
        self.z_particles = z_trunk_base.expand(num_particles, -1, -1, -1).clone().float()
        self.dc_module = diffusion_cond_module
        self.rel_pos_enc = rel_pos_enc
        self.feats = feats
        self.s_inputs = s_inputs
        self.cached_dc: Optional[dict] = None
        self._backbone_bond_dist = 3.8  # protein Cα; switched to 5.9 for RNA C1' on first call

    def is_active(self, step_idx: int, total_steps: int) -> bool:
        frac = step_idx / total_steps
        if not (self.config.start_frac <= frac < self.config.stop_frac):
            return False
        if self.config.update_interval > 1 and step_idx % self.config.update_interval != 0:
            return False
        return True

    def _should_checkpoint(self) -> bool:
        return self.config.gradient_checkpointing

    def _recompute_dc_single(self, s_i: Tensor, z_i: Tensor) -> dict:
        q, c, to_keys, ae, ad, tt = self.dc_module(
            s_trunk=s_i, z_trunk=z_i,
            relative_position_encoding=self.rel_pos_enc, feats=self.feats,
        )
        return {"q": q, "c": c, "to_keys": to_keys,
                "atom_enc_bias": ae, "atom_dec_bias": ad, "token_trans_bias": tt}

    def _forward_single(self, s_i, z_i, x_noisy_i, t_hat, structure_module):
        dc_i = self._recompute_dc_single(s_i, z_i)
        kwargs = dict(
            multiplicity=1, s_trunk=s_i, s_inputs=self.s_inputs,
            feats=self.feats, diffusion_conditioning=dc_i,
        )
        result = structure_module.preconditioned_network_forward(
            x_noisy_i, t_hat, network_condition_kwargs=kwargs,
        )
        x_hat_0_i = result[0] if isinstance(result, tuple) else result
        return x_hat_0_i[:, self.ca_indices, :].float()

    def _recompute_dc_all(self) -> dict:
        q_l, c_l, ae_l, ad_l, tt_l = [], [], [], [], []
        to_keys_shared = None
        for i in range(self.M):
            dc = self._recompute_dc_single(self.s_particles[i:i + 1], self.z_particles[i:i + 1])
            q_l.append(dc["q"])
            c_l.append(dc["c"])
            ae_l.append(dc["atom_enc_bias"])
            ad_l.append(dc["atom_dec_bias"])
            tt_l.append(dc["token_trans_bias"])
            if to_keys_shared is None:
                to_keys_shared = dc["to_keys"]
        self.cached_dc = {
            "q": torch.cat(q_l, dim=0),
            "c": torch.cat(c_l, dim=0),
            "to_keys": to_keys_shared,
            "atom_enc_bias": torch.cat(ae_l, dim=0),
            "atom_dec_bias": torch.cat(ad_l, dim=0),
            "token_trans_bias": torch.cat(tt_l, dim=0),
        }
        return self.cached_dc

    def _refresh_kwargs_for_euler(self, network_condition_kwargs: dict) -> None:
        with torch.no_grad():
            self._recompute_dc_all()
        network_condition_kwargs["s_trunk"] = self.s_particles.detach()
        network_condition_kwargs["diffusion_conditioning"] = self.cached_dc

    def _check_and_resample(self, cas: list[Tensor], step_idx: int) -> None:
        cfg = self.config
        if not cfg.resample or step_idx % cfg.resample_interval != 0:
            return
        with torch.no_grad():
            M = len(cas)
            healthy: list[int] = []
            broken: list[int] = []
            bond_scores: list[float] = []
            for i in range(M):
                ca = cas[i].detach()
                dists = ((ca[1:] - ca[:-1]) ** 2).sum(dim=-1).sqrt()
                if i == 0 and step_idx == 0 and dists.median().item() > 4.5:
                    self._backbone_bond_dist = 5.9
                deviations = (dists - self._backbone_bond_dist).abs()
                n_bad = (deviations > cfg.bond_tol).sum().item()
                bond_scores.append(deviations.mean().item())
                (broken if n_bad > len(dists) * 0.1 else healthy).append(i)
            if not broken or not healthy:
                return
            best = min(healthy, key=lambda i: bond_scores[i])
            for b in broken:
                self.s_particles[b] = self.s_particles[best].clone()
                self.z_particles[b] = self.z_particles[best].clone()

    def step_embedding_update(
        self,
        x_noisy: Tensor,
        t_hat: float,
        structure_module,
        network_condition_kwargs: dict,
        step_idx: int,
        total_steps: int,
    ) -> None:
        cfg = self.config
        M = self.M
        update_z = cfg.alpha_z > 0
        noise_scale = (1.0 + math.log1p(max(0.0, t_hat - 1.0))) if cfg.noise_scale else 1.0
        with torch.inference_mode(mode=False), torch.enable_grad():
            # Clone x_noisy out of Lightning's inference_mode so autograd can save it.
            x_noisy = x_noisy.detach().clone()
            # Restore original forwards under any fairscale checkpoint_wrapper.
            _restored: list[tuple[object, object]] = []
            if structure_module is not None:
                for mod in structure_module.modules():
                    fwd = getattr(mod, "forward", None)
                    if isinstance(fwd, partial) and "checkpoint" in getattr(
                        getattr(fwd, "func", None), "__qualname__", ""
                    ):
                        orig_cls_forward = type(mod).forward
                        _restored.append((mod, mod.forward))
                        mod.forward = (
                            lambda *a, _m=mod, _f=orig_cls_forward, **kw: _f(_m, *a, **kw)
                        )
            try:
                use_checkpoint = self._should_checkpoint()
                s_list: list[Tensor] = []
                z_list: list[Tensor] = []
                cas: list[Tensor] = []
                for i in range(M):
                    s_i = self.s_particles[i:i + 1].clone().requires_grad_(True)
                    if update_z:
                        z_i = self.z_particles[i:i + 1].clone().requires_grad_(True)
                    else:
                        z_i = self.z_particles[i:i + 1]
                    s_list.append(s_i)
                    z_list.append(z_i)
                    if use_checkpoint:
                        ca_i = torch.utils.checkpoint.checkpoint(
                            self._forward_single,
                            s_i, z_i, x_noisy[i:i + 1], t_hat, structure_module,
                            use_reentrant=False,
                        )
                    else:
                        ca_i = self._forward_single(
                            s_i, z_i, x_noisy[i:i + 1], t_hat, structure_module,
                        )
                    cas.append(ca_i.squeeze(0))
                self._check_and_resample(cas, step_idx)
                device = cas[0].device
                rmsd = torch.zeros(M, M, device=device)
                for i in range(M):
                    for j in range(i + 1, M):
                        r = torch.clamp(differentiable_rmsd(cas[i], cas[j]), min=0.1)
                        rmsd[i, j] = r
                        rmsd[j, i] = r
                L = torch.exp(-rmsd ** 2 / (2 * cfg.sigma ** 2))
                upper = torch.triu(torch.ones_like(L), diagonal=1).bool()
                loss = L[upper].sum()
                L_off = L.detach().clone()
                L_off.fill_diagonal_(0.0)
                max_offdiag = L_off.max().item()
                update_scale = max_offdiag * noise_scale if cfg.max_offdiag_scale else noise_scale
                if max_offdiag < cfg.kernel_saturation_threshold:
                    self._refresh_kwargs_for_euler(network_condition_kwargs)
                    return
                grad_targets: list[Tensor] = []
                for i in range(M):
                    grad_targets.append(s_list[i])
                    if update_z:
                        grad_targets.append(z_list[i])
                try:
                    grads = torch.autograd.grad(loss, grad_targets)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        self._refresh_kwargs_for_euler(network_condition_kwargs)
                        return
                    raise
            finally:
                for mod, orig_fwd in _restored:
                    mod.forward = orig_fwd
        with torch.no_grad():
            idx = 0
            for i in range(M):
                grad_s = grads[idx].squeeze(0)
                self.s_particles[i] = self.s_particles[i] - cfg.alpha_s * update_scale * rms_normalize(grad_s, cfg.rms_eps)
                idx += 1
                if update_z:
                    grad_z = grads[idx].squeeze(0)
                    self.z_particles[i] = self.z_particles[i] - cfg.alpha_z * update_scale * rms_normalize(grad_z, cfg.rms_eps)
                    idx += 1
        self._refresh_kwargs_for_euler(network_condition_kwargs)


def build_state_from_trunk(
    s_trunk: Tensor,
    z_trunk: Tensor,
    num_particles: int,
    ca_indices: Tensor,
    config: ConforFluxConfig,
    diffusion_cond_module,
    rel_pos_enc,
    feats: dict,
    s_inputs: Tensor,
) -> ConforFluxGuidanceState:
    return ConforFluxGuidanceState(
        s_trunk_base=s_trunk,
        z_trunk_base=z_trunk,
        num_particles=num_particles,
        ca_indices=ca_indices,
        config=config,
        diffusion_cond_module=diffusion_cond_module,
        rel_pos_enc=rel_pos_enc,
        feats=feats,
        s_inputs=s_inputs,
    )
