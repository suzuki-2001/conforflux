"""Boltz-2 plumbing for the repulsion. The method itself is in `repulsion.py`."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from conforflux.config import ConforFluxConfig
from conforflux.repulsion import ConforFluxState, is_guided_step


class ConforFluxGuidanceState(ConforFluxState):
    """The hook interface Boltz-2's sampler calls: `is_active` then `step_embedding_update`."""

    def __init__(self, s_trunk_base: Tensor, z_trunk_base: Tensor, num_particles: int,
                 ca_indices: Tensor, config: ConforFluxConfig,
                 diffusion_cond_module=None, rel_pos_enc=None, feats=None,
                 s_inputs=None, checkpoint_threshold: int = 500) -> None:
        super().__init__(s_trunk_base, z_trunk_base, num_particles, config)
        self.ca_indices = ca_indices
        self.dc_module = diffusion_cond_module
        self.rel_pos_enc = rel_pos_enc
        self.feats = feats
        self.s_inputs = s_inputs
        self.cached_dc: Optional[dict] = None
        if len(ca_indices) >= checkpoint_threshold:
            self.cfg = type(config)(**{**config.__dict__, "gradient_checkpointing": True})

    def is_active(self, step_idx: int, total_steps: int) -> bool:
        return is_guided_step(step_idx, self.cfg)

    def _dc_single(self, s_i: Tensor, z_i: Tensor) -> Optional[dict]:
        if self.dc_module is None:                      # Boltz-1 takes s/z directly
            return None
        q, c, to_keys, ae, ad, tt = self.dc_module(
            s_trunk=s_i, z_trunk=z_i,
            relative_position_encoding=self.rel_pos_enc, feats=self.feats)
        return {"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": ae,
                "atom_dec_bias": ad, "token_trans_bias": tt}

    def _denoise_ca(self, s_i: Tensor, z_i: Tensor, x_noisy_i: Tensor, t_hat: float,
                    structure_module) -> Tensor:
        kwargs = dict(multiplicity=1, s_trunk=s_i, s_inputs=self.s_inputs, feats=self.feats)
        if self.dc_module is not None:
            kwargs["diffusion_conditioning"] = self._dc_single(s_i, z_i)
        else:
            kwargs["z_trunk"] = z_i
            kwargs["relative_position_encoding"] = self.rel_pos_enc
        out = structure_module.preconditioned_network_forward(
            x_noisy_i, t_hat, network_condition_kwargs=kwargs)
        x0 = out[0] if isinstance(out, tuple) else out
        return x0[:, self.ca_indices, :].float().squeeze(0)

    def _recompute_dc_all(self) -> None:
        if self.dc_module is None:
            self.cached_dc = None
            return
        parts = [self._dc_single(self.s_particles[i:i + 1], self.z_particles[i:i + 1])
                 for i in range(self.M)]
        self.cached_dc = {
            "q": torch.cat([p["q"] for p in parts], dim=0),
            "c": torch.cat([p["c"] for p in parts], dim=0),
            "to_keys": parts[0]["to_keys"],
            "atom_enc_bias": torch.cat([p["atom_enc_bias"] for p in parts], dim=0),
            "atom_dec_bias": torch.cat([p["atom_dec_bias"] for p in parts], dim=0),
            "token_trans_bias": torch.cat([p["token_trans_bias"] for p in parts], dim=0),
        }

    def prepare(self) -> None:
        """Build the initial per-particle conditioning, before the first denoise step."""
        with torch.no_grad():
            self._recompute_dc_all()

    def _publish(self, network_condition_kwargs: dict) -> None:
        """Hand the updated particles to the Euler step."""
        with torch.no_grad():
            self._recompute_dc_all()
        network_condition_kwargs["s_trunk"] = self.s_particles.detach()
        if self.cached_dc is not None:
            network_condition_kwargs["diffusion_conditioning"] = self.cached_dc
            return
        # Boltz-1 has no conditioning module, so every conditioning tensor is expanded to M.
        network_condition_kwargs["z_trunk"] = self.z_particles.detach()
        for key in ("s_inputs", "relative_position_encoding"):
            v = network_condition_kwargs.get(key)
            if v is not None and v.shape[0] < self.M:
                network_condition_kwargs[key] = v.expand(self.M, *v.shape[1:])
        feats = network_condition_kwargs.get("feats")
        if feats is not None:
            network_condition_kwargs["feats"] = {
                k: (v.expand(self.M, *v.shape[1:])
                    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1 else v)
                for k, v in feats.items()}

    def step_embedding_update(self, x_noisy: Tensor, t_hat: float, structure_module,
                              network_condition_kwargs: dict, step_idx: int,
                              total_steps: int) -> None:
        restored = _unwrap_checkpointed_forwards(structure_module)
        try:
            self.step(lambda s_i, z_i, i: self._denoise_ca(
                s_i, z_i, x_noisy[i:i + 1], t_hat, structure_module), t_hat, step_idx)
        finally:
            for mod, fwd in restored:
                mod.forward = fwd
        self._publish(network_condition_kwargs)


def _unwrap_checkpointed_forwards(structure_module) -> list:
    """fairscale's checkpoint_wrapper replaces forward with a partial that autograd.grad
    cannot differentiate through. Swap the class methods back for the duration of the step."""
    restored = []
    if structure_module is None:
        return restored
    from functools import partial
    for mod in structure_module.modules():
        fn = getattr(mod, "forward", None)
        if isinstance(fn, partial) and "checkpoint" in getattr(fn.func, "__qualname__", ""):
            restored.append((mod, mod.forward))
            cls_forward = type(mod).forward
            mod.forward = lambda *a, _m=mod, _f=cls_forward, **kw: _f(_m, *a, **kw)
    return restored


def build_state_from_trunk(s_trunk: Tensor, z_trunk: Tensor, num_particles: int,
                           ca_indices: Tensor, config: ConforFluxConfig,
                           diffusion_cond_module, rel_pos_enc, feats: dict,
                           s_inputs: Tensor) -> ConforFluxGuidanceState:
    return ConforFluxGuidanceState(
        s_trunk_base=s_trunk, z_trunk_base=z_trunk, num_particles=num_particles,
        ca_indices=ca_indices, config=config, diffusion_cond_module=diffusion_cond_module,
        rel_pos_enc=rel_pos_enc, feats=feats, s_inputs=s_inputs)
