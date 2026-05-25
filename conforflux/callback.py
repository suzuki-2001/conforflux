from __future__ import annotations

import torch
from pytorch_lightning.callbacks import Callback

from conforflux.config import ConforFluxConfig
from conforflux.hooks import ConforFluxHooks
from conforflux.state import ConforFluxGuidanceState
from conforflux.trunk import get_ca_indices, run_trunk


class ConforFluxCallback(Callback):

    def __init__(self, config: ConforFluxConfig, num_particles: int) -> None:
        self.config = config
        self.num_particles = num_particles
        self._hooks: ConforFluxHooks | None = None

    def setup(self, trainer, pl_module, stage: str | None = None) -> None:
        if stage not in (None, "predict"):
            return
        if self.num_particles > 1 and pl_module.predict_args["diffusion_samples"] != self.num_particles:
            pl_module.predict_args["diffusion_samples"] = self.num_particles

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0,
    ) -> None:
        device = next(pl_module.parameters()).device

        with torch.inference_mode(mode=False), torch.no_grad():
            try:
                ca_indices = get_ca_indices(batch).clone().to(device)
            except ValueError:
                n_tokens = batch["token_pad_mask"].shape[-1]
                ca_indices = torch.arange(n_tokens, device=device)

        s_trunk, z_trunk, s_inputs, rel_pos_enc = run_trunk(pl_module, batch)

        # batch comes from Lightning's inference_mode; clone so autograd-tracked
        # DC recomputations can use it.
        with torch.inference_mode(mode=False), torch.no_grad():
            feats = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

        hooks = ConforFluxHooks(self.config, ca_indices)
        state = ConforFluxGuidanceState(
            s_trunk_base=s_trunk,
            z_trunk_base=z_trunk,
            num_particles=self.num_particles,
            ca_indices=ca_indices,
            config=self.config,
            diffusion_cond_module=pl_module.diffusion_conditioning,
            rel_pos_enc=rel_pos_enc,
            feats=feats,
            s_inputs=s_inputs,
        )
        with torch.inference_mode(mode=False), torch.no_grad():
            state._recompute_dc_all()
        hooks.set_state(state)

        pl_module.guidance_hooks = hooks
        self._hooks = hooks
