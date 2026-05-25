from __future__ import annotations

from typing import Optional

from torch import Tensor

from conforflux.config import ConforFluxConfig
from conforflux.state import ConforFluxGuidanceState


class ConforFluxHooks:

    def __init__(self, config: ConforFluxConfig, ca_indices: Tensor) -> None:
        self.config = config
        self.ca_indices = ca_indices
        self._state: Optional[ConforFluxGuidanceState] = None

    def set_state(self, state: ConforFluxGuidanceState) -> None:
        self._state = state

    def pre_denoise_embedding(
        self,
        x_noisy: Tensor,
        t_hat: float,
        step_idx: int,
        total_steps: int,
        structure_module,
        network_condition_kwargs: dict,
    ) -> dict:
        state = self._state
        if state is None or not state.is_active(step_idx, total_steps):
            return network_condition_kwargs
        state.step_embedding_update(
            x_noisy, t_hat, structure_module, network_condition_kwargs,
            step_idx, total_steps,
        )
        return network_condition_kwargs
