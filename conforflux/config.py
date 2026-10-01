from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConforFluxConfig:
    """The defaults are the paper's Ours (lambda); noise_scale gives Ours (lambda_t)."""

    sigma: float = 2.0  # RBF bandwidth on Calpha-RMSD, in Angstrom
    alpha_s: float = 0.02  # step size on s_trunk, in units of its own RMS
    alpha_z: float = 0.02  # step size on z_trunk; 0 disables the pair update
    update_interval: int = 3  # fire the gradient every K diffusion steps
    noise_scale: bool = False  # scale the step by the EDM noise level
    max_offdiag_scale: bool = True  # scale the step by the largest off-diagonal kernel
    kernel_saturation_threshold: float = 0.01  # below this the particles are already apart
    reg_weight: float = 0.0  # fraction of the displacement from the trunk output removed per update
    resample: bool = True  # replace particles whose backbone has broken
    resample_interval: int = 10
    bond_tol: float = 1.0  # Angstrom around the expected Calpha-Calpha bond
    rms_eps: float = 1e-30
    gradient_checkpointing: bool = False  # recompute activations instead of storing them
