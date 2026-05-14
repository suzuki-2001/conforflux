from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConforFluxConfig:
    sigma: float = 2.0  # RBF kernel bandwidth on Cα RMSD (Å)
    alpha_s: float = 0.02  # RMS-normalised step size for s_trunk
    alpha_z: float = 0.02  # RMS-normalised step size for z_trunk
    start_frac: float = 0.0  # fraction of the diffusion trajectory at which guidance starts
    stop_frac: float = 0.8  # fraction of the diffusion trajectory at which guidance stops
    update_interval: int = 3  # fire the gradient every K diffusion steps
    rms_eps: float = 1e-30
    noise_scale: bool = True  # scale the update by Boltz-2's EDM noise level
    max_offdiag_scale: bool = True  # scale by max off-diagonal kernel value (vanishes when spread)
    kernel_saturation_threshold: float = 0.01  # skip update when max off-diagonal < threshold
    resample: bool = True  # replace particles with broken Cα geometry
    resample_interval: int = 10
    bond_tol: float = 1.0  # Å around the expected backbone bond length
    gradient_checkpointing: bool = False
    gradient_checkpointing_threshold: int = 500  # auto-enable when N_ca exceeds this
