# LAO example (UniProt P02911)

A single-target example contrasting default Boltz-2 sampling and ConforFlux on the lysine-arginine-ornithine binding protein. At seed 42, default Boltz-2 collapses all five samples onto the closed reference (PDB 6MLP_E). ConforFlux at σ = 2.5 Å splits them across the closed and open (PDB 6ML0_A) references plus one off-path outlier.

## Files

| Path | Purpose |
|---|---|
| `input.yaml` | Boltz input — sequence + MSA path. |
| `msa.a3m` | Unpaired MSA from ColabFold MMseqs2. |
| `refs/6MLP_E.pdb` | Closed-state reference (chain E of PDB 6MLP). |
| `refs/6ML0_A.pdb` | Open-state reference (chain A of PDB 6ML0). |
| `expected_output/default/` | Five default Boltz-2 samples (seed 42, no guidance). |
| `expected_output/conforflux/` | Five ConforFlux particles (σ = 2.5 Å, seed 42). |

## Run

Default Boltz-2 (no diversification).

```bash
boltz predict input.yaml \
    --out_dir output_default \
    --diffusion_samples 5 \
    --recycling_steps 3 --sampling_steps 200 \
    --output_format pdb --seed 42
```

ConforFlux.

```bash
conforflux predict input.yaml \
    --out_dir output_conforflux \
    --num_particles 5 --sigma 2.5 \
    --recycling_steps 3 --sampling_steps 200 \
    --output_format pdb --seed 42
```

## Expected metrics (NVIDIA H100, seed 42)

Per-sample sequence-aligned Cα Kabsch RMSD and TM-score against each reference.

### Default Boltz-2 — all samples collapse onto the closed reference

| Sample | RMSD vs 6MLP (Å) | TM-score vs 6MLP | RMSD vs 6ML0 (Å) | TM-score vs 6ML0 |
|---|---:|---:|---:|---:|
| `input_model_0.pdb` | 0.385 | 0.996 | 4.931 | 0.709 |
| `input_model_1.pdb` | 0.371 | 0.996 | 4.851 | 0.706 |
| `input_model_2.pdb` | 0.373 | 0.996 | 4.952 | 0.692 |
| `input_model_3.pdb` | 0.373 | 0.996 | 4.935 | 0.708 |
| `input_model_4.pdb` | 0.391 | 0.995 | 4.889 | 0.700 |

### ConforFlux at σ = 2.5 Å — closed/open split plus one outlier

| Sample | RMSD vs 6MLP (Å) | TM-score vs 6MLP | RMSD vs 6ML0 (Å) | TM-score vs 6ML0 |
|---|---:|---:|---:|---:|
| `input_model_0.pdb` | 0.344 | 0.996 | 5.147 | 0.687 |
| `input_model_1.pdb` | 0.335 | 0.997 | 5.015 | 0.703 |
| `input_model_2.pdb` | 2.889 | 0.828 | 2.346 | 0.872 |
| `input_model_3.pdb` | 6.276 | 0.663 | 1.756 | 0.917 |
| `input_model_4.pdb` | 10.216 | 0.585 | 5.701 | 0.643 |

Results vary by target and seed.
