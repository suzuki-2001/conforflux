# Example: LAO (UniProt P02911)

A single-target example of ConforFlux on the lysine-arginine-ornithine binding protein. Five particles at σ = 2 Å produce a bimodal set of conformations: three near the closed reference (PDB 6MLP_E), two near the open reference (PDB 6ML0_A).

## Files

| Path | Purpose |
|---|---|
| `input.yaml` | Boltz input — sequence + MSA path. |
| `msa.a3m` | Unpaired MSA from ColabFold MMseqs2. |
| `refs/6MLP_E.pdb` | Closed-state reference (chain E of PDB 6MLP). |
| `refs/6ML0_A.pdb` | Open-state reference (chain A of PDB 6ML0). |
| `expected_output/` | The five output structures pre-computed on an NVIDIA H100 with seed 0, so you can inspect them without rerunning. |

## Run

```bash
conforflux predict input.yaml \
    --out_dir output \
    --num_particles 5 \
    --sigma 2.0 \
    --output_format pdb \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --seed 0
```

## Expected RMSDs (NVIDIA H100, ConforFlux v0.1.0, seed 0)

```
sample                     →6MLP (closed)       →6ML0 (open)
input_model_0.pdb                   0.379              4.944
input_model_1.pdb                   0.555              5.116
input_model_2.pdb                   0.365              5.082
input_model_3.pdb                   5.569              1.288
input_model_4.pdb                   7.014              2.470
min                                 0.365              1.288
```

Results vary by target and seed.
