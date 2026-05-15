# ConforFlux container

Docker and Apptainer/Singularity definition files for reproducible runs. Both build on `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel` and pin cuequivariance 0.9.1, matching the env used to generate `examples/expected_output/` byte-for-byte.

Boltz-2 model weights (~8 GB) are downloaded on first run into `~/.boltz`. Mount this directory to persist them across runs.

## Docker

```bash
docker build -f container/Dockerfile -t conforflux .

mkdir -p out
docker run --rm --gpus all --shm-size=8g \
    -v ~/.boltz:/root/.boltz \
    -v $(pwd)/examples:/work:ro \
    -v $(pwd)/out:/out \
    -w /work \
    conforflux predict input.yaml --out_dir /out \
    --num_particles 5 --sigma 2.5 --seed 42 \
    --recycling_steps 3 --sampling_steps 200 --output_format pdb
```

`--shm-size=8g` is required so PyTorch's DataLoader workers can use shared memory.

## Apptainer / Singularity

```bash
apptainer build conforflux.sif container/Singularity.def

mkdir -p out
apptainer run --nv \
    --bind ~/.boltz:/root/.boltz \
    --bind $(pwd)/examples:/work \
    --bind $(pwd)/out:/out \
    --pwd /work \
    conforflux.sif predict input.yaml --out_dir /out \
    --num_particles 5 --sigma 2.5 --seed 42 \
    --recycling_steps 3 --sampling_steps 200 --output_format pdb
```

If your home partition lacks space for the Apptainer build cache, set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to a larger filesystem before `apptainer build`.
