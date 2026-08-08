# ConforFlux container

`BACKBONE` selects which backbone the image installs — `boltz` (default), `protenix` or
`openfold3` — and becomes the image's entry point. One backbone per image.

Model weights are downloaded on first run into the backbone's cache (`~/.boltz`,
`~/.openfold3`); mount it to persist them.

## Docker

```bash
docker build --build-arg BACKBONE=boltz -f container/Dockerfile -t conforflux:boltz .

docker run --rm --gpus all --shm-size=8g \
    -v ~/.boltz:/root/.boltz \
    -v $(pwd)/in:/work:ro -v $(pwd)/out:/out -w /work \
    conforflux:boltz predict input.yaml --out_dir /out \
    --num_particles 5 --sigma 2.5 --seed 42 \
    --recycling_steps 3 --sampling_steps 200 --output_format pdb
```

`--shm-size=8g` is required for PyTorch's DataLoader workers.

## Apptainer / Singularity

```bash
apptainer build --build-arg BACKBONE=boltz conforflux.sif container/Singularity.def

apptainer run --nv \
    --bind ~/.boltz:/root/.boltz \
    --bind $(pwd)/in:/work --bind $(pwd)/out:/out --pwd /work \
    conforflux.sif predict input.yaml --out_dir /out \
    --num_particles 5 --sigma 2.5 --alpha_s 0.02 --alpha_z 0.02 --seed 42 \
    --recycling_steps 3 --sampling_steps 200 --output_format pdb
```

The other two take their backbone's own arguments:

```bash
apptainer run --nv conforflux-protenix.sif pred -i input.json -o /out \
    --num_particles 5 --sigma 2.0 --alpha_s 0.02 --alpha_z 0.02

apptainer run --nv conforflux-openfold3.sif predict \
    --query_json input.json --output_dir /out --runner_yaml of3_runner.yaml --seeds 101 \
    --num_particles 5 --sigma 2.0 --alpha_s 0.02 --alpha_z 0.02
```

Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` if your home partition lacks build space.
