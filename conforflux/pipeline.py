from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from conforflux.config import ConforFluxConfig
from conforflux.filter import SUFFIXES, check, reference_plddt, write_tsv
from conforflux.structures import write_ensemble

SIGMAS = (0.5, 1.0, 1.5, 2.0, 2.5)


def _structures(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in SUFFIXES)


def run(
    sample: Callable[[Path, int, int, ConforFluxConfig | None], None],
    out_dir: Path,
    cfg: ConforFluxConfig,
    target_samples: int,
    num_particles: int,
    plddt_filter: bool = False,
    num_reference: int = 5,
    sigmas: tuple[float, ...] = SIGMAS,
    max_rounds: int = 20,
    seed: int = 0,
) -> list[Path]:
    """`sample(dir, seed, n, cfg)` writes one round; `cfg` None is unguided."""
    out_dir = Path(out_dir)
    ref_plddt = None
    if plddt_filter:
        ref_dir = out_dir / "unguided"
        sample(ref_dir, seed, num_reference, None)
        refs = _structures(ref_dir)
        if not refs:
            raise RuntimeError(f"the unguided round wrote no structures under {ref_dir}")
        ref_plddt = reference_plddt(refs)

    rows = []
    for r in range(max_rounds):
        if sum(row["keep"] for row in rows) >= target_samples:
            break
        round_seed = seed + 1 + r
        sigma = sigmas[r % len(sigmas)]
        tag = f"round{r:03d}"
        round_dir = out_dir / "rounds" / tag
        sample(round_dir, round_seed, num_particles, replace(cfg, sigma=sigma))
        for path in _structures(round_dir):
            row = {"round": tag, "seed": round_seed, "sigma": sigma, "path": str(path)}
            rows.append({**row, **check(path, ref_plddt)})

    kept = sorted((row for row in rows if row["keep"]), key=lambda row: -row["mean_plddt"])
    kept = kept[:target_samples]
    selected = {row["path"] for row in kept}
    for row in rows:
        row["selected"] = row["path"] in selected

    kept_dir = out_dir / "kept"
    kept_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in kept:
        dest = kept_dir / f"{row['round']}_{Path(row['path']).name}"
        shutil.copyfile(row["path"], dest)
        paths.append(dest)
    if paths:
        write_ensemble(paths, out_dir / "kept.cif")
    with open(out_dir / "samples.tsv", "w") as fh:
        write_tsv(rows, fh)
    settings = {
        **{k: v for k, v in asdict(cfg).items() if k != "sigma"},
        "target_samples": target_samples,
        "num_particles": num_particles,
        "plddt_filter": plddt_filter,
        "num_reference": num_reference if plddt_filter else 0,
        "sigmas": list(sigmas),
        "max_rounds": max_rounds,
        "seed": seed,
        "rounds": len({row["round"] for row in rows}),
        "kept": len(paths),
    }
    (out_dir / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"[ConforFlux] {len(paths)} of {target_samples} samples kept in {kept_dir}", flush=True)
    return paths
