"""ConforMix's `filter_unphysical_traj` and `filter_by_plddt_quality`, on structure files."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click
import gemmi
import numpy as np
from scipy.spatial import cKDTree

from conforflux.structures import read_structure

SUFFIXES = (".cif", ".mmcif", ".pdb")


def _structures(paths: tuple[str, ...]) -> list[tuple[Path, Path]]:
    found = []
    for arg in map(Path, paths):
        if arg.is_dir():
            root = arg.resolve()
            files = sorted(p for p in root.rglob("*") if p.suffix in SUFFIXES)
            found += [(p, p.relative_to(root.parent)) for p in files]
        else:
            found.append((arg, Path(arg.name)))
    return found


def geometry(
    st: gemmi.Structure, ca_max: float = 4.5, cn_max: float = 2.0, clash: float = 0.5
) -> tuple[bool, bool, bool]:
    ok_ca = ok_cn = ok_clash = True
    for chain in st[0]:
        residues = list(chain)
        for a, b in zip(residues[:-1], residues[1:]):
            if b.seqid.num - a.seqid.num != 1:
                continue
            ca_a, ca_b = a.find_atom("CA", "*"), b.find_atom("CA", "*")
            if ca_a and ca_b and ca_a.pos.dist(ca_b.pos) >= ca_max:
                ok_ca = False
            c, n = a.find_atom("C", "*"), b.find_atom("N", "*")
            if c and n and c.pos.dist(n.pos) >= cn_max:
                ok_cn = False
        xyz, owner = [], []
        for k, res in enumerate(r for r in residues if r.find_atom("CA", "*")):
            for atom in res:
                xyz.append(atom.pos.tolist())
                owner.append(k)
        if xyz:
            pairs = cKDTree(np.asarray(xyz)).query_pairs(clash, output_type="ndarray")
            owner = np.asarray(owner)
            if len(pairs) and (np.abs(owner[pairs[:, 0]] - owner[pairs[:, 1]]) >= 3).any():
                ok_clash = False
    return ok_ca, ok_cn, ok_clash


def plddt(st: gemmi.Structure) -> np.ndarray:
    """Per-residue pLDDT in [0, 1], from the Calpha B-factor the predictors write."""
    values = []
    for chain in st[0]:
        for res in chain:
            ca = res.find_atom("CA", "*")
            if ca:
                values.append(ca.b_iso)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no Calpha atoms to read pLDDT from")
    return values / 100.0 if values.max() > 1.0 else values


def plddt_drop(sample: np.ndarray, reference: np.ndarray, window: int) -> float:
    """Largest drop of a windowed mean pLDDT below the reference's."""
    kernel = np.ones(window) / window
    ref = np.convolve(reference, kernel, mode="valid")
    return float((ref - np.convolve(sample, kernel, mode="valid")).max())


def reference_plddt(paths) -> np.ndarray:
    """Per-residue minimum over the unguided predictions."""
    refs = [plddt(read_structure(p)) for p in paths]
    if len({len(r) for r in refs}) != 1:
        raise ValueError("reference structures differ in residue count")
    return np.min(refs, axis=0)


def check(
    path,
    ref_plddt: np.ndarray | None = None,
    window: int = 10,
    max_plddt_drop: float = 0.2,
    max_ca_ca: float = 4.5,
    max_c_n: float = 2.0,
    clash: float = 0.5,
) -> dict:
    st = read_structure(path)
    ok_ca, ok_cn, ok_clash = geometry(st, max_ca_ca, max_c_n, clash)
    p = plddt(st)
    row = {"ca_ca": ok_ca, "c_n": ok_cn, "clash": ok_clash, "plddt_drop": None, "plddt": True}
    if ref_plddt is not None:
        if len(p) != len(ref_plddt):
            raise ValueError(f"{path}: {len(p)} residues, reference {len(ref_plddt)}")
        row["plddt_drop"] = plddt_drop(p, ref_plddt, window)
        row["plddt"] = row["plddt_drop"] <= max_plddt_drop
    row["keep"] = ok_ca and ok_cn and ok_clash and row["plddt"]
    row["mean_plddt"] = float(p.mean())
    return row


def write_tsv(rows: list[dict], fh) -> None:
    cols = list(rows[0]) if rows else []
    fh.write("\t".join(cols) + "\n")
    for row in rows:
        fh.write("\t".join(_fmt(row[c]) for c in cols) + "\n")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


@click.command()
@click.argument("samples", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--reference",
    multiple=True,
    type=click.Path(exists=True),
    help="Unguided predictions of the same input. Without it pLDDT is not checked.",
)
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="TSV, default stdout.")
@click.option(
    "--keep_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Copy the structures that pass here.",
)
@click.option("--window", type=int, default=10, show_default=True)
@click.option("--max_plddt_drop", type=float, default=0.2, show_default=True)
@click.option("--max_ca_ca", type=float, default=4.5, show_default=True)
@click.option("--max_c_n", type=float, default=2.0, show_default=True)
@click.option("--clash", type=float, default=0.5, show_default=True)
def main(samples, reference, out, keep_dir, window, max_plddt_drop, max_ca_ca, max_c_n, clash):
    """Filter predicted structures by backbone geometry, clashes and pLDDT."""
    ref_plddt = None
    if reference:
        try:
            ref_plddt = reference_plddt([p for p, _ in _structures(reference)])
        except ValueError as e:
            raise click.ClickException(str(e))

    found = _structures(samples)
    rels = [rel for _, rel in found]
    if keep_dir and len(set(rels)) < len(rels):
        raise click.ClickException(
            "two inputs share a relative path; pass their parent directories"
        )
    rows = []
    for path, rel in found:
        try:
            row = check(path, ref_plddt, window, max_plddt_drop, max_ca_ca, max_c_n, clash)
        except ValueError as e:
            raise click.ClickException(str(e))
        rows.append({"path": str(path), **row})
        if row["keep"] and keep_dir:
            dest = Path(keep_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)

    if out:
        with open(out, "w") as fh:
            write_tsv(rows, fh)
    else:
        write_tsv(rows, sys.stdout)
    click.echo(f"kept {sum(r['keep'] for r in rows)} of {len(rows)}", err=True)


if __name__ == "__main__":
    main()
