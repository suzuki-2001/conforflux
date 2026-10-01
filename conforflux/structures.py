from __future__ import annotations

from pathlib import Path

import gemmi


def read_structure(path: str | Path) -> gemmi.Structure:
    path = Path(path)
    if path.suffix in (".cif", ".mmcif"):
        block = gemmi.cif.read(str(path))[0]
        loop = block.find_loop("_atom_site.id").get_loop()
        # OpenFold3 writes no occupancy column, and gemmi < 0.7 then reads no atoms.
        if loop is not None and "_atom_site.occupancy" not in loop.tags:
            loop.add_columns(["_atom_site.occupancy"], "1.0")
        st = gemmi.make_structure_from_block(block)
    else:
        st = gemmi.read_structure(str(path))
    if len(st) == 0 or st[0].count_atom_sites() == 0:
        raise ValueError(f"no atoms read from {path}")
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    return st


def write_ensemble(paths, out: str | Path) -> None:
    st = read_structure(paths[0])
    for k, path in enumerate(paths[1:], 2):
        model = read_structure(path)[0]
        if hasattr(model, "num"):
            model.num = k
        else:
            model.name = str(k)
        st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(out))
