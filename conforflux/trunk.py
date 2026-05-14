from __future__ import annotations

import torch
from torch import Tensor


def get_ca_indices(feats: dict) -> Tensor:
    if "ref_atom_name_chars" in feats:
        name_chars = feats["ref_atom_name_chars"]
        while name_chars.dim() > 3:
            name_chars = name_chars[0]
        if name_chars.dim() == 3:
            char_indices = name_chars.argmax(dim=-1)
            c_idx = ord("C") - ord(" ")
            a_idx = ord("A") - ord(" ")
            one_idx = ord("1") - ord(" ")
            prime_idx = ord("'") - ord(" ")
            space_idx = 0
            is_ca = (
                (char_indices[:, 0] == c_idx)
                & (char_indices[:, 1] == a_idx)
                & (char_indices[:, 2] == space_idx)
                & (char_indices[:, 3] == space_idx)
            )
            is_c1p = (
                (char_indices[:, 0] == c_idx)
                & (char_indices[:, 1] == one_idx)
                & (char_indices[:, 2] == prime_idx)
                & (char_indices[:, 3] == space_idx)
            )
            indices = torch.where(is_ca | is_c1p)[0]
            if len(indices) > 0:
                return indices
    a2t_key = "atom_to_token" if "atom_to_token" in feats else "atom_to_token_index"
    if a2t_key in feats:
        a2t = feats[a2t_key]
        while a2t.dim() > 2:
            a2t = a2t[0]
        token_ids = a2t.argmax(dim=-1) if a2t.dim() == 2 else a2t
        ca: list[int] = []
        for token_id in range(int(token_ids.max()) + 1):
            atoms = torch.where(token_ids == token_id)[0]
            if len(atoms) >= 2:
                ca.append(atoms[1].item())
        return torch.tensor(ca, dtype=torch.long)
    raise ValueError("Cannot determine CA/C1' indices from features.")


def run_trunk(pl_module, batch):
    # Exit Lightning's inference_mode so the trunk outputs survive into autograd.
    with torch.inference_mode(mode=False), torch.no_grad():
        s_inputs = pl_module.input_embedder(batch)
        s_init = pl_module.s_init(s_inputs)
        z_init = (
            pl_module.z_init_1(s_inputs)[:, :, None]
            + pl_module.z_init_2(s_inputs)[:, None, :]
        )
        rel_pos_enc = pl_module.rel_pos(batch)
        z_init = z_init + rel_pos_enc
        z_init = z_init + pl_module.token_bonds(batch["token_bonds"].float())
        if pl_module.bond_type_feature:
            z_init = z_init + pl_module.token_bonds_type(batch["type_bonds"].long())
        z_init = z_init + pl_module.contact_conditioning(batch)
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = batch["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        recycling_steps = pl_module.predict_args.get("recycling_steps", 3)
        for _ in range(recycling_steps + 1):
            s = s_init + pl_module.s_recycle(pl_module.s_norm(s))
            z = z_init + pl_module.z_recycle(pl_module.z_norm(z))
            if pl_module.use_templates:
                tmpl = (
                    pl_module.template_module._orig_mod
                    if getattr(pl_module, "is_template_compiled", False) and not pl_module.training
                    else pl_module.template_module
                )
                z = z + tmpl(z, batch, pair_mask, use_kernels=pl_module.use_kernels)
            if not getattr(pl_module, "no_msa", False):
                msa = (
                    pl_module.msa_module._orig_mod
                    if getattr(pl_module, "is_msa_compiled", False) and not pl_module.training
                    else pl_module.msa_module
                )
                z = z + msa(z, s_inputs, batch, use_kernels=pl_module.use_kernels)
            pf = (
                pl_module.pairformer_module._orig_mod
                if getattr(pl_module, "is_pairformer_compiled", False) and not pl_module.training
                else pl_module.pairformer_module
            )
            s, z = pf(s, z, mask=mask, pair_mask=pair_mask, use_kernels=pl_module.use_kernels)
    return s, z, s_inputs, rel_pos_enc
