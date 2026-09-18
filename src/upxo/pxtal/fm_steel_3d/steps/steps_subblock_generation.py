"""Sub-block Generation -- Part G of the FM Steel 3D walkthrough (ADVANCED
TIER ONLY -- adv0/adv1; bas/int stop at Block Generation).

Further subdivides each block into sub-blocks with a small intra-block
orientation spread, for a finer conformal-meshing-ready hierarchy.
"""


def generate_subblocks(fm_ori, subblock_thickness_range_um=(0.5, 1.5),
                        intrablock_ori_spread_deg=2.0, thin_block_strategy='skip',
                        random_seed=None, subblock_slab_connectivity=26):
    """Subdivides every block in `fm_ori` into sub-blocks.

    Returns
    -------
    FMSteel3DWithSubBlocks
    """
    return fm_ori.generate_subblocks(
        subblock_thickness_range_um=subblock_thickness_range_um,
        intrablock_ori_spread_deg=intrablock_ori_spread_deg,
        thin_block_strategy=thin_block_strategy, random_seed=random_seed,
        subblock_slab_connectivity=subblock_slab_connectivity)
