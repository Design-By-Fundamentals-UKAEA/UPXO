"""Block Generation -- Part F of the FM Steel 3D walkthrough.

Slices each PAG into martensitic blocks, then assigns each block a
Kurdjumov-Sachs (KS) variant orientation.
"""


def generate_blocks(fm_pag, block_thickness_range=(2.0, 5.0), random_seed=None,
                     block_slab_connectivity=26):
    """Slices each of `fm_pag`'s PAGs into blocks.

    Returns
    -------
    FMSteel3DWithBlocks
    """
    return fm_pag.generate_blocks(
        block_thickness_range=block_thickness_range, random_seed=random_seed,
        block_slab_connectivity=block_slab_connectivity)


def assign_orientations(fm_blk, ks_variant_selection='random_per_block', random_seed=None):
    """Assigns a Kurdjumov-Sachs variant orientation to every block.
    Requires PAG orientations to already be set (steps_pag_clustering.
    assign_pag_orientations).

    Returns
    -------
    FMSteel3DWithOrientations
    """
    return fm_blk.assign_orientations(
        ks_variant_selection=ks_variant_selection, random_seed=random_seed)
