"""Subsetting -- Part N of the Twinned FCC walkthrough (OPTIONAL).

Extracts one or more cuboidal sub-volumes from the fully twinned, cleaned
3D structure -- useful when you want to export/inspect a smaller RVE than
the whole cleaned domain. Off by default (a single 1x1x1, 20%-length tile
is effectively a no-op unless you actually want fewer/larger tiles);
Visualization & Export works fine straight off the full cleaned structure
without ever calling this.
"""


def generate_subsets(cleaner, start_pct=(0.0, 0.0, 0.0), n_cuboids=(1, 1, 1),
                      length_pct=(20.0, 20.0, 20.0), stride_pct=(20.0, 20.0, 20.0),
                      overflow_policy='clip'):
    """Generates the tile grid then crops the cleaned structure to each
    tile. `start_pct`/`length_pct`/`stride_pct` are (x, y, z) percent-of-
    domain tuples; `overflow_policy`: 'clip' (default, clip to the domain
    boundary) or 'skip' (drop any tile that would overflow the domain
    instead of clipping it).

    Returns
    -------
    list of SubsetCleaner : one per generated tile -- each carries
    lgi_clean/twin_role_clean/twin_parent_of_clean/all_quats_clean just
    like the source StructureCleaner3D, ready for
    steps_visualization_export's functions.
    """
    from upxo.pxtal.twinned_simple_3d.subsetting import generate_subset_tiles, crop_cleaner

    axes = ('x', 'y', 'z')
    start = dict(zip(axes, start_pct))
    n_cub = dict(zip(axes, n_cuboids))
    length = dict(zip(axes, length_pct))
    stride = dict(zip(axes, stride_pct))

    tiles = generate_subset_tiles(cleaner.lgi_clean.shape, start, n_cub, length, stride, overflow_policy)
    return [crop_cleaner(cleaner, tile) for tile in tiles]
