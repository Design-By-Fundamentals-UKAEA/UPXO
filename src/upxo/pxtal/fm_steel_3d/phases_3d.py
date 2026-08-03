"""
phases_3d.py

Phase identifiers for FM (ferritic-martensitic) steel microstructures.

This is deliberately a small, standalone module: phase identity is a core
pipeline concept (which PAG transformed to martensite vs. which one stayed
untransformed retained austenite), not something invented at mesh-export
time -- so it needs to be importable by the PAG/orientation/block classes
themselves, not just by mesh_exporter_3d.py.

Scope
-----
Two phases are currently modelled, matching what generate_pags()'s
retained-austenite mechanism actually produces:

  PHASE_MARTENSITE (2)
      Transformed PAGs: full PAG -> packet -> block (-> sub-block)
      hierarchy, BCC, KS-derived orientation from the PAG's own FCC
      orientation.
  PHASE_RETAINED_AUSTENITE (1)
      Untransformed PAGs: whole-PAG granularity only (no packets/blocks/
      sub-blocks), FCC, orientation = the PAG's own orientation (the same
      one assigned to every PAG uniformly by assign_pag_orientations,
      before retained-austenite PAGs are singled out).

Other phases are physically plausible in ferritic-martensitic steel voxel
models (delta-ferrite: BCC, whole-grain, independent orientation with no
KS relationship to any parent PAG; bainite: BCC, partial/reduced
sub-structure) but are out of scope until a real use case needs them --
adding one means giving it its own id here plus its own geometry-selection
and orientation-assignment path elsewhere in the pipeline.
"""

PHASE_MARTENSITE = 2
PHASE_RETAINED_AUSTENITE = 1

PHASE_NAMES = {
    PHASE_MARTENSITE: 'martensite',
    PHASE_RETAINED_AUSTENITE: 'retained_austenite',
}


def retained_austenite_voxels_and_orientations(fm_state):
    """Grain-level {grain_id: voxels} + {grain_id: euler_angles} for every
    retained-austenite grain in fm_state (PAG-covered or leftover isolated).

    Shared by FMSteel3DWithOrientations/FMSteel3DWithSubBlocks (duck-typed:
    any object exposing grain_locs, clusters_dict, pag_orientations,
    retained_austenite_pag_ids, isolated_grains, get_isolated_grain_orientation
    works). Retained austenite has no packet/block/sub-block subdivision
    (see module docstring), so this is always grain granularity regardless
    of how far the rest of the structure has progressed through the pipeline.

    Calls ensure_isolated_grain_orientations() first (if available) so the
    non-negotiable "every non-void feature has an orientation" guarantee
    holds here too, rather than silently dropping grains with no orientation.
    """
    if hasattr(fm_state, 'ensure_isolated_grain_orientations'):
        fm_state.ensure_isolated_grain_orientations()

    grain_locs = fm_state.grain_locs
    clusters_dict = fm_state.clusters_dict
    pag_orientations = fm_state.pag_orientations
    retained_ids = getattr(fm_state, 'retained_austenite_pag_ids', set())

    features = {}
    orientations = {}

    for pag_id in retained_ids:
        ori = pag_orientations.get(pag_id)
        if ori is None:
            continue
        for gid in clusters_dict.get(pag_id, []):
            if gid in grain_locs:
                features[gid] = grain_locs[gid]
                orientations[gid] = ori

    get_iso_ori = getattr(fm_state, 'get_isolated_grain_orientation', None)
    for gid in getattr(fm_state, 'isolated_grains', set()) or ():
        if gid in features or gid not in grain_locs:
            continue
        ori = get_iso_ori(gid) if get_iso_ori is not None else None
        if ori is None:
            continue
        features[gid] = grain_locs[gid]
        orientations[gid] = ori

    return features, orientations


__all__ = [
    'PHASE_MARTENSITE', 'PHASE_RETAINED_AUSTENITE', 'PHASE_NAMES',
    'retained_austenite_voxels_and_orientations',
]
