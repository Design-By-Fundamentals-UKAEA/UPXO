"""FM-steel Abaqus decks for grain-conformal tetrahedral meshes.

Reads a raw feature export folder (label arrays, orientations and the
``raw_export.pkl`` bundle), resolves which raw feature each meshed grain
stands for after cleaning and merging, and writes a partitioned C3D4 deck
with element sets, materials, sections, node sets and a static step.
"""
from .audit import IdentityError, location_audit, reallocation_report
from .boundary_sets import BoundaryNodes, boundary_nodes
from .comparison import (
    ComparisonSettings, compare_before_after, compare_junction_lines, feature_diameters,
    format_summary, report_summary)
from .constraints import periodic_pairs
from .export import DeckSettings, applied_strain, domain_lengths, write_deck
from .fidelity import grain_fidelity
from .hierarchy import LEVELS, LEVEL_LABELS, Hierarchy, detect_levels, load_hierarchy
from .identity import IdentityMap, resolve_identities
from .mesh_files import tetrahedron_fields, write_msh, write_vtu
from .mergelog import merge_log, write_merge_log
from .subset import (
    BoxSpec, cut_statistics, describe_box, extract_labels, extract_subset, next_index,
    new_indexed_directory, resolve_box, subset_record, write_subset_record)
from .texture import TextureSettings, density_difference, orientation_table

__all__ = [
    'applied_strain', 'domain_lengths',
    'BoxSpec', 'cut_statistics', 'describe_box', 'extract_labels', 'extract_subset', 'next_index',
    'new_indexed_directory', 'resolve_box', 'subset_record', 'write_subset_record',
    'ComparisonSettings', 'TextureSettings', 'compare_before_after', 'compare_junction_lines',
    'density_difference', 'feature_diameters', 'format_summary', 'orientation_table', 'report_summary',
    'grain_fidelity', 'merge_log', 'write_merge_log',
    'BoundaryNodes', 'boundary_nodes', 'periodic_pairs', 'tetrahedron_fields', 'write_msh', 'write_vtu',
    'DeckSettings', 'Hierarchy', 'IdentityError', 'IdentityMap', 'LEVELS',
    'LEVEL_LABELS', 'detect_levels', 'load_hierarchy', 'location_audit',
    'reallocation_report', 'resolve_identities', 'write_deck',
]
