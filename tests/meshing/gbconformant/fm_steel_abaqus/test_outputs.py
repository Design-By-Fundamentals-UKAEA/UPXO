"""Boundary node sets, the constraint template and the mesh files of the deck."""
import itertools
import json
import re
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import (
    DeckSettings, boundary_nodes, periodic_pairs, tetrahedron_fields, write_msh, write_vtu)
from upxo.meshing.gbconformant.fm_steel_abaqus.constraints import template_text
from .helpers import SHAPE, voxel_tets, work_dir
from .test_deck import build


def brute_force_node_features(labels):
    """{grid node number (1-based, voxel_tets order): set of labels of the voxels around it}."""
    nx, ny, nz = labels.shape
    around = {}
    for i, j, k in itertools.product(range(nx + 1), range(ny + 1), range(nz + 1)):
        voxels = {int(labels[a, b, c])
                  for a in (i - 1, i) for b in (j - 1, j) for c in (k - 1, k)
                  if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz}
        around[(i * (ny + 1) + j) * (nz + 1) + k + 1] = voxels
    return around


def read_nsets(path):
    sets, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith('*Nset'):
            current = re.search(r'nset=([^,\s]+)', line).group(1)
            sets[current] = []
        elif current and line and not line.startswith('*'):
            sets[current] += [int(v) for v in line.split(',') if v.strip()]
    return sets


FULL = DeckSettings(boundary_sets=True, write_msh=True, write_vtu=True)


class BoundaryNodeTests(unittest.TestCase):
    def test_boundary_and_junction_nodes_match_the_voxel_neighbourhood(self):
        with work_dir() as root:
            hier, _, (pts, tets, grains), res = build(root, settings=FULL)
            sets = read_nsets(res['directory'] / '04b_nsets_boundaries.inp')
            around = brute_force_node_features(hier.lgi)
            self.assertEqual(sorted(sets['ns_gb_all']), sorted(n for n, s in around.items() if len(s) >= 2))
            self.assertEqual(sorted(sets['ns_junction_all']), sorted(n for n, s in around.items() if len(s) >= 3))
            feature_one = sorted(n for n, s in around.items() if 1 in s and len(s) >= 2)
            self.assertEqual(sorted(sets['ns_gb_es_blk_1_1_1']), feature_one)
            counts = res['info']['boundary_sets']
            self.assertEqual(counts['grain_boundary_nodes'], len(sets['ns_gb_all']))
            self.assertEqual(counts['junction_nodes'], len(sets['ns_junction_all']))

    def test_sets_are_instance_qualified_and_included_in_the_assembly(self):
        with work_dir() as root:
            _, _, _, res = build(root, settings=FULL)
            deck = res['directory']
            text = (deck / '04b_nsets_boundaries.inp').read_text()
            self.assertIn('*Nset, nset=ns_gb_all, instance=FM_STEEL_RVE-1', text)
            master = (deck / 'model_master.inp').read_text()
            self.assertLess(master.index('04_nsets_bc.inp'), master.index('04b_nsets_boundaries.inp'))
            self.assertLess(master.index('04b_nsets_boundaries.inp'), master.index('*End Assembly'))

    def test_sets_are_not_written_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            deck = res['directory']
            self.assertFalse((deck / '04b_nsets_boundaries.inp').exists())
            self.assertNotIn('04b_nsets', (deck / 'model_master.inp').read_text())
            self.assertIsNone(res['info']['boundary_sets'])

    def test_merged_features_have_no_boundary_between_them(self):
        with work_dir() as root:
            hier, _, _, res = build(root, settings=FULL, merge={2: 1})
            sets = read_nsets(res['directory'] / '04b_nsets_boundaries.inp')
            merged = hier.lgi.copy()
            merged[merged == 2] = 1
            around = brute_force_node_features(merged)
            self.assertEqual(sorted(sets['ns_gb_all']), sorted(n for n, s in around.items() if len(s) >= 2))
            per_feature = [name for name in sets if name.startswith('ns_gb_es_')]
            self.assertEqual(len(per_feature), 4)                # five features, two of them merged into one

    def test_a_single_feature_has_no_boundary_nodes(self):
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        found = boundary_nodes(tets, np.array([7, 7]))
        self.assertEqual(len(found.grain_boundary), 0)
        self.assertEqual(found.by_feature, {})

    def test_bad_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            boundary_nodes(np.zeros((3, 3), dtype=int), np.zeros(3, dtype=int))


class ConstraintTemplateTests(unittest.TestCase):
    def test_the_structured_mesh_matches_opposite_faces_node_by_node(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root)
            nx, ny, nz = SHAPE
            found = res['info']['constraints']['periodic_pairs']
            self.assertEqual(found['x']['matched_pairs'], (ny + 1) * (nz + 1))
            self.assertEqual(found['y']['matched_pairs'], (nx + 1) * (nz + 1))
            self.assertEqual(found['z']['matched_pairs'], (nx + 1) * (ny + 1))
            self.assertFalse(res['info']['constraints']['active'])

    def test_the_file_is_comments_only_and_names_the_deck_sets(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            text = (res['directory'] / '07_interactions.inp').read_text()
            for line in text.splitlines():
                self.assertTrue(line.startswith('**'), line)
            self.assertIn('ns_xminus', text)
            self.assertIn('matched pairs', text)
            self.assertIn('FM_STEEL_RVE-1.', text)
            self.assertNotIn('TODO', text)

    def test_pairs_are_one_to_one(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root)
            faces = {}
            for key, axis, value in (('XMIN', 0, pts[:, 0].min()), ('XMAX', 0, pts[:, 0].max()),
                                     ('YMIN', 1, pts[:, 1].min()), ('YMAX', 1, pts[:, 1].max()),
                                     ('ZMIN', 2, pts[:, 2].min()), ('ZMAX', 2, pts[:, 2].max())):
                faces[key] = np.flatnonzero(np.isclose(pts[:, axis], value)) + 1
            pairs = periodic_pairs(pts, faces, 1e-6)
            for item in pairs.values():
                self.assertEqual(len(set(item['pairs'][:, 0])), len(item['pairs']))
                self.assertEqual(len(set(item['pairs'][:, 1])), len(item['pairs']))

    def test_faces_with_different_nodes_report_no_pair(self):
        pts = np.array([[0, 0, 0], [0, 1, 0], [1, 0.5, 0.3]], dtype=float)
        faces = {key: np.array([1]) for key in ('XMIN', 'YMIN', 'ZMIN', 'YMAX', 'ZMAX')}
        faces['XMAX'] = np.array([3])
        pairs = periodic_pairs(pts, faces, 1e-6)
        self.assertEqual(len(pairs['x']['pairs']), 0)
        text = template_text(pairs, DeckSettings().nset_names, 'INST-1', 'x')
        self.assertIn('0 matched pairs', text)
        self.assertIn('<low node>', text)


def reference_fields(root):
    hier, _, (pts, tets, grains), res = build(root, settings=FULL)
    names = {label: f'es_blk_{"_".join(map(str, hier.features[label].key))}' for label in hier.features}
    fields = tetrahedron_fields(grains, hier.features, names)
    return hier, pts, tets, grains, fields, res


class MeshFileTests(unittest.TestCase):
    def test_fields_follow_the_feature_of_each_tetrahedron(self):
        with work_dir() as root:
            hier, pts, tets, grains, fields, _ = reference_fields(root)
            self.assertEqual(fields['feature_label'].tolist(), grains.tolist())
            for index in (0, len(grains) // 2, len(grains) - 1):
                feature = hier.features[int(grains[index])]
                np.testing.assert_allclose(fields['euler_deg'][index], feature.euler)
                self.assertEqual(int(fields['global_id'][index]), feature.global_id)
            for letter in 'xyz':
                colours = fields[f'ipf_{letter}']
                self.assertEqual(colours.shape, (len(grains), 3))
                self.assertTrue(((colours >= 0) & (colours <= 1)).all())

    def test_the_deck_writes_both_files_beside_the_inp_files(self):
        with work_dir() as root:
            _, _, _, _, _, res = reference_fields(root)
            deck = res['directory']
            self.assertTrue((deck / 'fm_mesh.msh').is_file())
            self.assertTrue((deck / 'fm_mesh.vtu').is_file())
            self.assertEqual(res['info']['mesh_files'], ['fm_mesh.msh', 'fm_mesh.vtu'])
            self.assertEqual(sorted(p.name for p in deck.iterdir()), sorted(res['files']))

    def test_the_files_are_not_written_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            self.assertFalse((res['directory'] / 'fm_mesh.msh').exists())
            self.assertEqual(res['info']['mesh_files'], [])

    def test_msh_holds_the_deck_mesh_and_the_orientations(self):
        with work_dir() as root:
            hier, pts, tets, grains, fields, res = reference_fields(root)
            lines = (res['directory'] / 'fm_mesh.msh').read_text().splitlines()
            self.assertEqual(lines[:3], ['$MeshFormat', '2.2 0 8', '$EndMeshFormat'])
            at = lines.index('$PhysicalNames')
            self.assertEqual(int(lines[at + 1]), len(hier.features))
            self.assertIn('3 1 "es_blk_1_1_1"', lines)
            at = lines.index('$Nodes')
            self.assertEqual(int(lines[at + 1]), len(pts))
            first = lines[at + 2].split()
            self.assertEqual(first[0], '1')
            at = lines.index('$Elements')
            self.assertEqual(int(lines[at + 1]), len(tets))
            row = lines[at + 2 + 5].split()                       # element 6
            self.assertEqual(row[:3], ['6', '4', '2'])
            self.assertEqual(int(row[3]), int(grains[5]))
            self.assertEqual([int(v) for v in row[5:]], (tets[5] + 1).tolist())
            at = lines.index('"euler_deg"')
            self.assertEqual(lines[at - 2], '$ElementData')
            self.assertEqual(lines[at + 1:at + 7], ['1', '0.0', '3', '0', '3', str(len(tets))])
            values = [float(v) for v in lines[at + 7 + 5].split()]
            self.assertEqual(int(values[0]), 6)
            np.testing.assert_allclose(values[1:], hier.features[int(grains[5])].euler, atol=1e-5)
            for name in ('feature_label', 'global_id', 'ipf_x', 'ipf_y', 'ipf_z', 'pag_id', 'packet_id', 'block_id',
                         'subblock_id'):
                self.assertIn(f'"{name}"', lines)
            self.assertEqual(lines.count('$EndElementData'), 10)

    def test_msh_coordinates_are_in_the_deck_units(self):
        with work_dir() as root:
            _, _, _, _, _, res = reference_fields(root)
            deck = res['directory']
            scale = res['info']['coordinate_scale']
            msh_lines = (deck / 'fm_mesh.msh').read_text().splitlines()
            at = msh_lines.index('$Nodes')
            last = [float(v) for v in msh_lines[at + 1 + int(msh_lines[at + 1])].split()]
            inp_last = [float(v) for v in (deck / '01_nodes.inp').read_text().splitlines()[-1].split(',')]
            np.testing.assert_allclose(last[1:], inp_last[1:], rtol=1e-6)
            self.assertAlmostEqual(last[1], SHAPE[0] * scale, places=6)

    def test_vtu_reads_back_with_the_cell_data(self):
        import pyvista as pv
        with work_dir() as root:
            hier, pts, tets, grains, fields, res = reference_fields(root)
            grid = pv.read(str(res['directory'] / 'fm_mesh.vtu'))
            self.assertEqual(grid.n_cells, len(tets))
            self.assertEqual(grid.n_points, len(pts))
            self.assertEqual(grid.cell_data['feature_label'].tolist(), grains.tolist())
            np.testing.assert_allclose(grid.cell_data['euler_deg'][3], hier.features[int(grains[3])].euler, atol=1e-6)
            for letter in 'xyz':
                data = grid.cell_data[f'ipf_{letter}']
                self.assertEqual(data.dtype, np.uint8)
                self.assertEqual(data.shape, (len(tets), 3))
            self.assertEqual({int(c) for c in grid.celltypes}, {10})

    def test_writers_return_the_path(self):
        with work_dir() as root:
            _, pts, tets, grains, fields, _ = reference_fields(root)
            self.assertEqual(write_msh(root / 'a.msh', pts, tets, fields), root / 'a.msh')
            self.assertEqual(write_vtu(root / 'a.vtu', pts, tets, fields), root / 'a.vtu')


if __name__ == '__main__':
    unittest.main()
