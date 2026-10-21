"""The before/after comparison of a run: the chosen pair, the units, the figures and the tables."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use('Agg')

from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import PARAMETERS
from upxo.meshing.gbconformant.fm_steel_abaqus.snapshots import (
    SURFACE_PREPARED, SURFACE_RAW, SURFACE_SMOOTHED, TETS, TITLES, VOXELS)
from upxo.meshing.gbconformant.pipeline import comparison_view as V
from upxo.meshing.gbconformant.pipeline.mesh_regions import mesh_surface
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, default_config
from upxo.meshing.gbconformant.pipeline.stage_volume import PRE_TET_POINTS, handle_tet_mesh

from ..fm_steel_abaqus.helpers import work_dir
from .test_results import merged_state, meshed_state


def surfaces_for(state):
    """Add the surface artefacts of a structured mesh: the open interfaces, and the closed surface with its points."""
    tets = state.get_artefact('verified_tets')
    interior = mesh_surface(tets.tetrahedra, tets.grain_ids, len(tets.points), include_rve_faces=False)
    state.store_artefact('interfaces_smoothed', SimpleNamespace(
        points=tets.points, original_points=tets.points * 1.5, triangles=interior.nodes,
        grain_pairs=interior.grains))
    closed = mesh_surface(tets.tetrahedra, tets.grain_ids, len(tets.points), include_rve_faces=True)
    pairs = closed.grains.copy()
    box = pairs[:, 0] == 0
    pairs[box, 0] = pairs[box, 1]
    state.store_artefact('repaired_rve', SimpleNamespace(points=tets.points, triangles=closed.nodes, grain_pairs=pairs))
    state.store_artefact(PRE_TET_POINTS, tets.points.copy())


class ViewSettingsTests(unittest.TestCase):
    def test_defaults(self):
        view = V.view_settings(default_config())
        self.assertEqual((view.before, view.after), (VOXELS, TETS))
        self.assertEqual((view.kind, view.units, view.bins), ('histogram', 'auto', 40))
        self.assertIsNone(view.bin_width)
        self.assertIsNone(view.bandwidth)
        self.assertEqual((view.color_before, view.color_after), ('#4878cf', '#d65f5f'))
        self.assertTrue(view.show_mean and view.show_std)
        self.assertFalse(view.show_median)

    def test_the_pair_must_run_in_pipeline_order(self):
        for before, after in ((TITLES[TETS], TITLES[VOXELS]), (TITLES[TETS], TITLES[TETS])):
            with self.assertRaisesRegex(ValueError, 'earlier in the pipeline'):
                V.view_settings(dict(default_config(), COMPARE_BEFORE=before, COMPARE_AFTER=after))
        view = V.view_settings(dict(default_config(), COMPARE_BEFORE=TITLES[SURFACE_RAW], COMPARE_AFTER=TITLES[SURFACE_SMOOTHED]))
        self.assertEqual((view.before, view.after), (SURFACE_RAW, SURFACE_SMOOTHED))

    def test_bad_values_are_refused(self):
        for key, value in (('COMPARE_COLOR_BEFORE', 'no such colour'), ('COMPARE_PLOT_KIND', 'pie'),
                           ('COMPARE_PLOT_UNITS', 'km'), ('COMPARE_MISORIENTATION_NORM', 'per grain'),
                           ('COMPARE_GRAIN_WEIGHT', 'mass')):
            with self.assertRaises(ValueError, msg=key):
                V.view_settings(dict(default_config(), **{key: value}))

    def test_automatic_numbers(self):
        for value in (None, '', 'auto', 0, -1):
            self.assertIsNone(V.view_settings(dict(default_config(), COMPARE_BIN_WIDTH=value)).bin_width)
        view = V.view_settings(dict(default_config(), COMPARE_BIN_WIDTH=2.5, COMPARE_KDE_BANDWIDTH=0.4))
        self.assertEqual((view.bin_width, view.bandwidth), (2.5, 0.4))

    def test_the_settings_are_in_the_table(self):
        self.assertEqual(SCHEMA['COMPARE_BEFORE'].choices, tuple(TITLES.values()))
        self.assertEqual(SCHEMA['COMPARE_AFTER'].choices, tuple(TITLES.values()))
        self.assertIn('histogram + kde', SCHEMA['COMPARE_PLOT_KIND'].choices)
        self.assertEqual(SCHEMA['COMPARE_PLOT_UNITS'].choices, ('auto', 'microns', 'mm', 'm'))
        self.assertTrue(SCHEMA['COMPARE_BIN_WIDTH'].auto and SCHEMA['COMPARE_KDE_BANDWIDTH'].auto)
        self.assertEqual(SCHEMA['COMPARE_PLOT_KIND'].stages, ())          # a plot setting never makes a result stale


class ViewTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state = meshed_state(self.root)

    def test_the_default_pair_is_the_voxels_and_the_mesh(self):
        before, after, view = V.pair_snapshots(self.state)
        self.assertEqual((before.title, after.title), ('Voxel structure', 'Tetrahedral mesh'))
        self.assertEqual(V.pair_phrase(before, after), 'Before and after tetrahedral meshing')
        self.assertIs(V.get_snapshot(self.state, VOXELS), before)             # measured once

    def test_a_setting_that_changes_the_numbers_remeasures(self):
        first = V.get_snapshot(self.state, VOXELS)
        self.state.config['COMPARE_INCLUDE_RVE_FACES'] = True
        second = V.get_snapshot(self.state, VOXELS)
        self.assertIsNot(first, second)
        self.assertGreater(len(second.lines['length']), len(first.lines['length']))

    def test_what_cannot_be_measured_says_what_to_run(self):
        problems = V.snapshot_problems(self.state)
        self.assertIsNone(problems[VOXELS])
        self.assertIsNone(problems[TETS])
        self.assertIn('Extract + smooth interfaces', problems[SURFACE_RAW])
        self.assertIn('Tetrahedral meshing', problems[SURFACE_PREPARED])
        self.state.store_artefact('repaired_rve', SimpleNamespace(points=np.zeros((3, 3))))
        self.assertIn('was not kept in this run', V.snapshot_problems(self.state)[SURFACE_PREPARED])
        with self.assertRaises(V.SnapshotUnavailable):
            V.get_snapshot(self.state, SURFACE_SMOOTHED)

    def test_without_a_hierarchy_nothing_is_measured(self):
        self.state.artefacts.pop('load_lgi.hierarchy', None)
        self.state.artefacts.pop('subset_extract.hierarchy', None)
        for key in list(self.state.artefacts):
            if 'hierarchy' in key:
                self.state.artefacts.pop(key)
        with self.assertRaises(V.SnapshotUnavailable):
            V.get_snapshot(self.state, VOXELS)

    def test_every_parameter_is_drawn_in_every_view(self):
        for p in PARAMETERS:
            for view in p.views:
                picture = V.comparison_figure(self.state, p.key, view)
                self.assertTrue(picture.figure.axes, (p.key, view))
                self.assertTrue(picture.key.startswith('fidelity.'))

    def test_the_figures_name_what_they_compare(self):
        picture = V.comparison_figure(self.state, 'diameter', 'distribution')
        figure = picture.figure
        self.assertEqual(figure.get_suptitle(), 'Sphere-equivalent diameter')
        self.assertEqual(figure.axes[0].get_title(),
                         'Before and after tetrahedral meshing: before = Voxel structure, after = Tetrahedral mesh')
        labels = [t.get_text().split('\n')[0] for t in figure.axes[0].get_legend().get_texts()]
        self.assertEqual(labels, ['Voxel structure: 5 grains', 'Tetrahedral mesh: 5 grains'])
        matched = V.comparison_figure(self.state, 'diameter', 'matched').figure
        self.assertIn('Voxel structure', matched.axes[0].get_xlabel())
        self.assertIn('Tetrahedral mesh', matched.axes[0].get_ylabel())

    def test_grains_lost_before_the_surface_are_explained(self):
        state = merged_state(self.root, 'lost')
        footnote = V.comparison_figure(state, 'grain_volume', 'distribution').figure.get_supxlabel()
        self.assertIn('1 grain of the voxel structure is absent from the tetrahedral mesh', footnote)
        self.assertIn('cleaning stages removed it', footnote)
        legend = [t.get_text().split('\n')[0] for t in V.comparison_figure(
            state, 'grain_volume', 'distribution').figure.axes[0].get_legend().get_texts()]
        self.assertEqual(legend, ['Voxel structure: 5 grains', 'Tetrahedral mesh: 4 grains'])

    def test_units(self):
        auto = V.comparison_figure(self.state, 'diameter', 'distribution').figure.axes[0].get_xlabel()
        self.assertEqual(auto, r'Sphere-equivalent diameter ($\mu$m)')
        for units, text in (('mm', '(mm)'), ('m', '(m)'), ('microns', r'($\mu$m)')):
            self.state.config['COMPARE_PLOT_UNITS'] = units
            label = V.comparison_figure(self.state, 'diameter', 'distribution').figure.axes[0].get_xlabel()
            self.assertTrue(label.endswith(text), (units, label))
        self.state.config['COMPARE_PLOT_UNITS'] = 'microns'
        self.assertIn(r'$\mu$m$^3$', V.comparison_figure(self.state, 'grain_volume', 'distribution').figure.axes[0].get_xlabel())
        self.assertTrue(V.comparison_figure(self.state, 'aspect', 'distribution').figure.axes[0].get_xlabel()
                        .endswith('Grain aspect ratio'))
        self.assertTrue(V.comparison_figure(self.state, 'misorientation_boundary', 'distribution').figure.axes[0]
                        .get_xlabel().endswith('(deg)'))

    def test_the_values_follow_the_units(self):
        self.state.config['COMPARE_PLOT_UNITS'] = 'microns'
        in_microns = V.comparison_figure(self.state, 'grain_volume', 'matched').figure.axes[0].collections[0].get_offsets()
        self.state.config['COMPARE_PLOT_UNITS'] = 'mm'
        in_mm = V.comparison_figure(self.state, 'grain_volume', 'matched').figure.axes[0].collections[0].get_offsets()
        np.testing.assert_allclose(np.asarray(in_microns), np.asarray(in_mm) * 1e9)

    def test_the_plot_style_settings_reach_the_figure(self):
        self.state.config.update(COMPARE_PLOT_KIND='histogram + kde', COMPARE_COLOR_BEFORE='#102030',
                                 COMPARE_COLOR_AFTER='#a0b0c0', COMPARE_SHOW_MEDIAN=True, COMPARE_BIN_WIDTH=0.2)
        picture = V.comparison_figure(self.state, 'diameter', 'distribution')
        ax = picture.figure.axes[0]
        edges = {matplotlib.colors.to_hex(p.get_edgecolor()) for p in ax.patches}
        self.assertTrue({'#102030', '#a0b0c0'} <= edges)
        self.assertTrue(all('median' in t.get_text() for t in ax.get_legend().get_texts()))
        self.assertTrue(any('KDE bandwidth' in n for n in picture.figure.compare_notes))
        self.assertFalse(any('bin width' in n for n in picture.figure.compare_notes))     # asked for, not automatic

    def test_the_misorientation_figure_follows_the_normalisation(self):
        texts = {}
        for norm in ('fraction of the boundary area', 'area per unit volume', 'area'):
            self.state.config['COMPARE_MISORIENTATION_NORM'] = norm
            texts[norm] = V.comparison_figure(self.state, 'misorientation_boundary', 'distribution').figure.axes[0].get_ylabel()
        self.assertEqual(texts['fraction of the boundary area'], 'fraction of the boundary area per degree')
        self.assertIn('per unit volume per degree', texts['area per unit volume'])
        self.assertTrue(texts['area'].startswith('boundary area per degree'))
        self.state.config['COMPARE_GRAIN_WEIGHT'] = 'grain volume'
        grain = V.comparison_figure(self.state, 'misorientation_grain', 'distribution').figure.axes[0]
        self.assertEqual(grain.get_ylabel(), 'volume fraction of the grains per degree')
        self.state.config['COMPARE_GRAIN_WEIGHT'] = 'number of grains'
        self.assertEqual(V.comparison_figure(self.state, 'misorientation_grain', 'distribution').figure.axes[0].get_ylabel(),
                         'number of grains')

    def test_a_view_the_parameter_does_not_have_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'no matched view'):
            V.comparison_figure(self.state, 'misorientation_boundary', 'matched')

    def test_the_surfaces_before_and_after_smoothing(self):
        surfaces_for(self.state)
        self.state.config.update(COMPARE_BEFORE=TITLES[SURFACE_RAW], COMPARE_AFTER=TITLES[SURFACE_SMOOTHED])
        before, after, _ = V.pair_snapshots(self.state)
        self.assertEqual(V.pair_phrase(before, after), 'Before and after smoothing the surface')
        self.assertAlmostEqual(before.pair_area.sum() / after.pair_area.sum(), 1.5 ** 2)
        picture = V.comparison_figure(self.state, 'grain_area', 'distribution')
        legend = [t.get_text().split('\n')[0] for t in picture.figure.axes[0].get_legend().get_texts()]
        self.assertEqual(legend, ['Surface before smoothing: 5 grains', 'Surface after smoothing: 5 grains'])
        with self.assertRaisesRegex(ValueError, 'no grain volume'):
            V.comparison_figure(self.state, 'diameter', 'distribution')
        self.state.config['COMPARE_PLOT_UNITS'] = 'mm'
        tables = {t[0]: t for t in V.comparison_tables(self.state)}
        self.assertIn('Not available for this pair', tables)
        self.assertEqual(tables['Grain boundary area per grain (mm^2)'][1], ['Statistic', 'Surface before smoothing', 'Surface after smoothing'])

    def test_the_closed_surface_before_tet_meshing(self):
        surfaces_for(self.state)
        self.state.config.update(COMPARE_BEFORE=TITLES[SURFACE_PREPARED], COMPARE_AFTER=TITLES[TETS])
        before, after, _ = V.pair_snapshots(self.state)
        self.assertEqual(V.pair_phrase(before, after), 'Before and after tetrahedral meshing, on the same surface')
        np.testing.assert_allclose(before.volume, after.volume)
        self.assertTrue(V.comparison_figure(self.state, 'grain_volume', 'matched').figure.axes)
        squeezed = self.state.get_artefact(PRE_TET_POINTS) * np.array([0.5, 1.0, 1.0])
        self.state.store_artefact(PRE_TET_POINTS, squeezed)
        np.testing.assert_allclose(V.get_snapshot(self.state, SURFACE_PREPARED).volume, after.volume * 0.5)

    def test_the_tables_name_both_sides_and_use_the_units(self):
        self.state.config['COMPARE_PLOT_UNITS'] = 'mm'
        tables = {t[0]: t for t in V.comparison_tables(self.state)}
        facts = dict(map(tuple, tables['Comparison'][2]))
        self.assertEqual((facts['Before'], facts['After']), ('Voxel structure', 'Tetrahedral mesh'))
        self.assertEqual(facts['Grains in the voxel structure'], 5)
        for title in ('Sphere-equivalent diameter (mm)', 'Grain volume (mm^3)', 'Grain aspect ratio',
                      'Junction line length (per line) (mm)', 'Grain boundary misorientation: area-weighted (deg)'):
            self.assertIn(title, tables)
        self.assertEqual(tables['Sphere-equivalent diameter (mm)'][1], ['Statistic', 'Voxel structure', 'Tetrahedral mesh'])
        text = V.comparison_text(self.state)
        self.assertIn('Voxel structure', text)
        self.assertIn('Tetrahedral mesh', text)
        self.assertNotIn('before', text.split('Comparison')[1].split('Item')[0])

    def test_without_a_pair_there_are_no_tables(self):
        self.state.artefacts.pop('verified_tets')
        self.assertEqual(V.comparison_tables(self.state), [])


class PreTetSurfaceTests(unittest.TestCase):
    def test_the_surface_is_kept_before_the_tetrahedra_are_made(self):
        with work_dir() as root, mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(root / 'data')}):
            state = meshed_state(root, 'pretet')
            surfaces_for(state)
            tets = state.get_artefact('verified_tets')
            surface = state.get_artefact('repaired_rve')
            before = np.array(surface.points, copy=True)
            state.artefacts.pop(PRE_TET_POINTS)
            with mock.patch('upxo.meshing.gbconformant.d3v2p1.gmsh_tets.mesh_repaired_rve_gmsh', return_value=tets), \
                    mock.patch('upxo.meshing.gbconformant.d3v2p1.gmsh_tets.save_grain_tetrahedra'), \
                    mock.patch('upxo.meshing.gbconformant.pipeline.stage_volume.tet_quality_snapshot', return_value={}):
                handle_tet_mesh(state)
            kept = state.get_artefact(PRE_TET_POINTS)
            np.testing.assert_array_equal(kept, before)
            surface.points *= 2.0                                  # later stages move the surface; the kept copy stays
            np.testing.assert_array_equal(state.get_artefact(PRE_TET_POINTS), before)


if __name__ == '__main__':
    unittest.main()
