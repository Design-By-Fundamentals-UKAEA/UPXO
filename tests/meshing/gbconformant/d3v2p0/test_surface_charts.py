import unittest
from unittest.mock import patch
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.surface_charts import disk_charts
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import _directed_boundary,remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces


class SurfaceChartTests(unittest.TestCase):
    def test_small_opening_triggers_geometric_restoration(self):
        from upxo.meshing.gbconformant.d3v2p0.facet_angles import small_facet_angles
        labels=np.ones((3,3,3),int);labels[1:]=2
        calls=[]
        def one_fold(points,faces,minimum):
            calls.append(minimum)
            if len(calls)==1:return np.array([[0,1]]),np.array([.01])
            return small_facet_angles(points,faces,minimum)
        with patch('upxo.meshing.gbconformant.d3v2p0.facet_angles.small_facet_angles',side_effect=one_fold):
            result=remesh_interfaces_gmsh(smooth_interfaces(labels),.8,check_intersections=True,minimum_facet_angle=2.3)
        self.assertTrue(all(v==2.3 for v in calls))
        self.assertEqual(result.report['geometric_repairs'][0]['small_facet_angles'],1)
        self.assertEqual(result.report['remaining_small_facet_angles'],0)

    def test_geometric_retry_has_fresh_parametrization_budget(self):
        import gmsh
        from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections
        labels=np.ones((3,3,3),int);labels[1:]=2
        source=smooth_interfaces(labels,iterations=1)
        checks=[];generations=[];generate=gmsh.model.mesh.generate
        def one_hit(points,faces):
            checks.append(True)
            return np.array([[0,1]]) if len(checks)==1 else find_surface_intersections(points,faces)
        def warn_on_restoration(dim):
            generate(dim);generations.append(True)
            if len(generations)==2:
                tag=gmsh.model.getEntities(2)[0][1]
                gmsh.logger.write(f'2 elements remain invalid in surface {tag}','warning')
        with patch('upxo.meshing.gbconformant.d3v2p0.surface_intersections.find_surface_intersections',side_effect=one_hit), patch.object(gmsh.model.mesh,'generate',side_effect=warn_on_restoration):
            result=remesh_interfaces_gmsh(source,.8,check_intersections=True,_retry_depth=3)
        self.assertEqual(result.report['remaining_intersections'],0)
        self.assertGreaterEqual(len(generations),3)

    def test_explicit_topology_keeps_endpoints_when_embedding_frozen_node(self):
        labels=np.ones((4,4,4),int);labels[2:]=2
        source=smooth_interfaces(labels,iterations=0)
        interior=np.flatnonzero(np.all(source.points==[2,2,2],axis=1))[0]
        source.node_kind[interior]=3
        result=remesh_interfaces_gmsh(source,.8,_retry_depth=1)
        self.assertTrue(result.report['shared_boundaries_verified'])
        self.assertEqual(result.report['frozen_junction_points_preserved'],1)

    def test_intersection_retry_preserves_labels_and_boundaries(self):
        from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections
        labels=np.ones((4,4,4),int);labels[2:]=2
        source=smooth_interfaces(labels,iterations=2)
        called=[]
        def one_hit(points,faces):
            if not called:
                called.append(True);return np.array([[0,1]],int)
            return find_surface_intersections(points,faces)
        with patch('upxo.meshing.gbconformant.d3v2p0.surface_intersections.find_surface_intersections',side_effect=one_hit):
            result=remesh_interfaces_gmsh(source,.8,check_intersections=True,rve_dimensions=[4,4,4])
        self.assertEqual(result.grain_pairs.shape[1],2)
        self.assertEqual(result.report['remaining_intersections'],0)
        self.assertTrue(result.report['shared_boundaries_verified'])
        self.assertEqual(result.report['geometric_repairs'][0]['detected_intersections'],1)

    def test_invalid_element_warning_triggers_targeted_retry(self):
        import gmsh
        labels=np.ones((4,4,4),int);labels[2:]=2
        source=smooth_interfaces(labels,iterations=2)
        original=gmsh.model.mesh.generate;warned=[]
        def warn_once(dim):
            original(dim)
            if not warned:
                warned.append(True)
                tag=gmsh.model.getEntities(2)[0][1]
                gmsh.logger.write(f'2 elements remain invalid in surface {tag}','warning')
        with patch.object(gmsh.model.mesh,'generate',side_effect=warn_once):
            result=remesh_interfaces_gmsh(source,.8)
        self.assertEqual(result.report['surface_meshing_warnings'],[])
        self.assertIn('invalid elements',result.report['parametrization_repairs'][0]['error'])

    def test_small_closed_grain_is_remeshed_without_losing_its_label(self):
        labels=np.ones((5,5,5),int);labels[2,2,2]=2
        source=smooth_interfaces(labels,iterations=2)
        result=remesh_interfaces_gmsh(source,.6)
        self.assertTrue(np.all(result.grain_pairs==[1,2]))
        self.assertTrue(result.report['shared_boundaries_verified'])
        _,counts=np.unique(np.sort(result.triangles[:,[[0,1],[1,2],[2,0]]]
                                   .reshape(-1,2),axis=1),axis=0,return_counts=True)
        self.assertTrue(np.all(counts==2))

    def test_explicit_shared_chart_topology(self):
        labels=np.ones((4,4,4),int);labels[2:]=2;labels[:2,2:]=3
        source=smooth_interfaces(labels,iterations=2)
        groups=disk_charts(source.triangles,max_triangles=4)
        partitions=np.zeros(len(source.triangles),int)
        for i,ids in enumerate(groups):partitions[ids]=i
        result=remesh_interfaces_gmsh(source,.8,
            _patch_keys=np.column_stack((np.sort(source.grain_pairs,axis=1),partitions)),
            _retry_depth=2)
        self.assertTrue(result.report['shared_boundaries_verified'])
        self.assertEqual(result.report['maximum_curve_coordinate_error'],0.)

    def test_closed_shell_is_cut_into_disks_without_losing_faces(self):
        f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        charts=disk_charts(f)
        self.assertGreater(len(charts),1)
        np.testing.assert_array_equal(np.sort(np.concatenate(charts)),np.arange(4))
        for ids in charts:
            boundary=_directed_boundary(f[ids])
            self.assertTrue(boundary)
            _,counts=np.unique(np.array(list(boundary)).ravel(),return_counts=True)
            self.assertTrue(np.all(counts==2))

    def test_parametrization_failure_retries_using_disk_charts(self):
        import gmsh
        labels=np.ones((4,4,4),int);labels[2:]=2
        source=smooth_interfaces(labels,iterations=2)
        original=gmsh.model.mesh.createGeometry
        failed=[]
        def fail_once(dimTags=[]):
            if any(dim==2 for dim,_ in dimTags) and not failed:
                failed.append(True)
                raise Exception('Wrong topology of boundary mesh for parametrization')
            return original(dimTags)
        with patch.object(gmsh.model.mesh,'createGeometry',side_effect=fail_once):
            result=remesh_interfaces_gmsh(source,.8)
        self.assertEqual(result.grain_pairs.shape[1],2)
        self.assertTrue(np.all(result.grain_pairs==[1,2]))
        self.assertEqual(result.report['parametrization_repairs'][0]['failed_charts'],1)
        self.assertTrue(result.report['shared_boundaries_verified'])

    def test_generation_failure_retries_without_leaking_chart_labels(self):
        import gmsh
        from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
        from upxo.meshing.gbconformant.d3v2p0.gmsh_closed import remesh_closed_rve_gmsh
        labels=np.ones((4,4,4),int);labels[2:]=2
        internal=remesh_interfaces_gmsh(smooth_interfaces(labels,iterations=2),.8)
        closed=close_rve_faces(internal,labels,mesh_size=.8)
        original=gmsh.model.mesh.generate;failed=[]
        def fail_once(dim):
            if not failed:
                failed.append(True);raise Exception('The 1D mesh seems not to be forming a closed loop')
            return original(dim)
        with patch.object(gmsh.model.mesh,'generate',side_effect=fail_once):
            result=remesh_closed_rve_gmsh(closed,.8)
        self.assertEqual(result.grain_pairs.shape[1],2)
        self.assertEqual(set(result.rve_face),{-1,0,1,2,3,4,5})
        self.assertTrue(result.report['all_six_faces_closed'])
        self.assertEqual(result.report['parametrization_repairs'][0]['stage'],'surface_generation')
