Orientation Conventions
=======================

.. contents:: On this page
   :local:
   :depth: 2

This page states which rotation each orientation format in UPXO holds and on
which side symmetry operators and crystallographic relationships are applied.
The values agree with mtex 1.0.0 for a cubic (m-3m) crystal.

----

Formats
-------

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Format
     - Maps
     - Where it appears
   * - Bunge Euler angles (phi1, Phi, phi2), degrees
     - (defines R)
     - EBSD files, texture components, ``ipf_color``, ``PoleFigure(convention='euler_deg')``
   * - Matrix **R** = Rz(phi1) Rx(Phi) Rz(phi2)
     - crystal to sample
     - ``euler_bunge_to_matrix``, ``PoleFigure(convention='matrix')``
   * - Matrix **g** = R transposed
     - sample to crystal
     - The Bunge matrix of textbooks; the FM steel orientation assigner
   * - Stored quaternion [w, x, y, z] = quaternion of **g**
     - sample to crystal
     - ``EBSDReader.quat_ebsd``, ``grain_avg_quats``, orientation pools,
       ``tops.synth_fcc_quats``, the quaternions of the twinned pipeline
   * - Active quaternion = quaternion of **R**
     - crystal to sample
     - ``PoleFigure(convention='quaternion')``; ``defdap_passive_to_active``
       converts a stored quaternion (it negates the vector part)

A stored quaternion read as crystal to sample places every pole at its
transposed position. ``build_ipf_rgb`` takes stored quaternions by default
(``quat_convention='defdap'``) and active ones with
``quat_convention='crystal_to_sample'``.

----

Symmetry operators and disorientation
-------------------------------------

The 24 proper cubic operators ``S`` act on the crystal side of an orientation:
``R @ S`` for R, ``S @ g`` for g and for a stored quaternion. Two matrices
describe the same orientation when they differ by such a product.

The disorientation of two orientations is the smallest rotation angle of
``Ra.T @ Rb @ S`` over the 24 operators (``cubic_misorientation``). For g
matrices the crystal-frame relative rotation is ``g1 @ g2.T``.

Applying the operators on the sample side gives wrong angles for large
misorientations. The Mackenzie distribution of a random pool cannot reveal
this, because the relative rotation of two uniformly random orientations is
uniform whichever side is used; comparison of individual pairs can.

----

Crystallographic relationships
------------------------------

A fixed crystal-frame relationship ``T`` (a Sigma3 twin, a Kurdjumov-Sachs
variant) is composed on the same side as a symmetry operator:

.. code-block:: text

   R_child = R_parent @ T          g_child = T.T @ g_parent

Composed on the other side, the child differs from the correct orientation by tens of degrees
and the relationship angles are no longer the tabulated ones. In the twinned
pipeline the four Sigma3 variants of one parent are mutually 38.94 degrees
(Sigma9) apart; in the FM steel pipeline two blocks of one parent PAG differ by
0 degrees or one of 10.53, 14.88, 20.61, 21.06, 47.11, 49.47, 50.51, 51.73,
57.21 and 60.0 degrees.

----

Directions and planes
---------------------

* The crystal direction of a sample axis ``d`` is ``R.T @ d``.
* The sample-frame normal of a crystal plane ``n`` is ``R @ n`` (``g.T @ n``).
  The {111} habit-plane normals of a Sigma3 twin and the packet planes of the
  FM steel pipeline are computed this way.
* Inverse-pole-figure colours use mtex's default cubic key: the crystal
  direction is reduced to the sector [001], [101], [111] and coloured from
  there, so symmetric equivalents have one colour.
