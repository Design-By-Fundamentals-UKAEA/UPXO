"""The long-running loops report progress, and reporting changes nothing."""
import numpy as np
import pytest

from upxo._sup import progress


def _messages(sink):
    return [e.message for e in sink.events]


@pytest.mark.parametrize("alg", ["300a", "300b"])
def test_mc_simulation_reports_every_time_step(alg):
    from upxo.ggrowth.mcgs_v2 import mcgs_v2, MCGSConfig
    config = MCGSConfig(xmin=0, xmax=7, xinc=1, ymin=0, ymax=7, yinc=1,
                        zmin=0, zmax=7, zinc=1, Q=4, mcalg=alg, mcsteps=4,
                        save_interval=2, consider_boltzmann=False)
    pxt = mcgs_v2(config, verbose=False)
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        pxt.simulate(verbose=False)
    steps = [e for e in sink.events if e.message == "Monte-Carlo step"]
    assert [e.current for e in steps] == [0, 1, 2, 3]
    assert all(e.total == 4 for e in steps)


def _two_grain_block():
    lgi = np.ones((6, 6, 6), dtype=np.int32)
    lgi[:, :, 3:] = 2
    lgi[2, 2, 2] = 2            # a single-voxel spike inside grain 1
    quats = {1: np.array([1.0, 0, 0, 0]), 2: np.array([0.0, 1.0, 0, 0])}
    role = {1: 'host', 2: 'host'}
    parent = {1: 1, 2: 2}
    return lgi, quats, role, parent


def test_cleaning_reports_each_pass_and_result_is_unchanged():
    from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D
    args = _two_grain_block()

    plain, n_plain = StructureCleaner3D.clean_recursive(
        *[a.copy() if hasattr(a, "copy") else a for a in args], n_passes=3, verbose=False)

    sink = progress.ListSink()
    with progress.progress_scope(sink):
        listened, n_listened = StructureCleaner3D.clean_recursive(
            *[a.copy() if hasattr(a, "copy") else a for a in args], n_passes=3, verbose=False)

    assert n_plain == n_listened
    assert np.array_equal(plain.lgi_clean, listened.lgi_clean)
    passes = [e for e in sink.events if e.message == "Cleaning pass"]
    assert passes and passes[0].current == 0 and passes[0].total == 3


def test_validator_reports_one_event_per_slice_across_axes():
    from upxo.pxtal.twinned_simple_3d.repr_validator_3d import RepresentativenessValidator3D
    lgi = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2).repeat(3, 0).repeat(3, 1).repeat(3, 2)
    validator = RepresentativenessValidator3D(
        n_slices_x=3, n_slices_y=2, n_slices_z=1, test_along_z=True)
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        validator.validate_morphological(lgi, np.array([1.0, 2.0, 3.0]))
    assert [e.current for e in sink.events] == [0, 1, 2, 3, 4, 5]
    assert all(e.total == 6 for e in sink.events)
    assert {e.message for e in sink.events} == {
        "Validating X slices", "Validating Y slices", "Validating Z slices"}


def test_no_listener_leaves_instrumented_code_silent():
    from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D
    assert not progress.is_active()
    cleaner, n = StructureCleaner3D.clean_recursive(*_two_grain_block(), n_passes=2, verbose=False)
    assert n >= 1
