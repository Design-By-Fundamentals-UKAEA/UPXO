import matplotlib
import pytest
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure

from upxo.viz import plot_quality as pq


@pytest.fixture(autouse=True)
def _restore_rcparams_and_state():
    saved_state = pq._current
    with matplotlib.rc_context():
        yield
    pq._current = saved_state


def test_all_presets_define_the_same_keys():
    keys = [set(p) for p in pq.PRESETS.values()]
    assert keys[0] == keys[1] == keys[2]
    assert pq.preset_names() == ["quick", "high", "publication"]


def test_every_value_is_accepted_by_matplotlib():
    for name in pq.preset_names():
        with matplotlib.rc_context(pq.get_preset(name)):
            pass


def test_quality_orders_dpi_and_fonts():
    quick, high, pub = (pq.PRESETS[n] for n in pq.preset_names())
    assert quick["savefig.dpi"] < high["savefig.dpi"] < pub["savefig.dpi"] == 300
    assert quick["figure.dpi"] < high["figure.dpi"] < pub["figure.dpi"]
    assert quick["font.size"] < high["font.size"] < pub["font.size"]
    assert quick["lines.linewidth"] < high["lines.linewidth"] < pub["lines.linewidth"]


def test_publication_uses_serif_stix_and_editable_text():
    pub = pq.PRESETS["publication"]
    assert pub["font.family"][0] == "STIXGeneral"
    assert pub["mathtext.fontset"] == "stix"
    assert pub["svg.fonttype"] == "none" and pub["pdf.fonttype"] == 42
    assert pub["xtick.direction"] == "in"
    assert pq.PRESETS["quick"]["svg.fonttype"] == "path"


def test_apply_preset_changes_rcparams_and_new_figures_pick_it_up():
    pq.apply_preset("publication")
    assert pq.current_preset() == "publication"
    assert matplotlib.rcParams["savefig.dpi"] == 300
    ax = Figure().add_subplot(111)
    ax.set_xlabel("x")
    assert ax.xaxis.label.get_fontsize() == pytest.approx(11)
    pq.apply_preset("quick")
    ax = Figure().add_subplot(111)
    ax.set_xlabel("x")
    assert ax.xaxis.label.get_fontsize() == pytest.approx(9)


def test_switching_presets_fully_replaces_previous_one():
    pq.apply_preset("publication")
    pq.apply_preset("quick")
    for key, value in pq.PRESETS["quick"].items():
        if key == "axes.prop_cycle":
            assert list(matplotlib.rcParams[key].by_key()["color"]) == pq.TAB10
        else:
            assert matplotlib.rcParams[key] == value


def test_quality_context_restores_rcparams_and_current_name():
    before = matplotlib.rcParams["font.size"]
    pq.apply_preset("high")
    with pq.quality_context("publication"):
        assert matplotlib.rcParams["font.size"] == 11
        assert pq.current_preset() == "publication"
    assert matplotlib.rcParams["font.size"] == pq.PRESETS["high"]["font.size"]
    assert pq.current_preset() == "high"
    assert before is not None


def test_quality_context_restores_on_exception():
    pq.apply_preset("high")
    with pytest.raises(RuntimeError):
        with pq.quality_context("publication"):
            raise RuntimeError
    assert pq.current_preset() == "high"
    assert matplotlib.rcParams["font.size"] == pq.PRESETS["high"]["font.size"]


def test_names_are_case_insensitive_and_unknown_names_are_rejected():
    assert pq.get_preset("  Publication ") == pq.get_preset("publication")
    with pytest.raises(KeyError):
        pq.apply_preset("ultra")
    with pytest.raises(KeyError):
        pq.get_preset("")


def test_get_preset_returns_a_copy():
    d = pq.get_preset("high")
    d["font.size"] = 99
    assert pq.PRESETS["high"]["font.size"] == 10.5


def test_savefig_dpi_uses_named_then_current_then_default():
    assert pq.savefig_dpi("quick") == 120
    pq._current = None
    assert pq.savefig_dpi() == pq.PRESETS[pq.DEFAULT_PRESET]["savefig.dpi"]
    pq.apply_preset("publication")
    assert pq.savefig_dpi() == 300


def test_palettes_are_valid_distinct_colours():
    assert all(is_color_like(c) for c in pq.OKABE_ITO + pq.TAB10)
    assert len(set(pq.OKABE_ITO)) == len(pq.OKABE_ITO) == 8
    assert pq.EBSD_COLOR != pq.SGS_COLOR
    assert pq.EBSD_COLOR in pq.OKABE_ITO and pq.SGS_COLOR in pq.OKABE_ITO


def test_describe_has_text_for_each_preset():
    assert all(pq.describe(n) for n in pq.preset_names())


@pytest.mark.parametrize("name", ["quick", "high", "publication"])
@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_figure_saves_in_every_format_under_every_preset(name, fmt):
    import io
    with pq.quality_context(name):
        fig = Figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 4], label=r"$x^2$")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=pq.savefig_dpi())
    assert buf.getbuffer().nbytes > 500
