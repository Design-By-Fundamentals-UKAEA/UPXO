import upxo


def test_version_is_string():
    assert isinstance(upxo.__version__, str) and len(upxo.__version__) > 0
