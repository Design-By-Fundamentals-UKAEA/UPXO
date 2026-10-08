import importlib
import sys

import pytest

from upxo._sup import optional_imports


def test_import_gmsh_returns_the_module():
    pytest.importorskip('gmsh')
    assert optional_imports.import_gmsh().__name__ == 'gmsh'


def test_missing_system_library_gives_an_actionable_message(monkeypatch):
    def fail(name, *args, **kwargs):
        raise OSError('libGLU.so.1: cannot open shared object file: No such file or directory')
    monkeypatch.setattr(importlib, 'import_module', fail)
    with pytest.raises(ImportError) as info:
        optional_imports.import_gmsh()
    text = str(info.value)
    assert 'libGLU.so.1' in text and 'apt-get install' in text and 'libglu1-mesa' in text
    assert isinstance(info.value.__cause__, OSError)


def test_missing_package_still_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gmsh', None)         # makes `import gmsh` raise ModuleNotFoundError
    with pytest.raises(ImportError):
        optional_imports.import_gmsh()
