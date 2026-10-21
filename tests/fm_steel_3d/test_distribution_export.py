"""Writing property distributions to files."""
import builtins
import csv
import pickle

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.distribution_export import FORMATS, export_distributions


DATA = {
    'pag': {'area': [1.0, 2.0, 3.0], 'aspect_ratio': [1.5, 2.5], 'solidity': []},
    'packet': {'area': [4.0, 5.0]},
    'empty': {'area': []},
}


def _rows(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


def test_csv_has_one_column_per_property_padded_to_the_longest(tmp_path):
    written = export_distributions(DATA, tmp_path, ['csv'])
    assert sorted(p.name for p in written) == ['packet_distributions.csv', 'pag_distributions.csv']
    assert _rows(tmp_path / 'pag_distributions.csv') == [
        ['area', 'aspect_ratio'], ['1.0', '1.5'], ['2.0', '2.5'], ['3.0', '']]      # 'solidity' has no values
    assert _rows(tmp_path / 'packet_distributions.csv') == [['area'], ['4.0'], ['5.0']]


def test_text_files_hold_one_value_per_line(tmp_path):
    written = export_distributions(DATA, tmp_path, ['txt'])
    assert sorted(p.name for p in written) == ['packet_area.txt', 'pag_area.txt', 'pag_aspect_ratio.txt']
    np.testing.assert_array_equal(np.loadtxt(tmp_path / 'pag_aspect_ratio.txt'), [1.5, 2.5])


def test_pickle_holds_the_dict_of_the_group(tmp_path):
    export_distributions(DATA, tmp_path, ['pickle'])
    with open(tmp_path / 'pag_distributions.pkl', 'rb') as f:
        assert pickle.load(f) == DATA['pag']


def test_excel(tmp_path):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('openpyxl')
    written = export_distributions(DATA, tmp_path, ['xlsx'])
    assert sorted(p.name for p in written) == ['packet_distributions.xlsx', 'pag_distributions.xlsx']
    frame = pd.read_excel(tmp_path / 'pag_distributions.xlsx')
    assert list(frame.columns) == ['area', 'aspect_ratio']
    assert frame['area'].tolist() == [1.0, 2.0, 3.0]
    assert frame['aspect_ratio'].isna().tolist() == [False, False, True]


def test_excel_falls_back_to_csv_without_pandas(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == 'pandas' or name.startswith('pandas.'):
            raise ImportError('no pandas')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_pandas)
    written = export_distributions(DATA, tmp_path, ['xlsx'])
    assert sorted(p.name for p in written) == ['packet_distributions_nopandas.csv', 'pag_distributions_nopandas.csv']
    assert _rows(tmp_path / 'pag_distributions_nopandas.csv')[0] == ['area', 'aspect_ratio']


def test_several_formats_and_a_new_folder(tmp_path):
    folder = tmp_path / 'new' / 'dir'
    written = export_distributions(DATA, folder, ['csv', 'txt', 'pickle'])
    assert folder.is_dir() and all(p.exists() and p.parent == folder for p in written)
    assert len(written) == 2 + 3 + 2


def test_nothing_is_written_for_no_formats_or_no_values(tmp_path):
    assert export_distributions(DATA, tmp_path, []) == []
    assert export_distributions({'a': {'x': []}}, tmp_path, ['csv', 'txt']) == []
    assert list(tmp_path.iterdir()) == []


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError, match='unknown formats'):
        export_distributions(DATA, tmp_path, ['csv', 'json'])
    assert set(FORMATS) == {'csv', 'xlsx', 'txt', 'pickle'}
