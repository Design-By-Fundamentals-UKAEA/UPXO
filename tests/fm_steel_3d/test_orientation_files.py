"""Reading tables of Euler angles from files."""
import pytest

from upxo.pxtal.fm_steel_3d.orientation_files import read_euler_table

pd = pytest.importorskip('pandas')

EXPECTED = {'1_1_1': (10.5, 1.0, 5.0), '1_1_2': (20.0, 2.0, 6.0), '2_1_3': (30.25, 3.0, 7.0)}


def _table(**overrides):
    columns = {' Block_ID ': ['1_1_1', '1_1_2', ' 2_1_3 '], 'PHI1': [10.5, 20, 30.25],
               'Phi': [1, 2, 3], 'phi2 ': [5, 6, 7]}
    columns.update(overrides)
    return pd.DataFrame(columns)


@pytest.mark.parametrize('name, writer', [
    ('t.csv', lambda df, p: df.to_csv(p, index=False)),
    ('t.dat', lambda df, p: df.to_csv(p, index=False)),                      # anything else is comma-separated
    ('t.tsv', lambda df, p: df.to_csv(p, index=False, sep='\t')),
    ('t.txt', lambda df, p: df.to_csv(p, index=False, sep='\t')),
    ('T.CSV', lambda df, p: df.to_csv(p, index=False)),
])
def test_text_formats(tmp_path, name, writer):
    path = tmp_path / name
    writer(_table(), path)
    assert read_euler_table(path) == EXPECTED


def test_excel(tmp_path):
    pytest.importorskip('openpyxl')
    path = tmp_path / 't.xlsx'
    _table().to_excel(path, index=False)
    assert read_euler_table(path) == EXPECTED


def test_a_path_given_as_text(tmp_path):
    path = tmp_path / 't.csv'
    _table().to_csv(path, index=False)
    assert read_euler_table(str(path)) == EXPECTED


def test_another_id_column(tmp_path):
    path = tmp_path / 't.csv'
    frame = _table()
    frame.columns = ['subblock_id', 'phi1', 'phi', 'phi2']
    frame.to_csv(path, index=False)
    assert set(read_euler_table(path, id_column='subblock_id')) == set(EXPECTED)
    with pytest.raises(ValueError, match="missing column"):
        read_euler_table(path)


def test_missing_columns_are_reported_with_the_ones_found(tmp_path):
    path = tmp_path / 't.csv'
    _table().drop(columns=['phi2 ']).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"missing column\(s\) \['phi2'\].*found"):
        read_euler_table(path)


def test_an_angle_that_is_not_a_number(tmp_path):
    path = tmp_path / 't.csv'
    _table(PHI1=[1, 'x', 3]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_euler_table(path)


def test_the_last_row_wins_for_a_repeated_id(tmp_path):
    path = tmp_path / 't.csv'
    _table(**{' Block_ID ': ['a', 'a', 'b']}).to_csv(path, index=False)
    assert read_euler_table(path) == {'a': (20.0, 2.0, 6.0), 'b': (30.25, 3.0, 7.0)}
