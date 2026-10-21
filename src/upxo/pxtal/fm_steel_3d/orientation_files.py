"""Reading tables of Euler angles per item (block, sub-block, ...) from a file."""
import os
from typing import Dict, Tuple

__all__ = ['read_euler_table']


def read_euler_table(path, id_column: str = "block_id") -> Dict[str, Tuple[float, float, float]]:
    """Bunge Euler angles ``(phi1, Phi, phi2)`` in degrees, keyed by item id.

    The file is a table with the columns ``id_column``, ``phi1``, ``phi`` and
    ``phi2``. Column names are compared in lower case without surrounding
    blanks. The format follows the extension: ``.xlsx`` / ``.xls`` Excel,
    ``.txt`` / ``.tsv`` tab-separated text, anything else comma-separated.

    Raises
    ------
    ValueError
        If a required column is missing or an angle is not a number.
    """
    import pandas as pd

    ext = os.path.splitext(str(path))[-1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif ext in (".txt", ".tsv"):
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    required = [id_column.lower(), "phi1", "phi", "phi2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")

    table = {}
    for _, row in df.iterrows():
        table[str(row[required[0]]).strip()] = (
            float(row["phi1"]), float(row["phi"]), float(row["phi2"]))
    return table
