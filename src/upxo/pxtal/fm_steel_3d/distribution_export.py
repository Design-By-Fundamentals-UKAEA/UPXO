"""Writing property distributions to files."""
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

__all__ = ['FORMATS', 'export_distributions']

FORMATS = ('csv', 'xlsx', 'txt', 'pickle')


def _write_csv(path, properties, columns, max_len):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(properties)
        for i in range(max_len):
            writer.writerow([columns[p][i] if i < len(columns[p]) else "" for p in properties])


def export_distributions(distributions: Dict[str, Dict[str, list]], folder, formats: Iterable[str]
                         ) -> List[Path]:
    """Write ``{group: {property: [values]}}`` to ``folder`` and return the files.

    Properties without values are left out and a group with none is skipped.
    Per group:

    ``csv``     ``<group>_distributions.csv``, one column per property, shorter
                columns padded with empty cells;
    ``xlsx``    ``<group>_distributions.xlsx`` (pandas), or, without pandas,
                ``<group>_distributions_nopandas.csv`` as for ``csv``;
    ``txt``     ``<group>_<property>.txt``, one value per line;
    ``pickle``  ``<group>_distributions.pkl``, the dict of the group.

    ``folder`` is created if needed.
    """
    formats = set(formats)
    unknown = formats - set(FORMATS)
    if unknown:
        raise ValueError(f"unknown formats {sorted(unknown)}; choose from {FORMATS}")
    folder = Path(folder)
    os.makedirs(folder, exist_ok=True)
    written: List[Path] = []

    for group, columns in distributions.items():
        properties = [p for p, values in columns.items() if values]
        if not properties:
            continue
        max_len = max(len(columns[p]) for p in properties)

        if 'csv' in formats:
            path = folder / f"{group}_distributions.csv"
            _write_csv(path, properties, columns, max_len)
            written.append(path)

        if 'xlsx' in formats:
            try:
                import pandas as pd
                df = pd.DataFrame({p: columns[p] + [None] * (max_len - len(columns[p])) for p in properties})
                path = folder / f"{group}_distributions.xlsx"
                df.to_excel(path, index=False)
            except ImportError:
                path = folder / f"{group}_distributions_nopandas.csv"
                _write_csv(path, properties, columns, max_len)
            written.append(path)

        if 'txt' in formats:
            for p in properties:
                path = folder / f"{group}_{p}.txt"
                np.savetxt(path, np.array(columns[p], dtype=float))
                written.append(path)

        if 'pickle' in formats:
            import pickle
            path = folder / f"{group}_distributions.pkl"
            with open(path, "wb") as f:
                pickle.dump(columns, f)
            written.append(path)

    return written
