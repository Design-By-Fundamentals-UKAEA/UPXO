# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:59:37 2025

@author: rg5749
"""
import numpy as np

def arrange_subplots(N, Rmax, Cmax):
    """Arranges N subplots within Rmax x Cmax grid, prioritizing rows <= columns.

    Calculates the optimal number of rows and columns for a subplot grid,
    prioritizing arrangements where the number of rows is less than or equal
    to the number of columns.

    Parameters
    ----------
    N: int
        The total number of subplots required.
    Rmax: int
        The maximum permissible number of rows.
    Cmax: int
        The maximum permissible number of columns.

    Returns
    -------
    tuple (int, int) or None
        A tuple containing the number of rows and columns (nrows, ncols) for the
        subplot arrangement. Returns None if no valid arrangement is possible
        within the constraints.

    """

    if N <= 0 or Rmax <= 0 or Cmax <= 0:
        return None  # Invalid input

    best_rows = -1
    best_cols = -1

    for rows in range(1, Rmax + 1):
        cols = np.ceil(N / rows).astype(int)  # Calculate columns, round up
        if cols <= Cmax and rows <= cols: #Check rows <= cols FIRST
            if best_rows == -1 or rows * cols < best_rows * best_cols: #Then check if it is the best option so far
                best_rows = rows
                best_cols = cols

    if best_rows == -1:
        return None  # No valid arrangement found

    return best_rows, best_cols
