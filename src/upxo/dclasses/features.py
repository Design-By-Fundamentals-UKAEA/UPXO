# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:06:19 2024

@author: Dr. Sunil Anandatheertha
"""
from dataclasses import dataclass

@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class twingen:
    """
    Twin-generation specification (volume fraction and thickness stats).

    Dataclass of target twin volume fraction and thickness distribution
    controls used by 2D/3D twinning workflows. Thickness may follow a
    named distribution (``tdis``) or an explicit user list (``tvalues``).

    Example
    -------
    >>> from upxo.dclasses.features import twingen
    >>> tg = twingen(vf=0.2, tmin=0.2, tmean=0.5, tmax=1.0,
    ...              tdis='user', tvalues=[0.2, 0.5, 1.0, 0.75],
    ...              allow_partial=True, partial_prob=0.2)

    Attributes
    ----------
    vf : float
        Target twin volume fraction.
    tmin, tmean, tmax : float
        Thickness bounds / mean for the twin lamella distribution.
    tdis : str
        Distribution name, or ``'user'`` to use ``tvalues``.
    tvalues : list
        Explicit thickness samples when ``tdis='user'``.
    allow_partial : bool
        Whether partial twins (truncated at grain bounds) are allowed.
    partial_prob : float
        Probability of partial-twin placement when allowed.
    """
    vf: float
    tmin: float
    tmean: float
    tmax: float
    tdis: str
    tvalues: list
    allow_partial: bool
    partial_prob: float


"""
* twin_vf: Twin volume fraction
* partial_twins: Allow partial twins or not
* pga_bounds: parent grain area bounds
* pgar_bounds: parent grain aspect ratio bounds
* min_twins_pg: Min. no. of twins per grain
* max_twins_pg: Max. no. of twins per grain

NOTE:
* pg: per grain
"""



vf=0.2
tspec='absolute'
trel='minil'
tdis='user'
t=[0.2, 0.5, 0.6, 0.7]
tw=[1, 1, 1, 1]
tmin=0.2
tmean=0.5
tmax=1.0
nmax_pg=1
placement='centroid'
factor_min=0.0
factor_max=1.0
