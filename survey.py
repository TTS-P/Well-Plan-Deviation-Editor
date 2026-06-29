"""Minimum-curvature survey math.

The :func:`minimum_curvature` function is the single source of truth for turning
a station list of ``(MD, Inclination, Azimuth)`` into 3D positions
``(TVD, Northing, Easting)`` plus dogleg severity (DLS).  Every other module
(validation, editing, export) funnels back through this function so positions
are always computed one consistent way.

Conventions
-----------
* Depths / coordinates: feet.
* Inclination: degrees from vertical (0 = straight down).
* Azimuth: degrees clockwise from grid north, 0-360.
* DLS: degrees per 100 ft.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Survey", "minimum_curvature"]


def minimum_curvature(
    md: np.ndarray,
    inc: np.ndarray,
    azi: np.ndarray,
    surface: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TVD / Northing / Easting / DLS via the minimum-curvature method.

    Parameters
    ----------
    md, inc, azi : array-like
        Measured depth (ft), inclination (deg), azimuth (deg).  Must be sorted
        by increasing MD and at least length 1.
    surface : (north, east, tvd)
        Surface reference position (ft).  Defaults to the origin.

    Returns
    -------
    tvd, north, east, dls : np.ndarray
        Arrays the same length as ``md``.  ``dls`` is degrees per 100 ft and is
        defined per *station* (dls[0] = 0; dls[i] is the severity of the
        interval ending at station i).
    """
    md = np.asarray(md, dtype=float)
    inc = np.asarray(inc, dtype=float)
    azi = np.asarray(azi, dtype=float)

    if md.ndim != 1:
        raise ValueError("md/inc/azi must be 1-D arrays")
    if not (md.shape == inc.shape == azi.shape):
        raise ValueError("md, inc and azi must all be the same length")
    n = md.size
    if n == 0:
        raise ValueError("survey must contain at least one station")

    n0, e0, tvd0 = surface

    inc_r = np.radians(inc)
    azi_r = np.radians(azi)

    tvd = np.empty(n)
    north = np.empty(n)
    east = np.empty(n)
    dls = np.zeros(n)

    tvd[0] = tvd0
    north[0] = n0
    east[0] = e0

    if n == 1:
        return tvd, north, east, dls

    i1, i2 = inc_r[:-1], inc_r[1:]
    a1, a2 = azi_r[:-1], azi_r[1:]
    dmd = np.diff(md)

    # Dogleg angle (radians) between consecutive stations.
    cos_beta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # Ratio factor RF = (2/beta) * tan(beta/2); -> 1 as beta -> 0.
    rf = np.ones_like(beta)
    nonzero = beta > 1e-9
    rf[nonzero] = (2.0 / beta[nonzero]) * np.tan(beta[nonzero] / 2.0)

    half = dmd / 2.0
    d_tvd = half * (np.cos(i1) + np.cos(i2)) * rf
    d_north = half * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    d_east = half * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf

    tvd[1:] = tvd0 + np.cumsum(d_tvd)
    north[1:] = n0 + np.cumsum(d_north)
    east[1:] = e0 + np.cumsum(d_east)

    # Dogleg severity per 100 ft (guard zero-length intervals).
    safe_dmd = np.where(dmd > 0, dmd, np.nan)
    dls[1:] = np.degrees(beta) * 100.0 / safe_dmd
    dls = np.nan_to_num(dls, nan=0.0)

    return tvd, north, east, dls


@dataclass
class Survey:
    """A directional survey with positions computed from MD/Inc/Azi.

    ``tvd``, ``north``, ``east`` and ``dls`` are derived on construction (and
    after :meth:`recompute`) from ``md``/``inc``/``azi`` and ``surface`` so they
    always stay consistent with the minimum-curvature math.
    """

    md: np.ndarray
    inc: np.ndarray
    azi: np.ndarray
    surface: tuple[float, float, float] = (0.0, 0.0, 0.0)

    tvd: np.ndarray = field(init=False)
    north: np.ndarray = field(init=False)
    east: np.ndarray = field(init=False)
    dls: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.md = np.asarray(self.md, dtype=float)
        self.inc = np.asarray(self.inc, dtype=float)
        self.azi = np.asarray(self.azi, dtype=float)
        self.recompute()

    def recompute(self) -> "Survey":
        """Recompute TVD/N/E/DLS from the current MD/Inc/Azi."""
        self.tvd, self.north, self.east, self.dls = minimum_curvature(
            self.md, self.inc, self.azi, self.surface
        )
        return self

    # -- convenience -------------------------------------------------------

    @property
    def n_stations(self) -> int:
        return int(self.md.size)

    @property
    def total_depth(self) -> float:
        """Total measured depth (MD of the last station)."""
        return float(self.md[-1])

    @property
    def max_dls(self) -> float:
        return float(np.max(self.dls)) if self.dls.size else 0.0

    @property
    def vertical_section(self) -> np.ndarray:
        """Horizontal step-out distance from surface for each station (ft)."""
        dn = self.north - self.surface[0]
        de = self.east - self.surface[1]
        return np.hypot(dn, de)

    @property
    def step_out(self) -> float:
        """Total horizontal departure of the last station from surface (ft)."""
        return float(self.vertical_section[-1])

    def to_array(self) -> np.ndarray:
        """Return an (n, 7) array of MD, Inc, Azi, TVD, N, E, DLS."""
        return np.column_stack(
            [self.md, self.inc, self.azi, self.tvd, self.north, self.east, self.dls]
        )

    def copy(self) -> "Survey":
        return Survey(self.md.copy(), self.inc.copy(), self.azi.copy(), self.surface)
