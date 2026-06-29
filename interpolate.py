"""Spline interpolation of inclination & azimuth versus measured depth.

The chosen interpolation strategy (locked with the user) is to fit smooth
splines to ``Inc`` and ``Azi`` as functions of ``MD``, resample on a fine MD
grid, then integrate to positions with the minimum-curvature method.

Azimuth is unwrapped before fitting so a 359 -> 1 degree crossing does not cause
the spline to swing all the way around; it is re-wrapped to 0-360 afterwards.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from wellplan.survey import Survey

__all__ = ["fit_angles", "resample", "interpolate_survey", "regrid_survey", "SPLINE_KINDS"]

SPLINE_KINDS = ("cubic", "pchip")


def _unwrap_deg(azi: np.ndarray) -> np.ndarray:
    """Unwrap an azimuth series (degrees) into a continuous curve."""
    return np.degrees(np.unwrap(np.radians(np.asarray(azi, dtype=float))))


def _make_spline(x: np.ndarray, y: np.ndarray, kind: str):
    if kind == "cubic":
        # ``natural`` keeps the end curvature tame, avoiding wild DLS at TD.
        return CubicSpline(x, y, bc_type="natural")
    if kind == "pchip":
        return PchipInterpolator(x, y)
    raise ValueError(f"unknown spline kind {kind!r}; expected one of {SPLINE_KINDS}")


def fit_angles(md: np.ndarray, inc: np.ndarray, azi: np.ndarray, kind: str = "cubic"):
    """Return spline callables ``(inc_spline, azi_spline)`` over MD.

    The azimuth spline operates on the unwrapped series; callers should wrap the
    result back to 0-360 with ``np.mod(value, 360)``.
    """
    md = np.asarray(md, dtype=float)
    inc = np.asarray(inc, dtype=float)
    azi_unwrapped = _unwrap_deg(azi)

    # Splines need strictly increasing, de-duplicated x values.
    md_u, idx = np.unique(md, return_index=True)
    inc_spline = _make_spline(md_u, inc[idx], kind)
    azi_spline = _make_spline(md_u, azi_unwrapped[idx], kind)
    return inc_spline, azi_spline


def resample(
    md: np.ndarray,
    inc: np.ndarray,
    azi: np.ndarray,
    step: float = 30.0,
    kind: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample Inc/Azi onto a uniform MD grid of spacing ``step`` ft.

    The original station MDs are merged into the grid so survey points are
    preserved exactly, and the first/last MD always appear.
    """
    md = np.asarray(md, dtype=float)
    if md.size < 2:
        return md.copy(), np.asarray(inc, float).copy(), np.asarray(azi, float).copy()
    if step <= 0:
        raise ValueError("resample step must be positive")

    inc_spline, azi_spline = fit_angles(md, inc, azi, kind)

    grid = np.arange(md[0], md[-1], step)
    grid = np.union1d(np.append(grid, md[-1]), md)  # keep stations + TD
    grid = grid[(grid >= md[0]) & (grid <= md[-1])]

    inc_new = inc_spline(grid)
    azi_new = np.mod(azi_spline(grid), 360.0)
    # Inclination is physically >= 0; clip tiny negative spline undershoots.
    inc_new = np.clip(inc_new, 0.0, 180.0)
    return grid, inc_new, azi_new


def interpolate_survey(
    survey: Survey, step: float = 30.0, kind: str = "cubic"
) -> Survey:
    """Return a new densely-sampled :class:`Survey` from ``survey``."""
    md, inc, azi = resample(survey.md, survey.inc, survey.azi, step=step, kind=kind)
    return Survey(md, inc, azi, surface=survey.surface)


def regrid_survey(survey: Survey, step: float = 30.0, kind: str = "cubic") -> Survey:
    """Re-grid a survey onto *uniform* ``step``-ft MD stations.

    Unlike :func:`resample`, the original station MDs are **not** preserved -- the
    Inc/Azi splines are evaluated on a clean uniform grid (with TD kept as the
    final station).  This is used to emit an output survey on tidy, even MD
    intervals regardless of how the editing reshaped the station spacing.
    """
    md = survey.md
    if md.size < 2:
        return survey.copy()
    if step <= 0:
        raise ValueError("regrid step must be positive")

    inc_spline, azi_spline = fit_angles(md, survey.inc, survey.azi, kind)
    grid = np.append(np.arange(md[0], md[-1], step), md[-1])  # uniform + TD
    grid = np.unique(grid)
    inc_new = np.clip(inc_spline(grid), 0.0, 180.0)
    azi_new = np.mod(azi_spline(grid), 360.0)
    return Survey(grid, inc_new, azi_new, surface=survey.surface)
