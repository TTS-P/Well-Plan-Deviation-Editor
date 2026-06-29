"""Add tortuosity (drilling deviation) to a smooth well path.

A planned trajectory is perfectly smooth, but a real drilled wellbore wanders.
This module reproduces that wander with the *deviation-level* model: seeded
Gaussian perturbations are applied to inclination and azimuth, scaled by a
single level in ``[0, 5]``.  Level 5 is auto-calibrated (by bisection on the
perturbation magnitude) so the resulting trajectory's maximum dogleg severity
reaches :data:`MAX_DLS_TARGET` deg/100 ft; intermediate levels scale linearly.

Because the seed is fixed, a given level reproduces the same wellbore run to run.
A per-station ``deviate_mask`` lets the caller confine deviation to chosen
sections (e.g. leave the vertical section clean and only rough up the lateral).

The original survey is never mutated -- a new :class:`~wellplan.survey.Survey`
is returned, so deviation is applied as a non-destructive overlay.
"""

from __future__ import annotations

import numpy as np

from wellplan.survey import Survey, minimum_curvature

__all__ = ["apply_deviation", "MAX_DLS_TARGET", "DEFAULT_SEED"]

# DLS (deg/100ft) that level 5 is calibrated to produce.
MAX_DLS_TARGET = 20.0
# Fixed RNG seed so a given level is reproducible run-to-run.
DEFAULT_SEED = 42


def _max_dls(md: np.ndarray, inc: np.ndarray, azi: np.ndarray) -> float:
    """Maximum dogleg severity (deg/100ft) of a trajectory."""
    if md.size < 2:
        return 0.0
    *_, dls = minimum_curvature(md, inc, azi)
    return float(np.max(dls))


def _deviated(inc, azi, d_inc, d_azi, magnitude):
    """Apply scaled perturbations, returning clipped/wrapped inc & azi."""
    inc_out = np.clip(inc + magnitude * d_inc, 0.0, 180.0)
    azi_out = np.mod(azi + magnitude * d_azi, 360.0)
    return inc_out, azi_out


def apply_deviation(
    survey: Survey,
    level: float,
    seed: int = DEFAULT_SEED,
    deviate_mask: np.ndarray | None = None,
) -> tuple[Survey, dict]:
    """Return ``(deviated_survey, info)`` with tortuosity applied to inc/azi.

    Parameters
    ----------
    survey : Survey
        Smooth path to roughen (left unmodified).
    level : float in [0, 5]
        0 = unchanged; 5 = scaled so the resulting trajectory's max DLS reaches
        :data:`MAX_DLS_TARGET`.  Intermediate values scale the magnitude linearly.
    seed : int
        RNG seed for the (reproducible) Gaussian perturbations.
    deviate_mask : array of bool, optional
        Per-station mask; stations that are ``False`` are held fixed (e.g. a
        protected vertical section).  ``None`` deviates every station.

    Returns
    -------
    (Survey, info) where ``info`` carries ``original_max_dls``, ``result_max_dls``,
    ``magnitude`` and ``warning``.
    """
    md = survey.md.astype(float)
    inc = survey.inc.astype(float)
    azi = survey.azi.astype(float)

    original_max = _max_dls(md, inc, azi)
    info = {
        "original_max_dls": original_max,
        "result_max_dls": original_max,
        "magnitude": 0.0,
        "warning": None,
    }

    level = float(level)
    scale = level / 5.0
    if scale <= 0 or md.size < 2:
        return survey.copy(), info

    # Unit perturbation vectors (deg) -- fixed seed for reproducibility.
    rng = np.random.default_rng(seed)
    d_inc = rng.standard_normal(md.size)
    d_azi = rng.standard_normal(md.size)

    if deviate_mask is not None:
        mask = np.asarray(deviate_mask, dtype=bool)
        d_inc = np.where(mask, d_inc, 0.0)
        d_azi = np.where(mask, d_azi, 0.0)

    if np.allclose(d_inc, 0.0) and np.allclose(d_azi, 0.0):
        info["warning"] = "No stations are eligible for deviation (check the section selection)."
        return survey.copy(), info

    def result_max_for(magnitude: float) -> float:
        i_o, a_o = _deviated(inc, azi, d_inc, d_azi, magnitude)
        return _max_dls(md, i_o, a_o)

    if original_max >= MAX_DLS_TARGET:
        # Cannot lower existing curvature; use a magnitude proportional to the
        # inclination range and warn that the 20 deg/100ft target is not enforced.
        inc_range = max(float(np.ptp(inc)), 1.0)
        m5 = 0.10 * inc_range
        info["warning"] = (
            f"Well already exceeds {MAX_DLS_TARGET:.0f}°/100ft (max {original_max:.1f}); "
            "deviation is added on top and the level-5 calibration is not enforced."
        )
    else:
        # Bisection on magnitude: find m5 so result max DLS == MAX_DLS_TARGET
        # (result_max is monotonically increasing in magnitude).
        lo, hi = 0.0, 1.0
        for _ in range(60):  # grow the upper bound until it overshoots the target
            if result_max_for(hi) >= MAX_DLS_TARGET:
                break
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if result_max_for(mid) < MAX_DLS_TARGET:
                lo = mid
            else:
                hi = mid
        m5 = 0.5 * (lo + hi)

    magnitude = scale * m5
    inc_out, azi_out = _deviated(inc, azi, d_inc, d_azi, magnitude)
    deviated = Survey(md, inc_out, azi_out, surface=survey.surface)

    info["magnitude"] = float(magnitude)
    info["result_max_dls"] = deviated.max_dls
    return deviated, info
