"""Well-structure detection and parametric geometry editing.

This module turns a dense survey into a sequence of recognisable segments
(vertical / build / hold / drop / lateral), exposes the key control points
(kickoff point, heel/landing, toe), and provides the editing operations that
regenerate the trajectory:

* **MD-domain edits** (robust): move the KOP, change a build rate, change a
  section length.  These reshape the ``Inc/Azi`` vs ``MD`` profile by rescaling
  one segment in MD and shifting everything below it, then re-run minimum
  curvature.
* **Position-domain edits** (inverse problem): move the heel/landing or the
  toe/target to a new ``(North, East, TVD)`` point.  Below a fixed anchor the
  lower well is re-planned as a build-and-hold geometry, and
  ``scipy.optimize.least_squares`` solves for the orientation/length that lands
  on the target subject to a max-DLS limit.

Every edit ultimately emits an ``Inc/Azi`` profile that is fed back through
:func:`wellplan.survey.minimum_curvature`, so positions are always consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wellplan.survey import Survey

__all__ = [
    "Segment",
    "ControlPoints",
    "detect_segments",
    "control_points",
    "move_kop",
    "set_build_rate",
    "set_section_length",
    "move_point_to_target",
    "section_mask",
    "SolveResult",
]

# Classification thresholds.
VERT_INC = 2.0  # deg: below this the hole is treated as vertical
LATERAL_INC = 80.0  # deg: above this a hold is treated as a lateral
BUILD_THRESH = 0.5  # deg/100ft: |rate| above this counts as build/drop


@dataclass
class Segment:
    kind: str  # vertical | build | hold | drop | lateral
    i_start: int
    i_end: int
    md_start: float
    md_end: float
    inc_start: float
    inc_end: float
    azi_start: float
    azi_end: float

    @property
    def length(self) -> float:
        return self.md_end - self.md_start

    @property
    def build_rate(self) -> float:
        """Signed inclination build rate (deg / 100 ft)."""
        if self.length <= 0:
            return 0.0
        return (self.inc_end - self.inc_start) * 100.0 / self.length

    def describe(self) -> dict:
        return {
            "Segment": self.kind,
            "MD start": round(self.md_start, 1),
            "MD end": round(self.md_end, 1),
            "Length": round(self.length, 1),
            "Inc start": round(self.inc_start, 2),
            "Inc end": round(self.inc_end, 2),
            "Build rate (deg/100ft)": round(self.build_rate, 2),
        }


@dataclass
class ControlPoints:
    kop_md: float | None
    heel_md: float | None
    toe_md: float
    kop_index: int | None
    heel_index: int | None


def _classify_interval(inc0: float, inc1: float, rate: float) -> str:
    avg = 0.5 * (inc0 + inc1)
    if avg < VERT_INC:
        return "vertical"
    if rate > BUILD_THRESH:
        return "build"
    if rate < -BUILD_THRESH:
        return "drop"
    if avg > LATERAL_INC:
        return "lateral"
    return "hold"


def detect_segments(survey: Survey) -> list[Segment]:
    """Classify a survey into contiguous geometric segments."""
    md, inc, azi = survey.md, survey.inc, survey.azi
    n = md.size
    if n < 2:
        return []

    dmd = np.diff(md)
    dinc = np.diff(inc)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(dmd > 0, dinc / dmd * 100.0, 0.0)

    kinds = [
        _classify_interval(inc[i], inc[i + 1], rate[i]) for i in range(n - 1)
    ]

    segments: list[Segment] = []
    start = 0
    for i in range(1, len(kinds) + 1):
        if i == len(kinds) or kinds[i] != kinds[start]:
            seg = Segment(
                kind=kinds[start],
                i_start=start,
                i_end=i,  # station index where the segment ends
                md_start=float(md[start]),
                md_end=float(md[i]),
                inc_start=float(inc[start]),
                inc_end=float(inc[i]),
                azi_start=float(azi[start]),
                azi_end=float(azi[i]),
            )
            segments.append(seg)
            start = i
    return segments


def section_mask(
    survey: Survey,
    vertical: bool = False,
    build: bool = True,
    lateral: bool = True,
    cp: "ControlPoints | None" = None,
) -> np.ndarray:
    """Per-station boolean mask selecting the requested well sections.

    Sections are bounded by the detected KOP and heel:
    ``vertical`` (above KOP), ``build`` (KOP..heel) and ``lateral`` (heel..TD).
    Useful for confining tortuosity/deviation to chosen parts of the hole.
    """
    if cp is None:
        cp = control_points(survey)
    n = survey.n_stations
    kop = cp.kop_index if cp.kop_index is not None else 0
    lat = cp.heel_index if cp.heel_index is not None else n - 1

    idx = np.arange(n)
    is_vertical = idx < kop
    is_lateral = idx >= lat
    is_build = ~is_vertical & ~is_lateral

    mask = np.zeros(n, dtype=bool)
    if vertical:
        mask |= is_vertical
    if build:
        mask |= is_build
    if lateral:
        mask |= is_lateral
    return mask


def control_points(survey: Survey, segments: list[Segment] | None = None) -> ControlPoints:
    """Identify the KOP, heel/landing and toe from the segment list."""
    if segments is None:
        segments = detect_segments(survey)

    kop_md = kop_index = None
    for seg in segments:
        if seg.kind == "build":
            kop_md = seg.md_start
            kop_index = seg.i_start
            break

    heel_md = heel_index = None
    # Heel = start of the last lateral segment, else start of the last hold.
    for seg in reversed(segments):
        if seg.kind == "lateral":
            heel_md = seg.md_start
            heel_index = seg.i_start
            break
    if heel_md is None:
        for seg in reversed(segments):
            if seg.kind == "hold" and seg.inc_start > 45.0:
                heel_md = seg.md_start
                heel_index = seg.i_start
                break

    return ControlPoints(
        kop_md=kop_md,
        heel_md=heel_md,
        toe_md=survey.total_depth,
        kop_index=kop_index,
        heel_index=heel_index,
    )


# ---------------------------------------------------------------------------
# MD-domain edits
# ---------------------------------------------------------------------------


def _rescale_segment(md: np.ndarray, i0: int, i1: int, new_length: float) -> np.ndarray:
    """Rescale the MD span of stations ``i0..i1`` to ``new_length`` and shift
    every station below by the resulting delta.  Inc/Azi stay attached to their
    (re-positioned) stations.
    """
    md = md.copy()
    old_length = md[i1] - md[i0]
    if old_length > 1e-9:
        scale = new_length / old_length
        md[i0 : i1 + 1] = md[i0] + (md[i0 : i1 + 1] - md[i0]) * scale
    else:
        md[i0 : i1 + 1] = np.linspace(md[i0], md[i0] + new_length, i1 - i0 + 1)
    delta = new_length - old_length
    md[i1 + 1 :] = md[i1 + 1 :] + delta
    return md


def move_kop(survey: Survey, new_kop_md: float) -> Survey:
    """Move the kickoff point to ``new_kop_md`` (rigid shift of the lower well).

    The vertical segment is rescaled to end at the new KOP; the entire build/
    hold/lateral profile below shifts in MD by the same delta, preserving its
    shape.  The landing/target therefore moves with it.
    """
    cp = control_points(survey)
    if cp.kop_index is None:
        raise ValueError("no kickoff point detected; cannot move KOP")
    if new_kop_md <= survey.md[0]:
        raise ValueError("new KOP must be below the surface station")
    new_md = _rescale_segment(survey.md, 0, cp.kop_index, new_kop_md - survey.md[0])
    return Survey(new_md, survey.inc.copy(), survey.azi.copy(), survey.surface)


def set_build_rate(survey: Survey, segment_index: int, new_rate: float) -> Survey:
    """Change a build/drop segment's rate (deg/100ft), holding its Inc change.

    The segment's MD length becomes ``|delta_inc| / new_rate * 100`` and the
    well below shifts accordingly.
    """
    segments = detect_segments(survey)
    if not 0 <= segment_index < len(segments):
        raise IndexError("segment_index out of range")
    seg = segments[segment_index]
    if new_rate <= 0:
        raise ValueError("build rate must be positive")
    dinc = abs(seg.inc_end - seg.inc_start)
    if dinc < 1e-6:
        raise ValueError("selected segment has no inclination change to re-rate")
    new_length = dinc / new_rate * 100.0
    new_md = _rescale_segment(survey.md, seg.i_start, seg.i_end, new_length)
    return Survey(new_md, survey.inc.copy(), survey.azi.copy(), survey.surface)


def set_section_length(survey: Survey, segment_index: int, new_length: float) -> Survey:
    """Set the MD length of a segment (typically a hold or lateral)."""
    segments = detect_segments(survey)
    if not 0 <= segment_index < len(segments):
        raise IndexError("segment_index out of range")
    if new_length <= 0:
        raise ValueError("section length must be positive")
    seg = segments[segment_index]
    new_md = _rescale_segment(survey.md, seg.i_start, seg.i_end, new_length)
    return Survey(new_md, survey.inc.copy(), survey.azi.copy(), survey.surface)


# ---------------------------------------------------------------------------
# Position-domain edits (inverse problem)
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    survey: Survey
    miss_distance: float  # ft from the requested target
    max_dls: float  # deg/100ft of the re-planned section
    success: bool


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth 0->1 ramp with zero slope at both ends (3t^2 - 2t^3)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _positions_to_survey(
    pos: np.ndarray, md0: float, surface: tuple[float, float, float]
) -> Survey:
    """Back out an MD/Inc/Azi survey from a 3D position polyline.

    Inclination/azimuth come from the station tangents (a central-difference
    bisector, which matches the minimum-curvature arc tangent for smooth paths);
    MD is the cumulative chord length offset to start at ``md0``.
    """
    d = np.diff(pos, axis=0)
    seg = np.linalg.norm(d, axis=1)
    unit = d / np.where(seg[:, None] > 0, seg[:, None], 1.0)

    tang = np.zeros_like(pos)
    tang[1:-1] = unit[:-1] + unit[1:]  # bisector of adjacent chords
    tang[0] = unit[0]
    tang[-1] = unit[-1]
    tn = np.linalg.norm(tang, axis=1)
    tang = tang / np.where(tn[:, None] > 0, tn[:, None], 1.0)

    inc = np.degrees(np.arctan2(np.hypot(tang[:, 0], tang[:, 1]), tang[:, 2]))
    azi = np.mod(np.degrees(np.arctan2(tang[:, 1], tang[:, 0])), 360.0)
    md = np.concatenate([[0.0], np.cumsum(seg)]) + md0
    return Survey(md, inc, azi, surface)


def move_point_to_target(
    survey: Survey,
    ref_md: float,
    target: tuple[float, float, float],
    anchor_md: float,
) -> SolveResult:
    """Move the station at ``ref_md`` to a 3D ``target`` by morphing the path.

    The trajectory is deformed in *position* space: a smooth weight ramps from
    0 at ``anchor_md`` to 1 at ``ref_md`` (and stays 1 below it), and the
    displacement needed at ``ref_md`` is applied through that weight.  The
    section above ``anchor_md`` is untouched, the point at ``ref_md`` lands
    exactly on the target, and anything below ``ref_md`` (e.g. the lateral)
    rides along rigidly, keeping its shape -- including U-shaped laterals.

    Because the target is hit by construction, this never "misses": there is no
    optimiser, no bounds, and no DLS feasibility limit.  The cost of an
    aggressive move shows up instead as higher dogleg severity (``max_dls`` over
    the morphed section), which the caller can surface for drillability.
    """
    target = np.asarray(target, dtype=float)  # (north, east, tvd)
    if anchor_md >= ref_md:
        raise ValueError("anchor must be above the point being moved")

    pos = np.column_stack([survey.north, survey.east, survey.tvd]).astype(float)
    ref_idx = int(np.argmin(np.abs(survey.md - ref_md)))

    md = survey.md
    span = max(md[ref_idx] - anchor_md, 1e-9)
    weight = _smoothstep((md - anchor_md) / span)
    weight[md <= anchor_md] = 0.0
    weight[md >= md[ref_idx]] = 1.0  # tail (lateral) translates rigidly
    w = weight[:, None]

    # Displacement at the reference point; a few fixed-point passes absorb the
    # small round-trip error of the position->survey inversion so the hit is exact.
    disp = target - pos[ref_idx]
    new_survey = survey
    hit_err = 0.0
    for _ in range(5):
        new_pos = pos + w * disp
        new_survey = _positions_to_survey(new_pos, float(survey.md[0]), survey.surface)
        landed = np.array(
            [new_survey.north[ref_idx], new_survey.east[ref_idx], new_survey.tvd[ref_idx]]
        )
        err = target - landed
        hit_err = float(np.linalg.norm(err))
        if hit_err < 1e-3:
            break
        disp = disp + err

    # Dogleg severity introduced over the morphed section (anchor..ref).
    morphed = (md > anchor_md) & (md <= md[ref_idx])
    seg_dls = new_survey.dls[morphed]
    max_dls = float(np.max(seg_dls)) if seg_dls.size else 0.0
    return SolveResult(new_survey, hit_err, max_dls, hit_err < 1.0)
