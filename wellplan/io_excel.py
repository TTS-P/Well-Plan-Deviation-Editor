"""Excel import / validation / export for well surveys."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wellplan.survey import Survey

__all__ = ["ColumnMap", "read_table", "build_survey", "validate", "write_survey"]

# Header keyword -> canonical field.  First match wins, longest patterns first.
_PATTERNS: dict[str, list[str]] = {
    "md": [r"\bmd\b", r"measured.?depth", r"\bdepth\b", r"\bmeas"],
    "inc": [r"\binc", r"inclination", r"\bdip\b"],
    "azi": [r"\bazi", r"azimuth", r"\bhead", r"\bbearing"],
    "tvd": [r"\btvd\b", r"true.?vert"],
    "north": [r"\bnorth", r"\bn/s\b", r"\bns\b", r"\by\b", r"\bnorthing"],
    "east": [r"\beast", r"\be/w\b", r"\bew\b", r"\bx\b", r"\beasting"],
}


@dataclass
class ColumnMap:
    md: str | None = None
    inc: str | None = None
    azi: str | None = None
    tvd: str | None = None
    north: str | None = None
    east: str | None = None

    def required_ok(self) -> bool:
        return bool(self.md and self.inc and self.azi)


def _auto_match(columns: list[str]) -> ColumnMap:
    """Fuzzy-match spreadsheet headers to canonical survey fields."""
    cmap = ColumnMap()
    used: set[str] = set()
    for field, patterns in _PATTERNS.items():
        for pat in patterns:
            for col in columns:
                if col in used:
                    continue
                if re.search(pat, str(col).strip().lower()):
                    setattr(cmap, field, col)
                    used.add(col)
                    break
            if getattr(cmap, field) is not None:
                break
    return cmap


def _rewind(file) -> None:
    """Rewind a file-like buffer so it can be read again across reruns."""
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except (OSError, ValueError):
            pass


def list_sheets(file) -> list[str]:
    """Return the sheet names in a workbook."""
    _rewind(file)
    return pd.ExcelFile(file).sheet_names


def read_table(file, sheet_name=0) -> tuple[pd.DataFrame, ColumnMap, list[str]]:
    """Read a survey sheet and return ``(dataframe, auto_columnmap, sheet_names)``."""
    _rewind(file)
    xls = pd.ExcelFile(file)
    df = xls.parse(sheet_name)
    df = df.dropna(how="all").reset_index(drop=True)
    cmap = _auto_match(list(df.columns))
    return df, cmap, xls.sheet_names


def build_survey(
    df: pd.DataFrame, cmap: ColumnMap, surface: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[Survey, dict[str, np.ndarray]]:
    """Build a :class:`Survey` from a dataframe + column mapping.

    Returns the survey (positions recomputed from MD/Inc/Azi) plus any
    position columns found in the file, for validation.
    """
    if not cmap.required_ok():
        raise ValueError("MD, Inc and Azi columns must all be mapped")

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    md = col(cmap.md)
    inc = col(cmap.inc)
    azi = col(cmap.azi)

    mask = ~(np.isnan(md) | np.isnan(inc) | np.isnan(azi))
    md, inc, azi = md[mask], inc[mask], azi[mask]
    order = np.argsort(md, kind="stable")
    md, inc, azi = md[order], inc[order], azi[order]

    if md.size < 2:
        raise ValueError("need at least two valid survey stations")

    survey = Survey(md, inc, azi, surface=surface)

    file_pos: dict[str, np.ndarray] = {}
    for field in ("tvd", "north", "east"):
        name = getattr(cmap, field)
        if name and name in df.columns:
            vals = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            file_pos[field] = vals[mask][order]
    return survey, file_pos


def validate(survey: Survey, file_pos: dict[str, np.ndarray]) -> dict[str, float]:
    """Compare file-supplied positions to recomputed ones; return max |diff|."""
    out: dict[str, float] = {}
    computed = {"tvd": survey.tvd, "north": survey.north, "east": survey.east}
    for field, vals in file_pos.items():
        if vals is None or np.all(np.isnan(vals)):
            continue
        diff = np.abs(vals - computed[field])
        out[field] = float(np.nanmax(diff))
    return out


def survey_dataframe(survey: Survey) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD (ft)": survey.md,
            "Inc (deg)": survey.inc,
            "Azi (deg)": survey.azi,
            "TVD (ft)": survey.tvd,
            "Northing (ft)": survey.north,
            "Easting (ft)": survey.east,
            "DLS (deg/100ft)": survey.dls,
        }
    )


def write_survey(
    survey: Survey, original: Survey | None = None, summary: dict | None = None
) -> bytes:
    """Serialise a survey to a multi-sheet ``.xlsx`` workbook (returns bytes)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        survey_dataframe(survey).to_excel(writer, sheet_name="Modified_Survey", index=False)
        if original is not None:
            survey_dataframe(original).to_excel(
                writer, sheet_name="Original_Survey", index=False
            )
        if summary:
            pd.DataFrame(
                {"Property": list(summary.keys()), "Value": list(summary.values())}
            ).to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    return buffer.getvalue()
