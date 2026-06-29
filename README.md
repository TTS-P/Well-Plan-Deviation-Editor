# Well Plan Deviation Editor

A Streamlit app for directional-drilling well planning. Import a well survey from
Excel, interpolate a smooth trajectory with splines, edit the planned geometry,
and download a regenerated survey as Excel.

## What it does

1. **Import** a survey spreadsheet with `MD, Inc, Azi` (plus optional
   `TVD, Northing, Easting`). Columns are auto-detected; the mapping is editable.
   File-supplied positions are validated against recomputed minimum-curvature
   values, with `MD/Inc/Azi` treated as the source of truth.
2. **Add / edit / remove stations** in an editable table. The spline passes
   through every station, so inserting real survey readings (MD, Inc, Azi) makes
   the plan match the actual well. This redefines the interpolation baseline.
3. **Interpolate** by fitting splines to inclination & azimuth vs measured depth
   (Cubic or PCHIP), then resampling at a chosen interval and integrating to
   positions with the **minimum-curvature method**.
4. **Detect structure** — vertical, build, hold, drop and lateral segments, plus
   the kickoff point (KOP), heel/landing and toe.
5. **Edit geometry**:
   - Move the **kickoff point** (rigid shift, or "keep target fixed" re-solve).
   - Change a **build rate** (°/100 ft) or a **section length**.
   - Move the **landing/heel** or **target/toe** to a new `(N, E, TVD)` point.
     The path is morphed in position space — the curve above a held anchor stays
     put, the point lands *exactly* on the target (so an unchanged coordinate,
     e.g. TVD, is preserved), and anything below it (the lateral, including a
     U-shaped one) rides along with its shape intact. An aggressive move shows up
     as higher dogleg severity, which is reported for drillability.
6. **Resample output to interval** (optional) — geometry edits reshape the
   station spacing; this re-grids the survey onto uniform MD stations (TD kept)
   before tortuosity and export.
7. **Add tortuosity** — seeded Gaussian inc/azi deviation driven by a 0–5 level,
   where level 5 is auto-calibrated so the result reaches ~20°/100 ft max DLS.
   Pick which sections (vertical/build/lateral) to roughen. Reproducible per seed
   and applied as a live, non-destructive overlay, so the smooth plan and its
   geometry edits are preserved.
8. **Compare** original vs modified in plan, vertical-section and 3D views, with
   a delta summary (TVD, step-out, TD, max DLS).
9. **Export** the modified survey as a multi-sheet `.xlsx`
   (Modified / Original / Summary).

## Setup & run

```bash
pip install -r requirements.txt
python make_sample.py          # optional: create sample_survey.xlsx
streamlit run app.py
```

Then upload `sample_survey.xlsx` (or your own) in the sidebar.

## Layout

```
app.py              Streamlit UI / orchestration
make_sample.py      Generates a synthetic build-and-hold-plus-lateral survey
check_math.py       Sanity checks for the survey math and editing engine
wellplan/
  survey.py         Minimum-curvature math + Survey dataclass (source of truth)
  interpolate.py    Spline fit + resample of Inc/Azi vs MD
  wellmodel.py      Segment detection, MD-domain edits, position-morph target move
  tortuosity.py     Sine / random tortuosity overlay
  io_excel.py       Excel read / validate / write
```

## Conventions

Feet, inclination in degrees from vertical, azimuth 0–360° clockwise from grid
north, DLS in degrees per 100 ft.
