"""Well Plan Deviation Editor — Streamlit UI.

Upload a directional survey, add/edit survey stations, interpolate it with
splines, edit the planned geometry (kickoff point, build rates, section
lengths, landing/target positions), then download the regenerated survey as
Excel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wellplan.interpolate import SPLINE_KINDS, interpolate_survey, regrid_survey
from wellplan.io_excel import (
    ColumnMap,
    build_survey,
    list_sheets,
    read_table,
    survey_dataframe,
    validate,
    write_survey,
)
from wellplan.survey import Survey
from wellplan.tortuosity import MAX_DLS_TARGET, apply_deviation
from wellplan.wellmodel import (
    control_points,
    detect_segments,
    move_kop,
    move_point_to_target,
    section_mask,
)

st.set_page_config(page_title="Well Plan Editor", layout="wide")

ORIG_COLOR = "#888888"
MOD_COLOR = "#1f77b4"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plan_view(orig: Survey, mod: Survey) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=orig.east, y=orig.north, mode="lines",
                             name="Original", line=dict(color=ORIG_COLOR, dash="dash")))
    fig.add_trace(go.Scatter(x=mod.east, y=mod.north, mode="lines",
                             name="Modified", line=dict(color=MOD_COLOR)))
    fig.update_layout(title="Plan view", xaxis_title="Easting (ft)",
                      yaxis_title="Northing (ft)", height=420,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def section_view(orig: Survey, mod: Survey) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=orig.vertical_section, y=orig.tvd, mode="lines",
                             name="Original", line=dict(color=ORIG_COLOR, dash="dash")))
    fig.add_trace(go.Scatter(x=mod.vertical_section, y=mod.tvd, mode="lines",
                             name="Modified", line=dict(color=MOD_COLOR)))
    fig.update_layout(title="Vertical section", xaxis_title="Horizontal departure (ft)",
                      yaxis_title="TVD (ft)", height=420,
                      yaxis=dict(autorange="reversed"))
    return fig


def view_3d(orig: Survey, mod: Survey) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=orig.east, y=orig.north, z=orig.tvd, mode="lines",
                               name="Original", line=dict(color=ORIG_COLOR, width=3)))
    fig.add_trace(go.Scatter3d(x=mod.east, y=mod.north, z=mod.tvd, mode="lines",
                               name="Modified", line=dict(color=MOD_COLOR, width=5)))
    fig.update_layout(title="3D trajectory", height=520,
                      scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)",
                                 zaxis_title="TVD (ft)", zaxis=dict(autorange="reversed")))
    return fig


def summary_dict(survey: Survey) -> dict:
    cp = control_points(survey)
    return {
        "KOP MD (ft)": round(cp.kop_md, 1) if cp.kop_md else "n/a",
        "Heel MD (ft)": round(cp.heel_md, 1) if cp.heel_md else "n/a",
        "Total depth (ft)": round(survey.total_depth, 1),
        "TVD @ TD (ft)": round(float(survey.tvd[-1]), 1),
        "Step-out (ft)": round(survey.step_out, 1),
        "Max DLS (deg/100ft)": round(survey.max_dls, 2),
    }


# ---------------------------------------------------------------------------
# Sidebar: upload + column mapping + interpolation
# ---------------------------------------------------------------------------

st.title("Well Plan Editor")
st.caption("Import a survey · spline-interpolate · edit geometry · export a new survey. "
           "Units: feet, azimuth from grid north.")

with st.sidebar:
    st.header("1 · Survey input")
    uploaded = st.file_uploader("Survey spreadsheet (.xlsx)", type=["xlsx", "xls"])
    if uploaded is None:
        st.info("Upload a survey, or run `python make_sample.py` to create "
                "`sample_survey.xlsx`.")

if uploaded is None:
    st.stop()

# Reset cached state if a different file is uploaded.
if st.session_state.get("file_name") != uploaded.name:
    st.session_state.clear()
    st.session_state["file_name"] = uploaded.name

sheet_names = list_sheets(uploaded)
with st.sidebar:
    sheet = st.selectbox("Worksheet", sheet_names, index=0,
                         help="Pick which sheet in the workbook holds the survey.") \
        if len(sheet_names) > 1 else sheet_names[0]

df, auto_map, _ = read_table(uploaded, sheet_name=sheet)

with st.sidebar:
    cols = ["<none>"] + list(df.columns)

    def _pick(label, current):
        idx = cols.index(current) if current in cols else 0
        choice = st.selectbox(label, cols, index=idx)
        return None if choice == "<none>" else choice

    with st.expander("Column mapping", expanded=not auto_map.required_ok()):
        cmap = ColumnMap(
            md=_pick("MD", auto_map.md),
            inc=_pick("Inclination", auto_map.inc),
            azi=_pick("Azimuth", auto_map.azi),
            tvd=_pick("TVD (optional)", auto_map.tvd),
            north=_pick("Northing (optional)", auto_map.north),
            east=_pick("Easting (optional)", auto_map.east),
        )

    st.header("2 · Interpolation")
    kind = st.selectbox("Spline type", SPLINE_KINDS, index=0,
                        help="Cubic is smoothest; PCHIP avoids DLS overshoot at sharp corners.")
    step = st.number_input("Resample interval (ft)", 5.0, 200.0, 100.0, 5.0)

if not cmap.required_ok():
    st.warning("Map the MD, Inclination and Azimuth columns to continue.")
    st.stop()

try:
    raw_survey, file_pos = build_survey(df, cmap)
except ValueError as exc:
    st.error(f"Could not read survey: {exc}")
    st.stop()

# Validate file-supplied positions against recomputed ones.
mismatches = validate(raw_survey, file_pos)
if mismatches:
    worst = ", ".join(f"{k}: {v:.1f} ft" for k, v in mismatches.items())
    if max(mismatches.values()) > 1.0:
        st.warning(f"File positions differ from recomputed minimum-curvature values "
                   f"(max abs diff — {worst}). Using MD/Inc/Azi as the source of truth.")

# ---------------------------------------------------------------------------
# Survey stations — add / edit / remove control points
# ---------------------------------------------------------------------------

st.subheader("Survey stations")
st.caption("Add, edit or remove stations (MD · Inc · Azi). The spline passes "
           "through every station, so inserting real survey readings makes the "
           "plan match the actual well. Use the ＋ at the bottom of the table to "
           "add a station; edits here redefine the baseline below.")

station_seed = pd.DataFrame(
    {"MD (ft)": raw_survey.md, "Inc (deg)": raw_survey.inc, "Azi (deg)": raw_survey.azi}
)
edited = st.data_editor(
    station_seed,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    height=280,
    key=f"stations::{uploaded.name}::{sheet}",
    column_config={
        "MD (ft)": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
        "Inc (deg)": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=180.0),
        "Azi (deg)": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=360.0),
    },
)

# Build the working survey from the edited stations (cleaned + sorted by MD).
clean = edited.apply(pd.to_numeric, errors="coerce").dropna(how="any")
clean = clean[(clean["Inc (deg)"] >= 0) & (clean["MD (ft)"] >= 0)]
clean = clean.drop_duplicates(subset="MD (ft)").sort_values("MD (ft)")

if len(clean) < 2:
    st.warning("Need at least two valid stations — falling back to the imported survey.")
    work_survey = raw_survey
else:
    n_added = len(clean) - raw_survey.n_stations
    if n_added != 0:
        verb = "added" if n_added > 0 else "removed"
        st.info(f"{abs(n_added)} station(s) {verb} · {len(clean)} control points in the plan.")
    work_survey = Survey(
        clean["MD (ft)"].to_numpy(float),
        clean["Inc (deg)"].to_numpy(float),
        clean["Azi (deg)"].to_numpy(float),
        surface=raw_survey.surface,
    )

# Baseline interpolated survey (the "original" for editing & comparison).
interp = interpolate_survey(work_survey, step=step, kind=kind)
st.session_state["interp"] = interp

# Reset the edited plan whenever the station set or interpolation settings change.
signature = (
    round(float(work_survey.md.sum()), 4),
    round(float(work_survey.inc.sum()), 4),
    round(float(work_survey.azi.sum()), 4),
    work_survey.n_stations,
    sheet,
    kind,
    float(step),
)
if st.session_state.get("stations_sig") != signature:
    st.session_state["stations_sig"] = signature
    st.session_state["modified"] = interp.copy()
if "modified" not in st.session_state:
    st.session_state["modified"] = interp.copy()

# ---------------------------------------------------------------------------
# Detected structure
# ---------------------------------------------------------------------------

segments = detect_segments(interp)
cp = control_points(interp, segments)

st.subheader("Detected well structure")
c1, c2, c3, c4 = st.columns(4)
c1.metric("KOP MD", f"{cp.kop_md:.0f} ft" if cp.kop_md else "—")
c2.metric("Heel MD", f"{cp.heel_md:.0f} ft" if cp.heel_md else "—")
c3.metric("Total depth", f"{interp.total_depth:.0f} ft")
c4.metric("Max DLS", f"{interp.max_dls:.2f}°/100ft")
st.dataframe([s.describe() for s in segments], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Geometry editors
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("3 · Edit geometry")
    edit = st.selectbox(
        "Edit type",
        ["Move kickoff point", "Move landing / heel", "Move target / toe"],
    )
    base = st.session_state["modified"]
    seg_now = detect_segments(base)
    cp_now = control_points(base, seg_now)

    new_survey = None
    note = ""
    try:
        if edit == "Move kickoff point":
            cur = cp_now.kop_md or interp.md[1]
            new_kop = st.number_input("New KOP MD (ft)", float(interp.md[0]) + 1,
                                      float(base.total_depth), float(cur), 50.0)
            keep_target = st.checkbox("Keep landing/target fixed", value=False,
                                      help="Re-solve the lower well to land at the original "
                                           "heel instead of shifting it.")
            if st.button("Apply KOP change", use_container_width=True):
                if keep_target and cp_now.heel_md is not None:
                    tgt = (float(np.interp(cp_now.heel_md, base.md, base.north)),
                           float(np.interp(cp_now.heel_md, base.md, base.east)),
                           float(np.interp(cp_now.heel_md, base.md, base.tvd)))
                    shifted = move_kop(base, new_kop)
                    heel_shifted = control_points(shifted).heel_md
                    res = move_point_to_target(shifted, heel_shifted, tgt, new_kop)
                    new_survey = res.survey
                    note = (f"Re-tied to original heel · hit {res.miss_distance:.2f} ft "
                            f"· max DLS {res.max_dls:.1f}°/100ft")
                else:
                    new_survey = move_kop(base, new_kop)

        elif edit in ("Move landing / heel", "Move target / toe"):
            if edit == "Move landing / heel":
                ref_md = cp_now.heel_md
                anchor_md = cp_now.kop_md or interp.md[1]
            else:
                ref_md = base.total_depth
                anchor_md = cp_now.heel_md or cp_now.kop_md or interp.md[1]
            if ref_md is None:
                st.info("Could not locate this control point in the current well.")
            else:
                n_c = float(np.interp(ref_md, base.md, base.north))
                e_c = float(np.interp(ref_md, base.md, base.east))
                v_c = float(np.interp(ref_md, base.md, base.tvd))
                tail_kept = " The lateral below it rides along." \
                    if edit == "Move landing / heel" else ""
                st.caption(f"Curve above MD {anchor_md:.0f} ft is held fixed; the point "
                           f"lands exactly on the target.{tail_kept}")
                tn = st.number_input("Target Northing (ft)", value=round(n_c, 1), step=50.0)
                te = st.number_input("Target Easting (ft)", value=round(e_c, 1), step=50.0)
                tv = st.number_input("Target TVD (ft)", value=round(v_c, 1), step=20.0)
                # The move always hits the target; an aggressive move just needs a
                # tighter dogleg. Warn if it exceeds a drillability threshold,
                # defaulted to the well's own build rate.
                build_rates = [abs(s.build_rate) for s in seg_now
                               if s.kind in ("build", "drop") and abs(s.build_rate) > 1e-3]
                default_dls = float(np.clip(round(max(build_rates), 1), 1.0, 60.0)) \
                    if build_rates else 6.0
                max_dls_warn = st.number_input("Max allowable DLS (°/100ft)", 1.0, 60.0,
                                               default_dls, 0.5,
                                               help="The point always reaches the target. If the "
                                                    "morph needs a tighter dogleg than this, you "
                                                    "are warned it may not be drillable.")
                if st.button("Move to target", use_container_width=True):
                    res = move_point_to_target(base, ref_md, (tn, te, tv), anchor_md)
                    new_survey = res.survey
                    drillable = res.max_dls <= max_dls_warn + 1e-6
                    flag = "✅" if drillable else "⚠️"
                    note = (f"{flag} hit target within {res.miss_distance:.2f} ft · "
                            f"max DLS {res.max_dls:.1f}°/100ft")
                    if not drillable:
                        st.warning(
                            f"Target reached, but the bend needs up to {res.max_dls:.0f}°/100ft "
                            f"— above your {max_dls_warn:.0f}°/100ft limit. To soften it, move the "
                            "point a smaller distance or move the held anchor further up the hole.")
    except (ValueError, IndexError) as exc:
        st.error(str(exc))

    if new_survey is not None:
        st.session_state["modified"] = new_survey
        if note:
            st.success(note)

    st.divider()
    if st.button("↺ Reset to interpolated", use_container_width=True):
        st.session_state["modified"] = interp.copy()

# ---------------------------------------------------------------------------
# Tortuosity overlay (live, non-destructive)
# ---------------------------------------------------------------------------

modified = st.session_state["modified"]

with st.sidebar:
    st.header("4 · Output grid")
    regrid = st.checkbox("Resample output to interval", value=True,
                         help=f"Re-grid the survey onto uniform {step:.0f} ft MD stations "
                              "(TD kept). Editing reshapes the station spacing; this lays "
                              "it back on even intervals before tortuosity and export.")
out_base = regrid_survey(modified, step=step, kind=kind) if regrid else modified

display = out_base  # the survey shown / exported (smooth unless tortuosity is on)

with st.sidebar:
    st.header("5 · Tortuosity")
    tort_on = st.checkbox("Add tortuosity", value=False,
                          help="Add seeded inc/azi deviation to mimic a real drilled "
                               "wellbore. Non-destructive overlay — the smooth plan "
                               "above is preserved.")
    if tort_on:
        level = st.slider("Deviation level", 0.0, 5.0, 2.0, 0.5,
                          help=f"0 = smooth; 5 is calibrated so the result reaches "
                               f"~{MAX_DLS_TARGET:.0f}°/100ft max DLS.")
        st.caption("Apply to sections:")
        s1, s2, s3 = st.columns(3)
        dev_vert = s1.checkbox("Vertical", value=False)
        dev_build = s2.checkbox("Build", value=True)
        dev_lat = s3.checkbox("Lateral", value=True)
        seed = int(st.number_input("Seed", 0, 10_000, 42, 1,
                                   help="Same seed reproduces the same wellbore."))
        mask = section_mask(out_base, vertical=dev_vert, build=dev_build, lateral=dev_lat)
        lat_mask = section_mask(out_base, vertical=False, build=False, lateral=True)
        display, dev_info = apply_deviation(
            out_base, level, seed=seed, deviate_mask=mask, lateral_mask=lat_mask
        )
        if dev_info["warning"]:
            st.warning(dev_info["warning"])
        elif level > 0:
            st.caption(f"Max DLS {dev_info['result_max_dls']:.1f}°/100ft · "
                       f"perturbation ±{dev_info['magnitude']:.2f}° (1σ)")

# ---------------------------------------------------------------------------
# Comparison view + download
# ---------------------------------------------------------------------------

label = "Original vs modified" + (" (with tortuosity)" if display is not modified else "")
st.subheader(label)
o, m = summary_dict(interp), summary_dict(display)
cols = st.columns(len(m))
for col, key in zip(cols, m):
    try:
        delta = float(m[key]) - float(o[key])
        col.metric(key, m[key], f"{delta:+.1f}")
    except (ValueError, TypeError):
        col.metric(key, m[key])

left, right = st.columns(2)
left.plotly_chart(plan_view(interp, display), use_container_width=True)
right.plotly_chart(section_view(interp, display), use_container_width=True)
st.plotly_chart(view_3d(interp, display), use_container_width=True)

with st.expander("Modified survey table"):
    st.dataframe(survey_dataframe(display), use_container_width=True, hide_index=True)

st.subheader("6 · Export")
xlsx = write_survey(display, original=interp, summary=summary_dict(display))
st.download_button("⬇️ Download modified survey (.xlsx)", data=xlsx,
                   file_name="modified_survey.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   use_container_width=True)
