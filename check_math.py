"""Quick sanity checks for the survey math and editing engine."""

from __future__ import annotations

import numpy as np

from wellplan.interpolate import interpolate_survey, resample
from wellplan.survey import Survey, minimum_curvature
from wellplan.wellmodel import (
    control_points,
    detect_segments,
    move_kop,
    move_point_to_target,
    set_build_rate,
)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    assert ok, name


# 1. Straight vertical well: N=E=0, TVD=MD.
md = np.array([0.0, 1000.0, 2000.0])
tvd, n, e, dls = minimum_curvature(md, np.zeros(3), np.zeros(3))
check("vertical: TVD==MD", np.allclose(tvd, md))
check("vertical: N==E==0", np.allclose(n, 0) and np.allclose(e, 0))
check("vertical: DLS==0", np.allclose(dls, 0))

# 2. Single build interval vs hand calc. 0->90 deg over 100 ft MD, azi 0.
#    Straight-line build at azi 0: dN = (dMD/2)(sinI1+sinI2)*RF.
tvd, n, e, dls = minimum_curvature([0.0, 100.0], [0.0, 90.0], [0.0, 0.0])
beta = np.pi / 2
rf = (2 / beta) * np.tan(beta / 2)
exp_tvd = 50.0 * (1 + 0) * rf
exp_n = 50.0 * (0 + 1) * rf
check("build: TVD matches RF formula", np.isclose(tvd[-1], exp_tvd), f"{tvd[-1]:.3f} vs {exp_tvd:.3f}")
check("build: N matches RF formula", np.isclose(n[-1], exp_n), f"{n[-1]:.3f} vs {exp_n:.3f}")
check("build: E==0 (azi 0)", np.isclose(e[-1], 0.0))
check("build: DLS==90/100ft", np.isclose(dls[-1], 90.0))

# 3. Azimuth wrap: 350 -> 10 should not spline into a near-360 loop.
md_w = np.array([0.0, 500.0, 1000.0])
inc_w = np.array([30.0, 30.0, 30.0])
azi_w = np.array([350.0, 0.0, 10.0])
gmd, ginc, gazi = resample(md_w, inc_w, azi_w, step=50.0)
# All resampled azimuths should be within [340, 20] (mod), never near 180.
unwrapped = np.mod(gazi + 180, 360) - 180  # center near 0
check("azi wrap: stays near 0 deg", np.all(np.abs(unwrapped) < 25), f"range {gazi.min():.1f}-{gazi.max():.1f}")

# 4. Round-trip interpolation preserves endpoints and is dense.
survey = Survey(
    np.array([0.0, 2000.0, 3125.0, 6125.0]),
    np.array([0.0, 0.0, 90.0, 90.0]),
    np.array([45.0, 45.0, 45.0, 45.0]),
)
dense = interpolate_survey(survey, step=30.0)
check("interp: denser than input", dense.n_stations > survey.n_stations, f"{dense.n_stations} stations")
check("interp: same TD", np.isclose(dense.total_depth, survey.total_depth))

# 5. Segment detection on the build-and-hold profile.
segs = detect_segments(dense)
kinds = [s.kind for s in segs]
check("segments: has vertical/build/lateral", {"vertical", "build"}.issubset(kinds) and "lateral" in kinds, str(kinds))
cp = control_points(dense, segs)
check("control: KOP near 2000", cp.kop_md is not None and abs(cp.kop_md - 2000) < 100, f"KOP={cp.kop_md}")
check("control: heel detected", cp.heel_md is not None, f"heel={cp.heel_md}")

# 6. move_kop shifts the lower well rigidly (TD increases by the KOP delta).
moved = move_kop(dense, 2500.0)
expected_td = dense.total_depth + (2500.0 - cp.kop_md)
check("move_kop: TD shifts by KOP delta", np.isclose(moved.total_depth, expected_td), f"TD={moved.total_depth:.0f} exp={expected_td:.0f}")

# 7. set_build_rate: higher rate -> shorter build, higher DLS.
faster = set_build_rate(dense, [s.kind for s in segs].index("build"), 12.0)
check("set_build_rate: TD reduced", faster.total_depth < dense.total_depth, f"TD={faster.total_depth:.0f}")
check("set_build_rate: DLS increased", faster.max_dls > dense.max_dls - 1e-6, f"DLS={faster.max_dls:.2f}")

# 8. move_point_to_target: move the toe to a new point; must hit it exactly.
tgt = (dense.north[-1] + 500.0, dense.east[-1] + 200.0, dense.tvd[-1] + 50.0)
res = move_point_to_target(dense, ref_md=dense.total_depth, target=tgt, anchor_md=cp.heel_md)
check("move toe: hits target exactly", res.miss_distance < 1e-2, f"hit={res.miss_distance:.4f} ft, maxDLS={res.max_dls:.2f}")

# 9. Move the heel horizontally only (TVD unchanged) -> TVD must be preserved.
heel_idx = int(np.argmin(np.abs(dense.md - cp.heel_md)))
v_heel = dense.tvd[heel_idx]
tgt2 = (dense.north[heel_idx] + 400.0, dense.east[heel_idx], v_heel)  # TVD held
res2 = move_point_to_target(dense, ref_md=cp.heel_md, target=tgt2, anchor_md=cp.kop_md)
check("move heel: TVD preserved when unchanged",
      abs(res2.survey.tvd[heel_idx] - v_heel) < 0.05,
      f"dTVD={res2.survey.tvd[heel_idx] - v_heel:+.4f} ft, hit={res2.miss_distance:.4f}")
# Lateral below the heel is carried along: its own length is preserved (the
# overall TD may grow because bending the KOP->heel curve lengthens it).
lat_before = dense.total_depth - dense.md[heel_idx]
lat_after = res2.survey.total_depth - res2.survey.md[heel_idx]
check("move heel: lateral length kept", abs(lat_after - lat_before) < 1.0,
      f"lateral {lat_after:.0f} vs {lat_before:.0f} ft")

print("\nAll checks passed.")
