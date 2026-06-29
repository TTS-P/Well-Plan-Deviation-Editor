"""Generate a synthetic build-and-hold-plus-lateral survey -> sample_survey.xlsx.

Profile: vertical to KOP @ 2000 ft, build at 8 deg/100ft to 90 deg landing,
hold a short tangent, then a 3000 ft horizontal lateral on a 45 deg azimuth.
"""

from __future__ import annotations

import numpy as np

from wellplan.io_excel import survey_dataframe
from wellplan.survey import Survey

STEP = 30.0
AZI = 45.0
KOP = 2000.0
BUILD_RATE = 8.0  # deg/100ft
LATERAL = 3000.0


def build_profile():
    md, inc, azi = [], [], []

    # Vertical section.
    for m in np.arange(0.0, KOP + STEP, STEP):
        md.append(m); inc.append(0.0); azi.append(AZI)

    # Build section to 90 deg at 8 deg/100ft -> 1125 ft of build.
    build_len = 90.0 / BUILD_RATE * 100.0
    for m in np.arange(STEP, build_len + STEP, STEP):
        md.append(KOP + m)
        inc.append(min(90.0, BUILD_RATE * m / 100.0))
        azi.append(AZI)

    last_md = md[-1]
    # Lateral hold at 90 deg.
    for m in np.arange(STEP, LATERAL + STEP, STEP):
        md.append(last_md + m); inc.append(90.0); azi.append(AZI)

    return np.array(md), np.array(inc), np.array(azi)


def main():
    md, inc, azi = build_profile()
    survey = Survey(md, inc, azi)
    df = survey_dataframe(survey)
    df.to_excel("sample_survey.xlsx", index=False)
    print(f"Wrote sample_survey.xlsx with {len(df)} stations.")
    print(f"  TD={survey.total_depth:.0f} ft  TVD={survey.tvd[-1]:.1f} ft  "
          f"step-out={survey.step_out:.1f} ft  max DLS={survey.max_dls:.2f}")


if __name__ == "__main__":
    main()
