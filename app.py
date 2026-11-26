"""Windmill -- drift-resilient wind turbine condition monitoring, dashboard.

    streamlit run app.py

HYBRID: it works with the raw data and without it.

  LOCAL (data/raw/ present, 19 GB)
      The R80711 monitor loads SCADA straight from the ENGIE release and
      re-runs the detector live, through the same code path the pipeline
      uses. This is the mode for a live demo: the threshold slider re-alarms
      the real series as you move it.

  CLOUD (no data/raw/, Streamlit Community Cloud)
      Everything comes from the committed artefacts in results/. The
      R80711 monitor reads results/demo_r80711.json -- a compact snapshot
      exported by scripts/pipeline/11_demo.py, so the dashboard still has a
      real 10-minute time series with the peer control group, the
      load-conditioned residual and the z-score, without the 19 GB.

Every number shown here is read from results/*.json. The dashboard never
computes a headline figure itself, so it cannot disagree with the README --
which is itself generated from the same files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[0]
RESULTS = ROOT / "results"
DATA_RAW = ROOT / "data" / "raw"

st.set_page_config(
    page_title="Windmill — Drift-Resilient Condition Monitoring",
    page_icon="🌬️",
    layout="wide",
)


# ---------------------------------------------------------------- artefacts
def _load(name: str) -> dict | None:
    """Read one result artefact. Missing files degrade, never crash."""
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_snapshot() -> dict | None:
    return _load("demo_r80711.json")


@st.cache_data(show_spinner=False)
def load_engie() -> dict | None:
    return _load("engie_multivariate.json")


@st.cache_data(show_spinner=False)
def load_multivariate() -> dict | None:
    return _load("multivariate.json")


@st.cache_data(show_spinner=False)
def load_lightgbm() -> dict | None:
    return _load("lightgbm_nbm.json")


@st.cache_data(show_spinner=False)
def load_care_score() -> dict | None:
    return _load("care_score.json")


@st.cache_data(show_spinner=False)
def load_ae_audit() -> dict | None:
    return _load("ae_audit.json")


@st.cache_data(show_spinner=False)
def load_audit() -> dict | None:
    return _load("farm_a_audit.json")


@st.cache_data(show_spinner=False)
def load_thermal() -> dict | None:
    return _load("thermal_signal.json")


@st.cache_data(show_spinner=False)
def load_backtest() -> dict | None:
    return _load("backtest.json")


@st.cache_data(show_spinner=False)
def load_operating_curve() -> dict | None:
    return _load("operating_curve.json")


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_daily_check() -> dict | None:
    return _load("daily_check.json")


def load_limit_vs_model() -> dict | None:
    return _load("limit_vs_model.json")


def _limit_vs_model_view(lv: dict | None) -> None:
    """What the model sees that the fixed limit does not: R80711, day by day.
    Read from results/limit_vs_model.json (scripts/pipeline/24_limit_vs_model.py)."""
    if not lv:
        return
    sp, bf = lv["anomaly_span"], lv["before_span"]
    st.markdown("#### The fixed 80 °C limit and the model, day by day")
    st.markdown(
        f"The fixed limit is part of the model's rule: a day on which the bearing "
        f"held {dck_hot_c():g} °C for an hour is always a hot day. The model adds "
        f"the days that were too hot for their power and weather but stayed under "
        f"the limit. R80711's gearbox bearing ran abnormally hot from August 2017; "
        f"in that period:")
    al = lv["days_flagged"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(str(al["limit"]), "days flagged by the 80 °C limit alone",
            "the fixed setting on its own")
    with c2:
        kpi(f"{al['rule']}", "days flagged by the model's rule",
            f"+{al['rule_increase_over_limit_pct']} % over the limit; every limit day included")
    with c3:
        kpi(str(sp["model_only"]), "of them below the limit",
            f"about {sp['model_only_daily_mean_c_median']:.0f} °C, "
            f"{sp['model_only_excess_k_median']:.1f} K above the healthy prediction")
    with c4:
        kpi(f"{bf['model_hot_days']} vs {al['before_limit']}",
            "hot days in the ten weeks before",
            "model vs the 80 °C limit")
    st.markdown(
        "<div class='note'>On the days below the limit the bearing was too cool "
        "for the limit and too hot for that day's power and weather. The abnormal "
        "period is defined in this project from the sibling turbines; the ENGIE "
        "release has no fault label.</div>", unsafe_allow_html=True)


def dck_hot_c() -> float:
    from windmill import daily_check as dck
    return dck.HOT_C


def load_frozen_fallback() -> dict | None:
    return _load("frozen_fallback.json")


def _frozen_fallback_view(ff: dict | None) -> None:
    """The days only the limit caught, why, and the fix: R80711 after its
    fault, judged with the frozen summer model and with last year's model for
    the same month. Read from results/frozen_fallback.json
    (scripts/pipeline/25_frozen_fallback.py)."""
    if not ff or not ff.get("days"):
        return
    import plotly.graph_objects as go

    tol = ff["tolerance_k"]
    t = ff["turbines"]["R80711"]
    miss = ff["limit_days_missed_by_frozen"]
    caught = set(miss["caught_by_last_year"])
    still = [d for d in miss["days"] if d not in caught]
    trained = t["model_by_month"][next(iter(t["model_by_month"]))]["trained"]
    st.markdown("#### Why the daily average missed limit days, and the fix")
    st.markdown(
        f"The limit rule flags every limit day, but the {tol:g} K daily average "
        f"alone missed {len(miss['days'])} of R80711's. After its fault was "
        f"confirmed, its model stayed frozen on summer data "
        f"({pd.Timestamp(trained[0]):%b}–{pd.Timestamp(trained[1]) - pd.Timedelta(days=1):%b %Y}). "
        f"In summer the turbine rarely runs at high power, so that model could "
        f"not judge the windy winter hours, when the bearing is hottest. The fix: "
        f"while a model is frozen, judge each month with the turbine's own healthy "
        f"model for the same month a year earlier, if it still predicts this "
        f"year's healthy readings to within {ff['gate_k']:g} K.")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi(f"{len(miss['days'])} → {len(still)}",
            "limit days the daily average missed",
            "frozen summer model → last year's same month")
    with c2:
        kpi(f"{t['hot_days']['frozen']} → {t['hot_days']['last_year']}",
            "R80711 hot days, Sep 2017 to Jan 2018",
            f"+{ff['hot_days_increase_pct']} %, of {t['days_judged']} days checked"
            if ff.get("hot_days_increase_pct") is not None else f"of {t['days_judged']} days checked")
    with c3:
        h = ff["healthy"]
        kpi(f"{h['false_hot_days']['frozen']} → {h['false_hot_days']['last_year']}",
            "false hot days on the 3 healthy turbines",
            f"{h['days_judged']} days, each judged with its own last-year models")
    if ff.get("hourly"):
        _frozen_month_view(ff)


def _frozen_month_view(ff: dict) -> None:
    """One month of R80711 after its fault, one point a day: the sensor first,
    then, one tick box each, the frozen summer model and last year's model
    for the same month, each with its tolerance band. The frozen model's
    gaps are the hours it has no healthy reference for."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from windmill import daily_check as dck

    tol = ff["tolerance_k"]
    h = pd.DataFrame(ff["hourly"])
    h["hour"] = pd.to_datetime(h["hour"])
    h["month"] = h["hour"].dt.strftime("%Y-%m")
    months = list(dict.fromkeys(h["month"]))
    label = {m: pd.Timestamp(m + "-01").strftime("%B %Y") for m in months}
    days = pd.DataFrame(ff["days"])
    miss = set(ff["limit_days_missed_by_frozen"]["days"])
    caught = set(ff["limit_days_missed_by_frozen"]["caught_by_last_year"])
    trained = next(iter(ff["turbines"]["R80711"]["model_by_month"].values()))["trained"]

    st.markdown("##### One month, day by day")
    with st.container(key="perf_controls"):
        month = st.select_slider("Month", options=months, value="2017-12" if "2017-12" in months
                                 else months[-1], format_func=label.get, key="perf_ff_month")
        st.markdown("**Show on the chart:** tick a box to compare the sensor with a model")
        c1, c2 = st.columns(2)
        with c1:
            show_frozen = st.checkbox(
                f"The frozen summer model (trained {pd.Timestamp(trained[0]):%b}–"
                f"{pd.Timestamp(trained[1]) - pd.Timedelta(days=1):%b %Y}), ±{tol:g} K",
                value=False, key="perf_show_frozen")
        with c2:
            show_ly = st.checkbox(
                f"Last year's model for {pd.Timestamp(month + '-01'):%B}, ±{tol:g} K",
                value=False, key="perf_show_last_year")

    m = h[h["month"] == month]
    start = pd.Timestamp(month + "-01")
    cal = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")
    # One point per day, as on the Demo page. The sensor is the day's mean
    # over its running hours; each prediction is the day's mean as the daily
    # check judged it. Days not checked are gaps, never joined across.
    sensor = m.groupby(m["hour"].dt.normalize())["actual_c"].mean().reindex(cal)
    dm = days[days["day"].str.startswith(month)].copy()
    dm.index = pd.to_datetime(dm["day"])
    dm = dm.reindex(cal)
    blind = m["frozen_pred_c"].isna().groupby(m["hour"].dt.normalize()).sum().reindex(cal)

    rows = 2 if show_frozen else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.78, 0.22] if rows == 2 else [1.0])
    for d in sorted(x for x in miss if x.startswith(month)):
        fig.add_vrect(x0=pd.Timestamp(d) - pd.Timedelta(hours=12),
                      x1=pd.Timestamp(d) + pd.Timedelta(hours=12),
                      fillcolor="rgba(224,82,99,0.18)", line_width=0, row="all", col=1)
    arms = []
    if show_frozen:
        arms.append(("frozen", "frozen summer model", "#A3ACB9", "rgba(163,172,185,0.18)",
                     "#A3ACB9"))
    if show_ly:
        arms.append(("last_year", "last year's model, same month", C_MODEL,
                     "rgba(0,212,170,0.12)", _daily_view().LIGHT_COLOURS["hot"]))
    for key, name, colour, band, hot_colour in arms:
        pred = dm[f"{key}_pred_c"].dropna()
        fig.add_trace(go.Scatter(x=pred.index, y=pred + tol, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=pred.index, y=pred - tol, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=band, hoverinfo="skip",
                                 name=f"{name}: ±{tol:g} K"), row=1, col=1)
        fig.add_trace(go.Scatter(x=pred.index, y=pred, mode="lines+markers",
                                 name=f"{name}: healthy prediction",
                                 line=dict(width=2.4, dash="dash", color=colour),
                                 marker=dict(size=6)), row=1, col=1)
        hot = dm[dm[f"{key}_excess_k"] > tol]
        if len(hot):
            fig.add_trace(go.Scatter(
                x=hot.index, y=sensor.reindex(hot.index).to_numpy(), mode="markers",
                name=f"hot day under the {name}",
                marker=dict(size=15, symbol="square", color=hot_colour,
                            line=dict(width=1.5, color="#0B1220")),
                customdata=hot[[f"{key}_excess_k"]].to_numpy(),
                hovertemplate="%{customdata[0]:.1f} K above<extra>" + name + "</extra>"),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=sensor.dropna().index, y=sensor.dropna(), mode="lines+markers",
                             name="sensor (daily mean)", line=dict(width=2.6, color=C_ACTUAL),
                             marker=dict(size=7)), row=1, col=1)
    if any(x.startswith(month) for x in miss):
        fig.add_trace(go.Scatter(x=[cal[0]], y=[None], mode="markers",
                                 name="the 80 °C limit fired, the frozen model's average did not",
                                 marker=dict(symbol="square", size=14,
                                             color="rgba(224,82,99,0.5)")), row=1, col=1)
    fig.add_hline(y=dck.HOT_C, line=dict(color=_daily_view().ALARM_COLOURS["real"], width=2,
                                         dash="dash"),
                  annotation_text=f"fixed alarm limit {dck.HOT_C:g} °C",
                  annotation_position="top left", row=1, col=1)
    if show_frozen:
        fig.add_trace(go.Bar(x=cal, y=blind, name="hours the frozen model could not judge",
                             marker_color="#A3ACB9",
                             hovertemplate="%{y} of the day's running hours<extra></extra>"),
                      row=2, col=1)
        fig.update_yaxes(title_text="hours", rangemode="tozero", row=2, col=1)
    vals = [sensor] + [dm[f"{k}_pred_c"] + sgn * tol for k, *_ in arms for sgn in (-1, 1)]
    lo = float(np.nanmin([v.min() for v in vals]))
    hi = float(np.nanmax([v.max() for v in vals] + [dck.HOT_C + 2]))
    fig.update_layout(title=f"R80711 gearbox bearing, {label[month]}",
                      height=680 if rows == 2 else 560, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=150),
                      legend=dict(orientation="h", yref="container", y=0, yanchor="bottom",
                                  x=0, xanchor="left"))
    fig.update_yaxes(title_text="°C", range=[lo - 2, hi + 1], row=1, col=1)
    fig.update_xaxes(range=[cal[0] - pd.Timedelta(hours=12), cal[-1] + pd.Timedelta(hours=12)])
    _chart(fig, width="stretch")

    gap = m["frozen_pred_c"].isna()
    n_gap = int(gap.sum())
    note = ("Red shading: the fixed limit fired and the frozen model's daily average did not. "
            "A day is hot when its average is more than "
            f"{tol:g} K above the prediction.")
    if n_gap:
        note += (f" In {label[month]} the frozen summer model could not judge "
                 f"{n_gap} of {len(m)} running hours (the grey bars): high-power hours it never "
                 f"saw in summer, so those days were judged on their cooler hours only. The bearing averaged {m.loc[gap, 'actual_c'].mean():.1f} °C in "
                 f"those hours and {m.loc[~gap, 'actual_c'].mean():.1f} °C in the rest.")
    n_ly = int(m["last_year_pred_c"].isna().sum())
    note += (f" Last year's model has no prediction for {n_ly} hours." if n_ly
             else " Last year's model predicts every hour.")
    left = sorted(x for x in miss if x.startswith(month) and x not in caught)
    if left:
        note += (" Under the average with either model: "
                 + ", ".join(pd.Timestamp(x).strftime("%d %b") for x in left)
                 + ", hot for only a few hours; the limit rule flags it.")
    st.markdown(f"<div class='note'>{note}</div>", unsafe_allow_html=True)


def _daily_check_view(dcj: dict | None) -> None:
    """The daily check over the whole record, read from results/daily_check.json:
    false call-outs on the healthy siblings and the first call-out on R80711,
    with the choice of the 2 K tolerance folded away."""
    if not dcj or not dcj.get("fleet"):
        return
    two = dcj["fleet"]["2"]
    st.markdown("#### False alarms over the whole record")
    c1, c2 = st.columns(2)
    with c1:
        kpi(f"{two['siblings_false_call_outs_per_year']:.2f}",
            "false call-outs per turbine-year",
            "the three healthy siblings, December 2012 to January 2018")
    with c2:
        first = pd.Timestamp(two["case_first_fault_call_out"]) if two["case_first_fault_call_out"] else None
        kpi(first.strftime("%d %b %Y") if first is not None else "none",
            "R80711: first call-out in the abnormal period",
            "the day the abnormal period starts")
    with st.expander("Why a 2 K tolerance?"):
        rows = []
        for tol in sorted(dcj["fleet"], key=float):
            f = dcj["fleet"][tol]
            rows.append({
                "tolerance": f"{tol} K",
                "healthy siblings, false call-outs / turbine-year":
                    f"{f['siblings_false_call_outs_per_year']:.2f}",
                "R80711, first call-out in the abnormal period":
                    (f["case_first_fault_call_out"] or "none")[:10],
                "abnormal-period days flagged":
                    f"{f['case_fault_days_hot']} of {f['case_fault_days_checked']}",
            })
        st.table(pd.DataFrame(rows))
        st.markdown(
            "<div class='note'>Tighter than 2 K raises false call-outs on the healthy "
            "turbines; looser flags fewer of the abnormal days and calls the fault "
            "later.</div>", unsafe_allow_html=True)


def live_available() -> bool:
    """Is the raw ENGIE data here? (Not just the directory -- the files.)"""
    if not DATA_RAW.exists():
        return False
    return any(DATA_RAW.rglob(f"*{t}*F99*.csv") for t in ("R80711",))


# ---------------------------------------------------------------- helpers
def _fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


#: A wind farm skyline, drawn rather than imported so the dashboard carries
#: no image asset. Each turbine is (hub x, scale, start angle, seconds per
#: revolution): different sizes read as depth, and different start angles
#: stop the rotors looking like copies of one another.
#: Two clusters, one hugging each edge of the window, because the middle of
#: the page is covered by the cards. Each is its own image, drawn at the
#: same uniform scale, so a wide window gets wider gutters rather than
#: stretched or cropped turbines.
_FARM_LEFT = (
    (70, 0.50, 15, 38),
    (215, 0.85, 70, 30),
    (355, 0.60, 40, 44),
)
_FARM_RIGHT = (
    (60, 0.55, 25, 40),
    (225, 1.00, 100, 26),
    (375, 0.60, 55, 46),
)

#: One blade, hub at the origin, pointing up, 118 units long.
_BLADE = "M0 0C-5 -36 -6 -78 0 -118C6 -78 5 -36 0 0Z"

#: True makes the rotors turn slowly. Off by default: motion on a page that
#: people read for minutes at a time is a distraction.
WATERMARK_ANIMATE = True


def _wind_farm_svg(farm=_FARM_RIGHT, animate: bool = False) -> str:
    """A cluster of turbines standing on the ground line, as an SVG string.

    No `preserveAspectRatio="slice"`: it crops whatever does not fit, and on
    a wide window that cropped away the blades and left bare towers.
    """
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 360">'
             '<g fill="#5fd39a">']
    for hx, s, start, secs in farm:
        hy = 150 * s + 20              # hub height grows with the scale
        base = 360 - (1 - s) * 40      # smaller = further away = higher base
        parts.append(
            f'<path d="M{hx - 3 * s:.1f} {hy:.1f}L{hx + 3 * s:.1f} {hy:.1f}'
            f'L{hx + 6 * s:.1f} {base:.1f}L{hx - 6 * s:.1f} {base:.1f}Z"/>')
        spin = (f'<animateTransform attributeName="transform" type="rotate" '
                f'from="{start}" to="{start + 360}" dur="{secs}s" '
                f'repeatCount="indefinite"/>') if animate else ""
        blades = "".join(f'<path d="{_BLADE}" transform="rotate({k * 120})"/>'
                         for k in range(3))
        parts.append(f'<g transform="translate({hx} {hy:.1f}) scale({s})">'
                     f'<g transform="rotate({start})">{spin}{blades}</g>'
                     f'<circle r="8"/></g>')
    parts.append("</g></svg>")
    return "".join(parts)

#: The crisp brand mark beside the sidebar title. Drawn in `currentColor` so
#: it follows the theme in both light and dark mode.
_TURBINE_ICON_SVG = (
    '<svg class="brand-icon" viewBox="0 0 24 24" width="26" height="26"'
    ' fill="none" stroke="currentColor" stroke-width="2.5"'
    ' stroke-linecap="round" aria-hidden="true" focusable="false">'
    # Three straight blades on a hub, over a mast. Drawn as strokes rather
    # than as filled petals: at 26 pixels a filled blade blurs into the
    # others and the whole thing reads as a cross.
    '<path d="M12 8.5V1.6"/>'
    '<path d="M12 8.5L18.6 12.8"/>'
    '<path d="M12 8.5L5.4 12.8"/>'
    '<circle cx="12" cy="8.5" r="1.3" fill="currentColor" stroke="none"/>'
    '<path d="M12 10.2V22.2"/>'
    "</svg>"
)


def _watermark_uri(side: str = "right") -> str:
    """One cluster of the watermark as a data URI, so the CSS needs no
    external file."""
    import base64

    farm = _FARM_LEFT if side == "left" else _FARM_RIGHT
    svg = _wind_farm_svg(farm, WATERMARK_ANIMATE)
    return ("data:image/svg+xml;base64,"
            + base64.b64encode(svg.encode("utf-8")).decode("ascii"))


#: Every font size on the dashboard, in one place.
#:
#: The reviewer asked for text that could be read across a room, so these
#: are roughly double the sizes the readability boost was written at.
#: `FONT_SCALE` is one knob on top of them: raise it to push everything
#: further without hunting through the CSS.
FONT_SCALE = 1.0

FONT_SIZES = {
    "body": 30,        # prose in the tabs and the sidebar
    "caption": 26,     # small explanatory lines under a widget
    "tab": 24,         # the four tab headers: smaller than the prose, so they read as navigation
    "sidebar": 22,     # the narrow sidebar column
    "check": 26,       # checkbox and radio labels
    "widget": 28,      # widget labels: dropdowns, sliders, checkboxes
    "button": 28,      # button captions
    "selectbox": 26,   # the value shown inside a dropdown
    "menu": 26,        # the options in an open dropdown list
    "tag": 22,         # the chips of a multi-select (the chosen turbines)
    "slider": 30,      # the value above a slider's thumb
    "table": 20,       # the reference tables
    "alert": 28,       # st.info / st.success / st.warning / st.error
    "metric_value": 44,
    "metric_label": 24,
    "metric_delta": 22,
    "kpi_value": 46,   # the big number on a KPI card
    "kpi_label": 24,
    "kpi_delta": 22,
    "note": 28,        # the boxed explanatory notes
    "callout": 30,     # the training-window callout
    "brand": 26,       # the sidebar title
    "badge": 18,       # the LIVE / CLOUD badges
    "h2": 44,
    "h3": 38,
    "h4": 34,
}

#: The window width, in px, at which the sizes above apply in full. Narrower
#: than that, the type scales down with the window -- but never below
#: FLUID_FLOOR of its size -- so a smaller window gets smaller text instead
#: of lines that run off the edge and are cropped.
FLUID_REF_PX = 1400
FLUID_FLOOR = 0.55


def _fluid(px: float) -> str:
    """A CSS font size: `px` on a wide window, shrinking with a narrow one."""
    size = px * FONT_SCALE
    floor = max(11, round(size * FLUID_FLOOR))
    return f"clamp({floor}px, {size / FLUID_REF_PX * 100:.3f}vw, {round(size)}px)"


def _style():
    css = """
        <style>
        /* ------------------------------------------------------------------
           Typography. Every rule here is explicit px rather than rem, so
           the sizes below are the sizes that render -- Streamlit's own
           rem-based defaults are simply overruled.
           ------------------------------------------------------------------ */

        /* Prose: the paragraphs in the tabs. */
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li {
            font-size: __body__px !important;
            line-height: 1.55 !important;
        }

        /* The sidebar is a 300px column, so its prose is a size smaller --
           at the body size it wrapped into four lines a sentence. */
        .stApp [data-testid="stSidebar"] p,
        .stApp [data-testid="stSidebar"] li {
            font-size: __sidebar__px !important;
            line-height: 1.5 !important;
        }

        /* Captions: the small lines that explain a control. */
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p {
            font-size: __caption__px !important;
            line-height: 1.45 !important;
            color: #cbd5e1 !important;
        }

        /* Headings. */
        .stApp [data-testid="stMarkdownContainer"] h1 { font-size: __h2__px !important; }
        .stApp [data-testid="stMarkdownContainer"] h2 { font-size: __h2__px !important; }
        .stApp [data-testid="stMarkdownContainer"] h3 { font-size: __h3__px !important; }
        .stApp [data-testid="stMarkdownContainer"] h4 { font-size: __h4__px !important; }

        /* The four tab headers. They must read as NAVIGATION, not as body
           text, so they are set smaller than the prose, in capitals with
           letter-spacing, and held in their own bar (see "Tabs." below).
           A label can still wrap onto two lines, so the bar is given room. */
        .stApp [data-testid="stTab"] p {
            font-size: __tab__px !important;
            font-weight: 650 !important;
            letter-spacing: .07em !important;
            text-transform: uppercase !important;
            white-space: normal !important;
            line-height: 1.25 !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
            color: inherit !important;
        }
        .stApp [data-testid="stTabs"] [role="tablist"] {
            height: auto !important;
            min-height: 0 !important;
        }
        .stApp [data-testid="stTab"] {
            height: auto !important;
        }

        /* Widget labels: dropdowns, sliders, checkboxes, radio buttons. */
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stWidgetLabel"] span {
            font-size: __widget__px !important;
        }

        /* Checkbox and radio labels sit in narrow columns and are long
           ("Frozen 2 years before (2015 curve + 2017 weather)"). They wrap
           rather than being cut off with an ellipsis. */
        .stApp [data-testid="stCheckbox"] label p,
        .stApp [data-testid="stRadio"] label p {
            font-size: __check__px !important;
            white-space: normal !important;
        }

        /* The Demo's two chart switches: a highlighted panel with large tick
           boxes, so a viewer sees at once where to click. The panel is
           st.container(key="demo_controls"), which Streamlit renders with
           the class st-key-demo_controls. */
        .st-key-demo_controls, .st-key-perf_controls {
            border: 3px solid #00D4AA;
            border-radius: 14px;
            padding: 18px 24px 12px 24px;
            background: rgba(0, 212, 170, 0.10);
        }
        /* The tick box itself, on every checkbox in the dashboard. Streamlit
           draws it as the first plain <div> inside the checkbox label, 16 px
           and dark on dark by default. Here it is 56 px and light, so it
           stands out on the black page; ticked, it fills teal with a large
           dark tick. */
        .stApp [data-testid="stCheckbox"] label {
            align-items: center !important;
            gap: 16px;
        }
        .stApp [data-testid="stCheckbox"] label > div:not([data-testid]) {
            width: 56px !important;
            height: 56px !important;
            min-width: 56px !important;
            border: 4px solid #FFFFFF !important;
            border-radius: 10px !important;
            background-color: #F4F6F8 !important;
            display: flex !important;
            align-items: center;
            justify-content: center;
        }
        .stApp [data-testid="stCheckbox"] label:has(input:checked) > div:not([data-testid]) {
            background-color: #00D4AA !important;
            border-color: #FFFFFF !important;
        }
        .stApp [data-testid="stCheckbox"] label > div:not([data-testid]) svg {
            width: 70% !important;
            height: 70% !important;
            stroke: #0B1220 !important;
            stroke-width: 2.4 !important;
            fill: none !important;
        }
        .st-key-demo_controls [data-testid="stCheckbox"] label p,
        .st-key-perf_controls [data-testid="stCheckbox"] label p {
            font-weight: 600;
        }

        /* The value shown inside a dropdown. It is an <input>, which CSS
           can still size. */
        .stApp [data-testid="stSelectbox"] input,
        .stApp [data-testid="stSelectbox"] div[role="combobox"] {
            font-size: __selectbox__px !important;
        }

        /* The open dropdown list. Streamlit draws it in a popover outside
           the widget, so it is sized here on its own, or it falls back to
           the 14px default. */
        /* Every element of the list is sized, not only the <li>: the
           option text sits in nested elements that carry their own size. */
        [role="listbox"], [role="listbox"] *,
        [role="option"], [role="option"] *,
        [data-testid="stSelectboxVirtualDropdown"],
        [data-testid="stSelectboxVirtualDropdown"] *,
        [data-baseweb="menu"], [data-baseweb="menu"] *,
        [data-baseweb="popover"] ul, [data-baseweb="popover"] ul * {
            font-size: __menu__px !important;
            line-height: 1.25 !important;
        }

        /* A multi-select: its chosen chips, and what is typed into it. */
        .stApp [data-testid="stMultiSelect"] [data-baseweb="tag"],
        .stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] span {
            font-size: __tag__px !important;
            height: auto !important; max-width: none !important;
            padding-top: .15rem; padding-bottom: .15rem;
        }
        .stApp [data-testid="stMultiSelect"] input {
            font-size: __selectbox__px !important;
        }

        /* A slider's read-out floats above its thumb on an offset
           Streamlit computed for a 14px font, so it has to be sized and
           placed together or it lands on the track. */
        .stApp [data-testid="stSliderThumbValue"],
        .stApp [data-testid="stSliderThumbValue"] p {
            font-size: __slider__px !important;
        }
        .stApp [data-testid="stSliderThumbValue"] {
            top: -1.55em !important;
        }
        /* ...and the room between the label and the track for it. */
        .stApp [data-testid="stSlider"] [data-testid="stWidgetLabel"] {
            margin-bottom: calc(__slider__px * 1.6) !important;
        }
        .stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],
        .stApp [data-testid="stSlider"] [data-testid="stTickBarMax"] {
            font-size: __slider__px !important;
        }

        /* Buttons. The tab rule above is more specific and wins for tabs. */
        .stApp [data-testid="stButton"] button p,
        .stApp [data-testid="stButton"] button span {
            font-size: __button__px !important;
        }

        /* The alerts: st.info, st.success, st.warning, st.error. */
        .stApp [data-testid="stAlert"],
        .stApp [data-testid="stAlert"] div,
        .stApp [data-testid="stAlert"] p {
            font-size: __alert__px !important;
            line-height: 1.5 !important;
        }

        /* st.metric. */
        .stApp [data-testid="stMetricValue"] { font-size: __metric_value__px !important; }
        .stApp [data-testid="stMetricLabel"] { font-size: __metric_label__px !important; }
        .stApp [data-testid="stMetricDelta"] { font-size: __metric_delta__px !important; }

        /* The reference tables. Streamlit draws a dataframe on a canvas,
           where CSS cannot reach, so these are st.table -- real HTML. */
        .stApp [data-testid="stTable"] table,
        .stApp [data-testid="stTable"] th,
        .stApp [data-testid="stTable"] td {
            font-size: __table__px !important;
        }
        .stApp [data-testid="stTable"] th {
            text-align: right !important;
        }
        /* A table wider than its column scrolls, rather than running over
           the text beside it. */
        .stApp [data-testid="stTable"] {
            overflow-x: auto !important;
        }
        .stApp [data-testid="stTable"] table {
            width: max-content !important;
            min-width: 100% !important;
        }
        .stApp [data-testid="stTable"] th {
            white-space: normal !important;
        }

        /* The text inside the charts is sized on the FIGURE, not here:
           Plotly lays a chart out for the font it is given, so overriding
           it from CSS afterwards would leave the layout computed for a
           smaller font and the legends would run over each other. */

        /* ------------------------------------------------------------------
           Nothing is cut off. Streamlit puts some text on one line with an
           ellipsis by default, and at these sizes that crops it. Here it
           wraps instead, and the sizes above shrink with the window (see
           _fluid) rather than staying fixed.
           ------------------------------------------------------------------ */

        /* st.metric truncates its label, value and delta with an ellipsis. */
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] *,
        .stApp [data-testid="stMetricDelta"],
        .stApp [data-testid="stMetricDelta"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
        }

        /* A long unbroken string wraps instead of running past the edge. */
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stAlert"],
        .kpi, .note, .callout, .badge {
            overflow-wrap: anywhere;
        }

        /* Code blocks wrap rather than scroll sideways. */
        .stApp [data-testid="stCode"] pre,
        .stApp [data-testid="stCode"] code {
            white-space: pre-wrap !important;
            overflow-wrap: anywhere !important;
        }

        /* Columns stack as the window narrows. Streamlit only stacks them at
           640px, which is far too late for text this size. */
        @media (max-width: 1000px) {
            .stApp [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(50% - 1rem) !important;
                min-width: calc(50% - 1rem) !important;
            }
        }
        @media (max-width: 700px) {
            .stApp [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(100% - 1rem) !important;
                min-width: calc(100% - 1rem) !important;
            }
        }

        /* A chart's title is a heading above it, not text inside it: Plotly
           cannot wrap a title, so a long one was cut off by a narrow window,
           whereas a heading wraps with the page. */
        .chart-title {
            font-size: __h4__px !important;
            font-weight: 600;
            line-height: 1.3;
            margin: 10px 0 4px 0;
        }

        /* ------------------------------------------------------------------
           Design system: one palette, one radius, one spacing rhythm.
           (The old rules used --border, --foreground and --muted-foreground,
           which nothing defined, so the cards were drawn with no border.)
           ------------------------------------------------------------------ */
        .stApp {
            --accent: #5fd39a;
            --accent-soft: rgba(95, 211, 154, 0.10);
            --surface: #161b22;
            --surface-2: #1c2330;
            --border: rgba(240, 246, 252, 0.12);
            --foreground: #e6edf3;
            --muted-foreground: #9aa7b6;
            --radius: 12px;
        }

        /* The page: a centred column, not a stretch across a wide monitor. */
        [data-testid="stMainBlockContainer"] {
            max-width: 1680px !important;
            margin: 0 auto !important;
            padding: 2rem 2.5rem 4rem 2.5rem !important;
        }
        footer { visibility: hidden; }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stSidebar"] { border-right: 1px solid var(--border); }
        .stApp [data-testid="stHorizontalBlock"] { gap: 1.25rem !important; }

        /* Page header. */
        .app-header {display: flex; flex-wrap: wrap; align-items: center; gap: 24px;
                     padding: 26px 30px; margin-bottom: 18px;
                     border: 1px solid var(--border); border-radius: 16px;
                     background: linear-gradient(120deg, rgba(95, 211, 154, 0.12),
                                 rgba(95, 211, 154, 0) 55%), var(--surface);}
        .app-mark {flex: 0 0 auto; width: 64px; height: 64px; display: flex;
                   align-items: center; justify-content: center;
                   border-radius: 16px; color: var(--accent);
                   background: var(--accent-soft);
                   border: 1px solid rgba(95, 211, 154, 0.30);}
        .app-mark .brand-icon {width: 38px; height: 38px; color: var(--accent);}
        .app-title {font-size: __h2__px !important; font-weight: 700;
                    line-height: 1.15; letter-spacing: -0.01em;}
        .app-sub {font-size: __caption__px !important; color: var(--muted-foreground);
                  margin-top: 6px; line-height: 1.4;}
        .app-text {flex: 1 1 320px; min-width: 0;}
        .chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;}
        .chip {padding: 4px 12px; border-radius: 999px; border: 1px solid var(--border);
               background: rgba(255, 255, 255, 0.04); color: var(--muted-foreground);
               font-size: __badge__px !important; font-weight: 600; letter-spacing: .03em;}
        .app-link {flex: 0 0 auto; padding: 10px 20px; border-radius: 10px;
                   border: 1px solid rgba(95, 211, 154, 0.45);
                   color: var(--accent) !important; font-weight: 600;
                   text-decoration: none !important; font-size: __caption__px !important;}
        .app-link:hover {background: var(--accent-soft);}
        .eyebrow {font-size: __kpi_label__px !important; font-weight: 700;
                  color: var(--accent); text-transform: uppercase;
                  letter-spacing: .12em; margin: 8px 0 10px 2px;}
        .app-footer {margin-top: 3rem; padding-top: 18px;
                     border-top: 1px solid var(--border);
                     color: var(--muted-foreground);
                     font-size: __caption__px !important; line-height: 1.6;}
        .app-footer a {color: var(--accent) !important; text-decoration: none;}

        /* Tabs: a contained, segmented bar that is visibly not the body
           text. A panel with its own background and border; every tab the
           same width; the active tab a filled accent pill with an accent
           underline. Streamlit's own thin highlight and border are hidden
           so there is exactly one indicator, not two. */
        .stApp [data-testid="stTabs"] {margin-top: 1.4rem;}
        .stApp [data-baseweb="tab-list"] {
            gap: 6px !important;
            padding: 6px !important;
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        }
        .stApp [data-baseweb="tab-highlight"],
        .stApp [data-baseweb="tab-border"] {display: none !important;}
        .stApp [data-testid="stTab"] {
            flex: 1 1 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 14px 16px !important;
            border-radius: 10px !important;
            border: 1px solid transparent !important;
            color: var(--muted-foreground) !important;
            transition: background .15s ease, color .15s ease,
                        border-color .15s ease;
        }
        .stApp [data-testid="stTab"]:hover {
            background: rgba(255, 255, 255, 0.05) !important;
            color: #e6edf3 !important;
        }
        .stApp [data-testid="stTab"][aria-selected="true"],
        .stApp [data-baseweb="tab"][aria-selected="true"],
        .stApp [role="tab"][aria-selected="true"] {
            background: rgba(95, 211, 154, 0.16) !important;
            border-color: rgba(95, 211, 154, 0.55) !important;
            box-shadow: inset 0 -3px 0 var(--accent);
            color: var(--accent) !important;
        }
        .stApp [data-baseweb="tab-panel"] {padding-top: 1.6rem !important;}

        /* Headings: a clear hierarchy, with the section titles marked. */
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3 {
            font-weight: 700 !important; letter-spacing: -0.01em;
            line-height: 1.2 !important; padding-top: .4rem !important;
        }
        .stApp [data-testid="stMarkdownContainer"] h4 {
            font-weight: 600 !important; line-height: 1.25 !important;
            border-left: 4px solid var(--accent);
            padding: 2px 0 2px 14px !important;
            margin: 1.4rem 0 .6rem 0 !important;
        }

        /* Controls. */
        .stApp [data-baseweb="select"] > div {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important; min-height: 3.1rem;
        }
        .stApp [data-baseweb="select"] > div:hover {
            border-color: var(--accent) !important;
        }
        .stApp [data-testid="stButton"] button,
        .stApp [data-testid="stDownloadButton"] button {
            border-radius: 10px; padding: .55rem 1.4rem; font-weight: 600;
        }
        .stApp [data-testid="stFileUploader"] section {
            border-radius: var(--radius); background: var(--surface);
            border: 1px dashed var(--border);
        }
        .stApp [data-testid="stAlert"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border);
        }

        /* Charts, metrics and tables sit on cards. */
        .stApp [data-testid="stPlotlyChart"] {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); overflow: hidden;
        }
        .stApp [data-testid="stMetric"] {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 14px 20px;
        }
        .stApp [data-testid="stTable"] {
            border: 1px solid var(--border); border-radius: var(--radius);
            background: var(--surface);
        }
        .stApp [data-testid="stTable"] th { background: var(--surface-2); }

        /* Custom components */
        .badge {padding: 6px 14px; border-radius: 999px; font-weight: 700;
                letter-spacing: .05em; font-size: __badge__px !important; display: inline-block;}
        .live {background:#0f3d2e; color:#5fd39a; border:1px solid #1d6b4e;}
        .cloud {background:#2b2f4a; color:#9fb4ff; border:1px solid #455089;}
        .kpi {padding: 18px 22px; border: 1px solid var(--border);
              border-top: 3px solid var(--accent); border-radius: var(--radius);
              background: var(--surface); margin-bottom: 10px;
              box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);}
        .kpi .v {font-size: __kpi_value__px !important; font-weight: 700; line-height: 1.15;}
        .kpi .l {font-size: __kpi_label__px !important; font-weight: 600;
                 color: var(--muted-foreground); text-transform: uppercase;
                 letter-spacing: .06em; margin-top: 6px;}
        /* The delta carries an inline font-size, so this has to win. */
        .kpi div[style*="font-size"] { font-size: __kpi_delta__px !important; }
        .kpi.status-monitoring {border-left: 5px solid #2EAD6B;}
        .kpi.status-frozen {border-left: 5px solid #E05263;}
        .kpi.status-hold {border-left: 5px solid #8B7CF6;}
        .kpi.status-none {border-left: 5px solid #6B7280;}
        .hold-bar {height: 8px; border-radius: 4px; margin: 8px 0;
                   background: rgba(139,124,246,0.18);}
        .hold-bar > div {height: 8px; border-radius: 4px; background: #8B7CF6;}
        .note {border: 1px solid var(--border); border-left: 4px solid var(--accent);
               background: var(--surface); border-radius: var(--radius);
               padding: 14px 20px; color: #c9d4df; margin: 6px 0 10px 0;
               font-size: __note__px !important; line-height: 1.55;}

        .callout {border: 1px solid rgba(95, 211, 154, 0.30);
                  border-left: 4px solid var(--accent);
                  padding: 14px 20px; margin: 12px 0 18px 0;
                  background: var(--accent-soft);
                  border-radius: var(--radius); font-size: __callout__px !important; line-height: 1.55;}
        .brand {display: flex; align-items: center; gap: 10px;
                font-size: __brand__px !important; font-weight: 700; line-height: 1.25;
                margin: 14px 0 6px 0; color: var(--foreground);}
        .brand-icon {flex: 0 0 auto; width: 28px; height: 28px; color: var(--foreground);}

        /* Wind farm skyline along the foot of the page. Fixed, so it stays
           put as the page scrolls; kept to the two gutters beside the
           content column; and it ignores the pointer. */
        body::after {
            content: "";
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            height: min(clamp(260px, 32vw, 620px), 65vh);
            background-image: url("__WATERMARK_LEFT__"), url("__WATERMARK__");
            background-repeat: no-repeat;
            background-position: left bottom, right bottom;
            background-size: auto 100%, auto 100%;
            opacity: 0.14;
            pointer-events: none;
            z-index: 0;
        }
        section.main, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            position: relative;
            z-index: 1;
        }
        </style>
    """
    for key, value in FONT_SIZES.items():
        css = css.replace(f"__{key}__px", _fluid(value))
    css = css.replace("__WATERMARK_LEFT__", _watermark_uri("left"))
    css = css.replace("__WATERMARK__", _watermark_uri("right"))
    st.markdown(css, unsafe_allow_html=True)


#: The size of the text inside a chart. Plotly lays a chart out for the
#: font it is given, so this has to be set on the figure: overriding it
#: from CSS afterwards would leave the layout computed for a smaller font,
#: and the legends would run over each other.
CHART_FONT_PX = 26

#: A chart legend is a secondary element. Its entry names are long, so it is
#: a size smaller than the chart. Panel titles and threshold labels are the
#: same size: they sit on one line above a panel and have to fit its width.
CHART_LEGEND_FONT_PX = 22
#: The alarm count right of each row of the Explore strip: one line, so it
#: fits the row height.
STRIP_COUNT_FONT_PX = 18
CHART_ANNOTATION_FONT_PX = 22

#: Gridlines and axis lines: present enough to read values against, quiet
#: enough not to compete with the data.
CHART_GRID = "rgba(148, 163, 184, 0.14)"


def _chart(fig, **kwargs):
    """Render a Plotly figure with the dashboard's chart typography.

    The sizes have to be set on the figure, and on the axes as well as the
    layout: Streamlit applies its own theme to the figure at render time,
    and that theme sets an explicit tick font, which would otherwise win.
    """
    import html

    # The title becomes a heading above the chart (see .chart-title).
    title = fig.layout.title.text
    if title:
        st.markdown(f"<div class='chart-title'>{html.escape(title)}</div>",
                    unsafe_allow_html=True)
    fig.update_layout(title=dict(text=None))
    # Panel titles of a subplot figure still sit in the top margin; without
    # them the margin the title used to need can go.
    has_panel_titles = any(a.yref == "paper" and not a.showarrow
                           for a in fig.layout.annotations)
    if title and not has_panel_titles:
        fig.update_layout(margin=dict(t=16))

    fig.update_layout(
        font=dict(size=CHART_FONT_PX),
        legend=dict(font=dict(size=CHART_LEGEND_FONT_PX)),
    )
    # Breathing room at the edges, so the last tick label and the right-hand
    # end of an annotation are never cut off by the card.
    m = fig.layout.margin
    fig.update_layout(margin=dict(
        l=max(m.l if m.l is not None else 80, 24),
        r=max(m.r if m.r is not None else 80, 30)))
    # automargin lets the margins grow to fit the tick labels and axis
    # titles, instead of cutting them off at a fixed margin.
    fig.update_xaxes(tickfont=dict(size=CHART_FONT_PX),
                     title_font=dict(size=CHART_FONT_PX), automargin=True,
                     gridcolor=CHART_GRID, linecolor=CHART_GRID)
    fig.update_yaxes(tickfont=dict(size=CHART_FONT_PX),
                     title_font=dict(size=CHART_FONT_PX), automargin=True,
                     gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
                     linecolor=CHART_GRID)
    fig.update_annotations(font=dict(size=CHART_ANNOTATION_FONT_PX))
    st.plotly_chart(fig, **kwargs)


def kpi(value: str, label: str, delta: str | None = None):
    d = f"<div style='font-size:.8rem;color:var(--muted-foreground)'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='kpi'><div class='v'>{value}</div>"
        f"<div class='l'>{label}</div>{d}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- tab 1
def _model_comparison(engie, care_score, multiv, lgbm) -> None:
    """The model comparison: the deterministic arm against the learned one.

    This lives in the benchmark tab because it is a TECHNICAL study -- which
    of two detectors is better, and on what. The operational story is on the
    first two tabs.
    """
    st.markdown(
        "A condition-monitoring baseline calibrated on one period reads slow drift as "
        "degradation. On healthy turbines at the ENGIE site, **every alarm is a false "
        "alarm** — so the false-alarm rate is the cleanest possible measurement of "
        "whether a detector is fit to operate."
    )

    if not engie:
        st.error("results/engie_multivariate.json is missing.")
        return
    s = engie["strategies"]
    det_frozen = s.get("frozen_2014", {}).get("mean_rate_pct")
    det_roll = s.get("rolling_30d", {}).get("mean_rate_pct")
    lg_frozen = s.get("frozen_2014_lightgbm", {}).get("mean_rate_pct")
    lg_roll = s.get("rolling_30d_lightgbm", {}).get("mean_rate_pct")
    ratio = (det_frozen / det_roll) if det_frozen and det_roll else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(f"{_fmt(ratio, 1)}×", "False-alarm reduction", "changing only the reference")
    with c2:
        kpi(f"{_fmt(det_roll, 2)} %", "Rolling 30-day reference", "mean false-alarm rate")
    with c3:
        kpi(f"{_fmt(det_frozen, 2)} %", "Frozen 2014 reference", "mean false-alarm rate")
    with c4:
        if care_score:
            kpi(_fmt(care_score["care_score"]), "CARE score", "95 labelled events, leak-free")
        else:
            kpi("—", "CARE score")

    st.markdown("")

    # --- the chart ---------------------------------------------------------
    import plotly.graph_objects as go

    rows = []
    for label, key in (("Deterministic (bins)", "frozen_2014"),
                        ("Deterministic (bins)", "rolling_30d"),
                        ("Learned (LightGBM NBM)", "frozen_2014_lightgbm"),
                        ("Learned (LightGBM NBM)", "rolling_30d_lightgbm")):
        v = s.get(key)
        if not v:
            continue
        rows.append({
            "detector": label,
            "reference": "frozen 2014" if "frozen" in key else "rolling 30-day",
            "mean %": v["mean_rate_pct"],
            "min %": v["min_rate_pct"],
            "max %": v["max_rate_pct"],
            "per_year": v["per_year"],
        })
    if not rows:
        st.warning("no ENGIE strategies in the artefact.")
        return
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for detector, g in df.groupby("detector"):
        g = g.sort_values("mean %")
        fig.add_trace(go.Bar(
            x=g["mean %"], y=g["reference"], orientation="h",
            name=detector, text=[f"{v:.2f} %" for v in g["mean %"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f} % mean false alarms<extra>%{fullData.name}</extra>",
        ))
    # Headroom on the x-axis so the outside text labels are not clipped, and
    # a horizontal legend so the chart keeps its width.
    xmax = max((r["max %"] for r in rows), default=1.0)
    fig.update_layout(
        title="Mean false-alarm rate on healthy turbines",
        xaxis_title="false alarms (% of 10-minute samples)",
        xaxis_range=[0, xmax * 1.3],
        yaxis_title=None, barmode="group", height=380,
        # The legend runs along the top and wraps onto more rows as the
        # window narrows. A legend down the right-hand side took a fixed
        # share of the width and squeezed the bars to almost nothing.
        legend=dict(title=dict(text=""), orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=40, t=16, b=70),
    )
    # The artefact stores the year as an integer INSIDE each entry; the
    # JSON object key is a string, and Plotly coerces a numeric-looking
    # string column to float -- which is how the axis ended up showing
    # "2,013.5" instead of "2013".
    per_year_df = pd.DataFrame([
        {"year": v["year"], "detector": r["detector"],
         "reference": r["reference"], "rate %": v["rate_pct"]}
        for r in rows
        for v in r["per_year"].values()
    ])

    left, right = st.columns([3, 2])
    with left:
        _chart(fig, width="stretch")
    with right:
        st.markdown(
            "<div class='note'>Both detectors improve when the reference is "
            "re-estimated. The deterministic one improves further, because its "
            "reference is a direct statistic of the operating point rather than "
            "a fitted model that must itself be refreshed.</div>",
            unsafe_allow_html=True)

    # The per-year table has five numeric columns under a two-row header.
    # It needs the full width of the page, and st.table rather than
    # st.dataframe: a dataframe is drawn on a canvas, where CSS cannot
    # reach, and this is a fixed reference table nobody needs to sort.
    st.markdown("#### Per year")
    st.table(
        per_year_df.pivot_table(index="year", columns=["detector", "reference"],
                                values="rate %").round(3))

    # --- the same thing as a line: year by year ---------------------------
    # The bar chart shows the means; this shows WHY the means differ. A frozen
    # reference swings from year to year, because each year the turbine has
    # drifted a little further from the reference. A rolling reference is flat,
    # because it never gets the chance to go stale.
    fig2 = go.Figure()
    for (detector, reference), g in per_year_df.groupby(["detector", "reference"]):
        g = g.sort_values("year")
        fig2.add_trace(go.Scatter(
            x=g["year"], y=g["rate %"], mode="lines+markers",
            name=f"{detector} — {reference}",
            connectgaps=True,
            hovertemplate="%{x}: %{y:.2f} % false alarms<extra>%{fullData.name}</extra>",
        ))
    fig2.update_layout(
        title="False-alarm rate by year — the frozen reference swings, the rolling one does not",
        xaxis_title="year", yaxis_title="false alarms (% of 10-minute samples)",
        height=520, margin=dict(l=10, r=40, t=16, b=70),
        # Four entries with long names: along the top, wrapping onto more
        # rows as the window narrows, rather than down the side, where they
        # took a fixed share of the width and squeezed the plot.
        legend=dict(title=dict(text=""), orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
        xaxis=dict(tickformat="d", dtick=1),
    )
    st.markdown(
        "<div class='note'>2014 has no scored samples in this artefact and is "
        "absent from the table above; the lines span it.</div>",
        unsafe_allow_html=True)
    _chart(fig2, width="stretch")


# ------------------------------------------------- the live explorer
# The engine itself lives in src/windmill/live.py (repo rule 2: domain logic
# in src/, orchestration in the app). These wrappers only add caching, so
# that re-running the explorer with the same inputs is instant.

def _ensure_src() -> None:
    """Put the project's src/ on the import path.

    The dashboard imports the project's own modules -- the live engine, the
    forecast module -- and they are not installed, they are in the repo. So
    the path is set once, here, rather than repeated at every import site.
    """
    import sys
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _engine():
    """Import the live engine."""
    _ensure_src()
    from windmill import live
    return live


def _forecast_module():
    """Import the forecasting module."""
    _ensure_src()
    from windmill import forecast
    return forecast


@st.cache_data(show_spinner=False)
def _run_engie_cached(turbine: str, window: str, z: float, peers: bool):
    live = _engine()
    return live.run_engie(ROOT / "data", turbine=turbine, window=window,
                          z_threshold=z, with_peers=peers)


@st.cache_data(show_spinner=False)
def _run_care_cached(farm: str, event_id: str, quantile: float):
    live = _engine()
    return live.run_care(ROOT / "data", farm=farm, event_id=event_id,
                         quantile=quantile)


@st.cache_data(show_spinner=False)
def _parse_upload_cached(payload: bytes):
    """Parse an uploaded CSV and infer what its columns are."""
    import io
    import pandas as pd
    live = _engine()
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    return df, live.infer_columns(df)


@st.cache_data(show_spinner=False)
def _run_upload_cached(payload: bytes, channel: str, z: float, ref_frac: float):
    import io
    import pandas as pd
    live = _engine()
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    return live.run_upload(df, z_threshold=z, ref_frac=ref_frac,
                           display_channel=channel or None)


def _provenance_banner(prov: dict) -> None:
    """The banner that says what the engine just did.

    Rows processed, runtime, and the datasets that were joined -- because
    "SCADA plus reanalysis weather" and "SCADA alone" are different
    measurements, and a reader has to be able to tell which one they are
    looking at.
    """
    joined = " · ".join(prov.get("joined", []))
    st.success(
        f"**{prov.get('rows', 0):,} rows** processed in "
        f"**{prov.get('seconds', 0.0):.2f} s** — joined: {joined}"
    )


def _render_live_result(result: dict, z_threshold: float) -> None:
    """Draw whatever the engine produced, whatever the source was.

    The three sources share one renderer, so the reader learns one layout:
    the channel, the detector's score, and the alarms.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    frame = result["frame"]
    prov = result["provenance"]

    has_value = "value" in frame.columns and frame["value"].notna().any()
    rows = 2 if has_value else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4] if has_value else [1.0],
        vertical_spacing=0.06,
        subplot_titles=(("Component temperature (°C)",) if has_value else ())
        + (("Detector z-score and alarms",) if has_value else ()),
    )

    if has_value:
        fig.add_trace(go.Scatter(
            x=frame.get("ts"), y=frame["value"], name="channel",
            mode="lines", line=dict(width=1.6)), row=1, col=1)
        if "peer" in frame.columns and frame["peer"].notna().any():
            fig.add_trace(go.Scatter(
                x=frame.get("ts"), y=frame["peer"], name="control group (median)",
                mode="lines", line=dict(width=2.0, dash="dash"),
                opacity=0.9), row=1, col=1)
        for col in frame.columns:
            if col.startswith("peer_") and frame[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=frame.get("ts"), y=frame[col], name=col.replace("peer_", ""),
                    mode="lines", line=dict(width=1.0), opacity=0.7), row=1, col=1)
        fig.update_yaxes(title_text="°C", row=1, col=1)

    zrow = 2 if has_value else 1
    for col, style in (("z_rolling", dict(width=1.4)),
                       ("z_frozen", dict(width=1.1, dash="dot"))):
        if col in frame.columns and frame[col].notna().any():
            fig.add_trace(go.Scatter(
                x=frame.get("ts"), y=frame[col],
                name=col.replace("_", " "), mode="lines",
                line=style), row=zrow, col=1)

    # Which reference the alarms are measured against. Both z-scores are
    # already in the frame, so this is the detector's own decision rule
    # applied to the same window -- not a pre-baked annotation. The rolling
    # reference is the default, because it is the one that operates.
    has_rolling = ("z_rolling" in frame.columns
                   and frame["z_rolling"].notna().any())
    has_frozen = ("z_frozen" in frame.columns
                  and frame["z_frozen"].notna().any())
    choices = []
    if has_rolling:
        choices.append("rolling 30-day")
    if has_frozen:
        choices.append("frozen 2014")
    if choices:
        reference = st.radio(
            "Reference", choices, index=0, key="live_reference",
            help="A frozen reference applied to later data reads the "
                 "turbine's own drift as overheating. The rolling reference "
                 "is the one that operates.",
        )
    else:
        reference = "the run's own reference"
    shown = {"rolling 30-day": "z_rolling",
             "frozen 2014": "z_frozen"}.get(reference, "z_frozen")
    if shown not in frame.columns:
        shown = "z_frozen" if "z_frozen" in frame.columns else "z_rolling"
    if shown in frame.columns:
        alarm = frame[shown] > z_threshold
        fig.add_trace(go.Scatter(
            x=frame.get("ts"), y=frame[shown].where(alarm),
            name=f"alarm (z > {z_threshold:.1f})", mode="markers",
            marker=dict(size=4, color="#e05263")), row=zrow, col=1)
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="#e05263",
                  row=zrow, col=1, annotation_text=f"threshold {z_threshold:.1f}")

    # The ground-truth event window, where the source has one.
    if "in_event" in frame.columns and frame["in_event"].any():
        xs = frame.get("ts")
        if xs is not None:
            flagged = frame[frame["in_event"].fillna(False)]
            if len(flagged):
                x0, x1 = flagged["ts"].min(), flagged["ts"].max()
                fig.add_vrect(x0=x0, x1=x1, fillcolor="#e05263", opacity=0.12,
                              line_width=0,
                              annotation_text="ground-truth event window",
                              row=1, col=1)

    fig.update_yaxes(title_text="robust deviations", row=zrow, col=1)
    fig.update_xaxes(title_text="UTC", row=zrow, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10),
                      hovermode="x unified")
    _chart(fig, width="stretch")

    # --- the numbers ------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{prov.get('rows', 0):,}", "rows scored",
            f"{prov.get('seconds', 0.0):.2f} s")
    with k2:
        kpi(str(len(prov.get("channels") or [])), "thermal channels",
            prov.get("reference", "")[:48])
    with k3:
        if shown in frame.columns:
            n_alarm = int((frame[shown] > z_threshold).fillna(False).sum())
        else:
            n_alarm = int(prov.get("alarms", 0))
        kpi(f"{n_alarm:,}", f"alarms ({reference} reference)",
            f"z > {prov.get('z_threshold', z_threshold):.1f}")
    with k4:
        rate = (100.0 * prov.get("alarms", 0) / max(1, prov.get("rows", 1)))
        kpi(f"{rate:.2f} %", "alarm rate", "of the scored rows")

    st.markdown(
        f"<div class='note'><b>Reference:</b> {prov.get('reference', 'n/a')}. "
        f"<b>Joined:</b> {' · '.join(prov.get('joined', []))}.</div>",
        unsafe_allow_html=True,
    )


def tab_explorer(live: bool) -> None:
    """Live Multi-Turbine & Weather-Conditioned Explorer.

    Three sources, one detector. The reader can run the project against the
    ENGIE site, against a labelled CARE event, or against their own turbine
    file -- and see the detector calibrate, score and alarm on it, with the
    provenance of every run on screen.
    """
    st.markdown("#### Live Multi-Turbine & Weather-Conditioned Explorer")
    st.markdown(
        "<div class='note'>The same detector, run live on the data you "
        "choose — with the provenance of every run on screen.</div>",
        unsafe_allow_html=True,
    )

    live_mod = _engine()
    source = st.radio(
        "Source", ("ENGIE La Haute Borne + ERA5", "CARE benchmark events",
                   "Upload a turbine CSV"),
        horizontal=True,
        help="ENGIE is the healthy multi-turbine site with real weather "
             "joined on. CARE is the labelled benchmark. The upload path runs "
             "the detector on your own file, self-calibrated on its history.",
    )

    result = None
    z_threshold = 4.0

    # ------------------------------------------------------- ENGIE + ERA5
    if source == "ENGIE La Haute Borne + ERA5":
        if not live:
            st.info(
                "**The 19 GB raw release is not present**, so this source "
                "cannot run here. The *Upload a turbine CSV* source below "
                "works anywhere, including on Streamlit Community Cloud — "
                "download the sample and run the detector on it."
            )
        else:
            c1, c2, c3 = st.columns([1.1, 1.6, 1.0])
            with c1:
                turbine = st.selectbox("Turbine", live_mod.ENGIE_TURBINES)
            with c2:
                window = st.selectbox("Window", list(live_mod.ENGIE_WINDOWS))
            with c3:
                with_peers = st.checkbox("Control group", value=True,
                                         help="Score the three sibling "
                                              "turbines as the control group.")
            z_threshold = st.slider("Alarm threshold (z)", 0.5, 12.0, 4.0, 0.1)
            if st.button("Run the explorer", type="primary",
                         key="run_engie"):
                with st.spinner(
                    f"Reading {turbine}'s SCADA, joining ERA5, fitting the "
                    f"reference and scoring…"
                ):
                    try:
                        result = _run_engie_cached(turbine, window, z_threshold,
                                                   with_peers)
                    except Exception as exc:
                        st.error(f"the live engine failed: {exc}")
            if turbine == "R80711":
                st.markdown(
                    "<div class='note'>R80711 is the case study: its "
                    "gearbox bearing 2 runs hot from September 2017 while its "
                    "three siblings do not. The others are the control group "
                    "— healthy throughout, so every alarm on them is a "
                    "false alarm.</div>",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------- CARE events
    elif source == "CARE benchmark events":
        if not live:
            st.info(
                "**The CARE dataset is not present** (it is 19 GB "
                "extracted), so the labelled benchmark cannot be run live "
                "here. Its scores are in the *CARE benchmark* tab, and the "
                "upload path below runs the same detector on any file."
            )
        else:
            events = live_mod.care_events(ROOT / "data")
            if not events:
                st.warning("No CARE events found under data/.")
            else:
                labels = {
                    e["event_id"]: (
                        f"{'ANOMALY' if e['label'] == 'anomaly' else 'normal '} "
                        f"· {e['farm']} · dataset {e['event_id']} "
                        f"— {e['description'][:44]}"
                    )
                    for e in events
                }
                c1, c2 = st.columns([1.6, 1.0])
                with c1:
                    event_id = st.selectbox(
                        "Event", list(labels),
                        format_func=lambda e: labels[e],
                        help="The labelled anomaly events first, then the "
                             "normal control datasets — the healthy case "
                             "matters as much as the faulty one.",
                    )
                with c2:
                    st.metric("labelled events", f"{len(events)}")
                chosen = next(e for e in events if e["event_id"] == event_id)
                st.caption(f"ground truth: rows {chosen['start']}–{chosen['end']}")
                if st.button("Run the explorer", type="primary",
                             key="run_care"):
                    with st.spinner(
                        f"Loading CARE {chosen['farm']} dataset "
                        f"{chosen['event_id']} and scoring it…"
                    ):
                        try:
                            result = _run_care_cached(chosen["farm"],
                                                     chosen["event_id"], 0.90)
                        except Exception as exc:
                            st.error(f"the live engine failed: {exc}")

    # ---------------------------------------------------------- upload
    else:
        sample = live_mod.sample_csv_path()
        d1, d2 = st.columns([1.4, 1.0])
        with d1:
            upload = st.file_uploader(
                "Turbine CSV", type=["csv"],
                help="Any 10-minute SCADA file with a timestamp, a power "
                     "column and at least one component temperature. The "
                     "columns are recognised by name.",
            )
        with d2:
            if sample.exists():
                st.download_button(
                    "Download sample turbine CSV",
                    data=sample.read_bytes(),
                    file_name="sample_turbine.csv",
                    mime="text/csv",
                    help="A real four-month slice of the ENGIE release, with "
                         "the ERA5 ambient temperature added. Use it to try "
                         "the detector without a 19 GB download.",
                )
            else:
                st.caption("the sample file is not committed")

        if sample.exists():
            with st.expander("What is in the sample?"):
                st.markdown(
                    "\n".join([
                        f"- {sample.stat().st_size/1e6:.2f} MB, "
                        "May-November 2017 for R80711, straight from the "
                        "ENGIE release (Mendeley v2, CC BY 4.0).",
                        "- `era5_temp_2m` is the **reanalysis** ambient "
                        "temperature, not a turbine sensor: the ENGIE release "
                        "ships no ambient sensor, so ERA5 supplies it.",
                        "- The engine calibrates on the file's **own first "
                        "60 %** and scores the rest - the same thing a "
                        "commissioned machine does.",
                    ])
                )

        if upload is None:
            st.info(
                "Upload a CSV, or download the sample above and upload it, to "
                "watch the detector calibrate on it."
            )
        else:
            payload = upload.getvalue()
            try:
                df, mapping = _parse_upload_cached(payload)
            except Exception as exc:
                st.error(f"could not read that file: {exc}")
                df, mapping = None, None
            if df is not None:
                st.markdown(
                    f"<div class='note'>{_engine()._describe_mapping(mapping)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([1.2, 1.0])
                with c1:
                    channel = st.selectbox(
                        "Channel to display", mapping["thermal"],
                        help="The detector always uses every temperature "
                             "channel it found; this is only which one you "
                             "look at.",
                    ) if mapping["thermal"] else None
                with c2:
                    ref_frac = st.slider(
                        "Reference fraction", 0.3, 0.85, 0.6, 0.05,
                        help="The first share of the file is the reference; "
                             "the rest is scored.")
                z_threshold = st.slider("Alarm threshold (z)", 0.5, 12.0,
                                        4.0, 0.1, key="upload_z")
                if st.button("Run the explorer", type="primary",
                             key="run_upload"):
                    with st.spinner(
                        "Calibrating on the file's own history and scoring…"
                    ):
                        try:
                            result = _run_upload_cached(payload, channel or "",
                                                        z_threshold, ref_frac)
                        except Exception as exc:
                            st.error(f"the live engine failed: {exc}")

    # The result is kept, so changing the reference below re-renders the
    # same run instead of emptying the chart.
    if result is None:
        result = st.session_state.get("explorer_result")
    else:
        st.session_state["explorer_result"] = result

    if result is not None:
        _provenance_banner(result["provenance"])
        _render_live_result(result, z_threshold)


# ---------------------------------------------------------------- tab 1
@st.cache_data(ttl=3600, show_spinner=False)
def _forecast_cached(lat: float, lon: float, days: int = 7):
    """The 7-day hourly forecast, cached for an hour.

    The cache matters twice over: it keeps the dashboard from calling the
    API on every rerun, and it means the what-if sliders are instant. If the
    network is not there -- in CI, or on a laptop offline -- the call falls
    back to a deterministic profile derived from the committed sample, and
    the caller is told which one it got.
    """
    fc = _forecast_module()
    sample = _engine().sample_csv_path()

    try:
        frame = fc.fetch_forecast(lat, lon, days=days)
        return frame, "live — Open-Meteo forecast API"
    except Exception:
        try:
            frame = fc.sample_forecast(sample, days=days)
            return frame, ("offline fallback — the sample's own hourly "
                           "climatology, NOT a forecast")
        except Exception as exc:
            return None, f"unavailable: {exc}"


@st.cache_data(show_spinner=False)
def _power_curve_cached():
    """The turbine's own power curve, from the committed sample CSV."""
    fc = _forecast_module()
    try:
        return fc.sample_power_curve(_engine().sample_csv_path())
    except Exception as exc:
        return exc


@st.cache_data(show_spinner=False)
def _checkpoint_bundle_cached(checkpoint: str, live: bool):
    """The checkpoint's bundle: computed live when the raw data is there,
    read from the committed artefact when it is not.

    The two paths must agree, and a test checks that they do -- otherwise
    the Cloud deployment would quietly be showing different numbers from
    the local one, which is the failure mode the whole single-source-of-
    truth rule exists to prevent.
    """
    fc = _forecast_module()
    if live:
        from windmill import loaders

        try:
            frame = fc.engie_checkpoint_frame(ROOT / "data", "R80711")
            cp = fc.CHECKPOINTS[checkpoint]
            return fc.checkpoint_bundle(frame, cp["prediction_start"],
                                        cp["prediction_end"]), None
        except Exception as exc:
            return None, str(exc)
    exported = fc.load_frozen_baselines()
    bundle = (exported.get("checkpoints") or {}).get(checkpoint)
    if bundle is None:
        return None, (
            f"{checkpoint} is not in results/{fc.FROZEN_BASELINES_ARTEFACT} -- "
            f"run `make snapshot`"
        )
    return bundle, None


def _shared_axis_range(timestamps) -> dict:
    """The x-axis both charts of a checkpoint must share.

    Left to itself, Plotly auto-ranges each figure separately, and the
    power chart can end up zoomed to a single day while the weather chart
    below it shows the whole week -- which makes the two unreadable
    together. So the range is computed once, from the data, and handed to
    both.
    """
    ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True)
    return dict(range=[ts.min(), ts.max()], tickformat="%b %d")


def _training_callout(bundle: dict) -> str:
    """The bold training-date callout.

    The rule is that the model is trained on exactly the thirty days
    immediately before the prediction window, and this is where that is
    stated -- not in a footnote, where nobody reads it.
    """
    fit = bundle["fits"]["primary"]
    return (f"**Trained on: {fit['window_start']} → {fit['window_end']}** "
            f"(Trailing 30 Days · {fit['rows']:,} clean rows · "
            f"{fit.get('seconds', 0.0):.2f} s)")


def _power_lines(show_30d: bool, show_1yr: bool,
                 show_2yr: bool) -> list[tuple[str, str, str, str, float]]:
    """Which of the four lines the power chart draws.

    The actual output is always drawn -- it is the thing being predicted.
    The three predictions are each behind a toggle, and the chart opens
    with the fresh one only, so a presenter can add the stale curves when
    they want to show the divergence.

    Returned as (fit key, label, dash, colour, width).
    """
    lines = [("actual", "Actual measured power (kW)", "solid", "#E5E7EB", 2.2)]
    if show_30d:
        lines.append(("primary", "Predicted — last 30d fit (kW)", "solid",
                      "#00D4AA", 2.2))
    if show_1yr:
        lines.append(("1_year_back", "Predicted — frozen 1-yr-back (kW)",
                      "dash", "#F59E0B", 1.8))
    if show_2yr:
        lines.append(("2_year_back", "Predicted — frozen 2-yrs-back (kW)",
                      "dot", "#F43F5E", 1.8))
    return lines


def _tab_forecast_checkpoint(fc, checkpoint: str, live: bool,
                             show_30d: bool = True,
                             show_1yr: bool = False,
                             show_2yr: bool = False) -> None:
    """One historical checkpoint: the multi-year staleness comparison.

    The primary model is always trained on exactly the thirty days
    immediately before the prediction window. The two stale baselines are
    the same rule applied one and two years earlier, so the only thing
    that differs between them is how old the reference is.
    """
    import plotly.graph_objects as go

    with st.spinner(f"Scoring the {checkpoint.replace('_', ' ')}…"):
        bundle, error = _checkpoint_bundle_cached(checkpoint, live)
    if error or bundle is None:
        st.error(f"The checkpoint could not be scored: {error}")
        return

    scored = fc.score_checkpoint(bundle)
    fits = bundle["fits"]
    rows = bundle.get("rows") or []
    if not rows:
        st.error("The checkpoint has no rows in the target week.")
        return

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # --- the bold training callout ----------------------------------------
    # The callout is raw HTML, so the emphasis has to be an HTML tag: a
    # markdown ** inside a div is not parsed and comes out as literal
    # asterisks.
    st.markdown(
        f"<div class='callout'>{_training_callout(bundle).replace('**', '<b>')}</div>",
        unsafe_allow_html=True,
    )

    # --- the KPI row: the tracking error of each fit ----------------------
    def _mae(label):
        s = scored["fits"].get(label, {})
        return s.get("mae_kw") if s.get("available") else None

    def _bias(label):
        s = scored["fits"].get(label, {})
        if not s.get("available"):
            return None
        pred = [v for v in s["predicted_kw"] if v is not None]
        actual = df["actual_power_kw"].to_numpy()[:len(pred)]
        if not pred:
            return None
        return float(sum(p - a for p, a in zip(pred, actual)) / len(pred))

    # Each card follows its toggle: the card is the number for that line,
    # so a line that is not drawn has no card either. The row therefore
    # opens with two cards and grows to four.
    cards: list[tuple[str, str, str]] = []
    if show_30d:
        cards.append((
            f"{_mae('primary'):,.1f} kW" if _mae("primary") is not None else "—",
            "MAE — last 30-day fit",
            f"trained {fits['primary']['window_start']} → "
            f"{fits['primary']['window_end']}"))
    if show_1yr:
        cards.append((
            f"{_mae('1_year_back'):,.1f} kW"
            if _mae("1_year_back") is not None else "—",
            "MAE — frozen 1-year-back",
            f"trained {fits['1_year_back']['window_start']} → "
            f"{fits['1_year_back']['window_end']}"))
    if show_2yr:
        cards.append((
            f"{_mae('2_year_back'):,.1f} kW"
            if _mae("2_year_back") is not None else "—",
            "MAE — frozen 2-years-back",
            f"trained {fits['2_year_back']['window_start']} → "
            f"{fits['2_year_back']['window_end']}"))
    cards.append((f"{df['actual_power_kw'].mean():,.0f} kW",
                  "actual mean power",
                  f"{len(df):,} rows in the target week"))

    for col, (value, label, delta) in zip(st.columns(len(cards)), cards):
        with col:
            kpi(value, label, delta)

    # With every prediction switched off there is nothing to compare the
    # actual output against, so say so rather than showing a lone line.
    if not (show_30d or show_1yr or show_2yr):
        st.info(
            "Select at least one prediction baseline above to compare it "
            "against the actual measured power. The actual output is still "
            "plotted below."
        )

    # --- the chart: four lines --------------------------------------------
    # One x-range for BOTH charts, so a gust in the weather panel lines up
    # vertically with the power curve above it. Left to itself, Plotly
    # auto-ranges each panel separately, and the power chart can zoom to a
    # single day while the weather chart shows the whole week.
    x_axis = _shared_axis_range(df["ts"])

    # The colour language: the ACTUAL output is the neutral, quiet line,
    # and the predictions are what the demo is about -- so healthy power
    # never looks like an alarm. The chart opens with two lines: the actual
    # output and the fresh 30-day prediction. The stale curves are one
    # click away.
    fig = go.Figure()
    for key, name, dash, colour, width in _power_lines(
            show_30d, show_1yr, show_2yr):
        if key == "actual":
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["actual_power_kw"], name=name, mode="lines",
                line=dict(width=width, dash=dash, color=colour)))
            continue
        s = scored["fits"].get(key, {})
        if not s.get("available"):
            continue
        fig.add_trace(go.Scatter(
            x=df["ts"], y=s["predicted_kw"], name=name, mode="lines",
            line=dict(width=width, dash=dash, color=colour)))
    fig.update_layout(
        title=(f"Power in the target week — the actual output against what "
               f"each reference predicts"),
        xaxis=x_axis, xaxis_title="UTC", yaxis_title="power (kW)", height=480,
        # The same left and right margins as the weather chart below, which
        # has a second axis on the right: the two plots must be the same
        # width, or the dates on one do not line up with the other.
        margin=dict(l=110, r=140, t=54, b=96), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.34, xanchor="left",
                    x=0),
    )
    _chart(fig, width="stretch")

    # --- the wind and the air --------------------------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["ts"], y=df["wind_speed"], name="wind speed at 100 m (m/s)",
        mode="lines", line=dict(width=1.6, color="#5fd39a")))
    fig2.add_trace(go.Scatter(
        x=df["ts"], y=df["air_density"], name="air density (kg/m³)",
        mode="lines", line=dict(width=1.6, color="#9fb4ff"), yaxis="y2"))
    fig2.update_layout(
        title="The weather the prediction was made from",
        xaxis=x_axis, xaxis_title="UTC", height=360,
        margin=dict(l=110, r=140, t=54, b=96),
        hovermode="x unified",
        yaxis=dict(title="wind speed (m/s)"),
        yaxis2=dict(title="air density (kg/m³)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.42, xanchor="left",
                    x=0),
    )
    _chart(fig2, width="stretch")

    # --- what the numbers actually say ------------------------------------
    st.info(
        "**What the tracking error says**\n"
        "\n"
        "- Steep-region MAE (where the curve's *shape* matters, not its "
        "level): **43.9 → 45.0 → 49.0 kW** for the 30-day, "
        "one-year-back and two-year-back fits.\n"
        "- Signed bias, ordered the same way: **−13.0 → −14.7 → "
        "−19.7 kW**. The older the reference, the more it "
        "under-predicts.\n"
        "- **But** a single week's power is dominated by that week's "
        "weather, so reference age is second-order here — unlike Tab 2, "
        "where a stale reference is the whole problem."
    )


def tab_forecast() -> None:
    """Weather Forecast & Power Prediction -- the operational opening.

    The order matters: a control room asks "what will the wind do, what
    should we produce, and is the machine producing it?" before it asks
    which model scored best. So the forecast comes first and the benchmark
    comparisons come later, in their own tab.
    """
    fc = _forecast_module()
    live = live_available()

    st.subheader("Weather Forecast & Power Prediction")
    st.markdown(
        "Given this week's weather, **how much power should this turbine "
        "be producing** — and how much is it actually producing? The power "
        "curve is the machine's own, measured from its SCADA."
    )

    # --- the demo checkpoint selector, and the stale-baseline toggles -------
    # A presenter can move between curated historical windows without
    # touching anything else, and the two toggles sit right beside it.
    labels = {"__forecast__": "Live 7-day forecast — Open-Meteo (now)"}
    for key, cp in fc.CHECKPOINTS.items():
        labels[key] = f"{cp['label']} — {cp['note']}"
    options = list(labels)
    # The app opens on the first historical checkpoint, so the comparison
    # controls are on screen immediately; the live forecast is one click away.
    default_index = (options.index("checkpoint_1")
                     if "checkpoint_1" in options else 0)
    choice = st.selectbox(
        "Demo Checkpoint", options,
        index=default_index,
        format_func=lambda k: labels[k], key="forecast_checkpoint",
        help="Switch between the live forecast and curated historical "
             "windows. Each checkpoint is scored by a reference trained on "
             "exactly the thirty days before it, and by references one and "
             "two years older.")

    # The stale-baseline labels follow the CHECKPOINT, not a hard-coded
    # year: on a 2016 checkpoint the one-year-back curve is 2015's and the
    # weather is 2016's.
    if choice == "__forecast__":
        st.caption(
            "The frozen-baseline comparison applies to the historical "
            "checkpoints; choose one above to see it."
        )
        show_30d = show_1yr = show_2yr = False
    else:
        target_year = int(fc.CHECKPOINTS[choice]["prediction_start"][:4])
        t1, t2, t3 = st.columns(3)
        with t1:
            show_30d = st.checkbox(
                "Last 30 days rolling", value=True, key="show_30d",
                help="The reference fitted on the thirty days before the "
                     "target week. On by default; switch it off to plot only "
                     "the actual output against the stale curves.")
        with t2:
            show_1yr = st.checkbox(
                f"Frozen 1 year before "
                f"({target_year - 1} curve + {target_year} weather)",
                value=False, key="show_1yr",
                help="Off by default, so the chart opens with two lines.")
        with t3:
            show_2yr = st.checkbox(
                f"Frozen 2 years before "
                f"({target_year - 2} curve + {target_year} weather)",
                value=False, key="show_2yr",
                help="Off by default, so the chart opens with two lines.")

    if choice != "__forecast__":
        _tab_forecast_checkpoint(fc, choice, live, show_30d, show_1yr, show_2yr)
        return

    # --- the site ----------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1.3, 1.0, 1.0, 1.0])
    with c1:
        site_label = st.selectbox(
            "Site", ["ENGIE La Haute Borne (Meuse, France)", "Custom"],
            help="The wind farm the project measures on. The custom option "
                 "takes any latitude and longitude.")
    custom = site_label == "Custom"
    with c2:
        lat = st.number_input("Latitude", value=fc.SITE["lat"], format="%.4f",
                               disabled=not custom)
    with c3:
        lon = st.number_input("Longitude", value=fc.SITE["lon"], format="%.4f",
                               disabled=not custom)
    with c4:
        st.caption(f"**{'La Haute Borne' if not custom else 'Custom site'}**  \n"
                   f"`{lat:.4f}, {lon:.4f}`\n\n"
                   f"The forecast and the power curve are the site's own.")

    # --- the what-if -------------------------------------------------------
    w1, w2 = st.columns(2)
    with w1:
        wind_shift = st.slider(
            "Wind speed shift (m/s)", -3.0, 3.0, 0.0, 0.1,
            help="What if the wind were stronger or weaker than the "
                 "forecast? The predicted power moves with it.")
    with w2:
        temp_shift = st.slider(
            "Temperature shift (°C)", -10.0, 10.0, 0.0, 0.5,
            help="What if the air were colder or warmer? Colder air is "
                 "denser, and denser air carries more energy at the same "
                 "wind speed.")

    # --- the forecast ------------------------------------------------------
    with st.spinner("Fetching the 7-day hourly forecast…"):
        frame, source = _forecast_cached(float(lat), float(lon))
    curve = _power_curve_cached()

    if frame is None:
        st.error(f"The forecast is unavailable: {source}")
        return
    if isinstance(curve, Exception):
        st.error(f"The power curve could not be built from the sample: {curve}")
        return
    if curve is None:
        st.error("The power curve is missing.")
        return

    st.caption(f"Forecast source: **{source}** · "
               f"{len(frame)} hourly rows · "
               f"wind {frame['wind_speed_100m'].min():.1f}–"
               f"{frame['wind_speed_100m'].max():.1f} m/s · "
               f"air {frame['temperature_2m'].min():.1f}–"
               f"{frame['temperature_2m'].max():.1f} °C")

    # The forecast is predicted with the same rule as the checkpoints: a
    # reference fitted on the trailing thirty days of the record.
    try:
        win_start, win_end = fc.trailing_window(fc.sample_csv_path()
                                                 if hasattr(fc, "sample_csv_path")
                                                 else _engine().sample_csv_path())
        curve_rows = len(pd.read_csv(_engine().sample_csv_path()))
        st.markdown(
            f"<div class='callout'><b>Trained on: {win_start} → {win_end}</b> "
            f"(Trailing 30 Days · the power curve is fitted on the recent "
            f"past, not the whole record)</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    with st.spinner("Applying the power curve…"):
        pred = fc.forecast_frame(frame, curve, wind_shift=wind_shift,
                                 temp_shift=temp_shift)

    mean_kw = float(pred["predicted_power_kw"].mean())
    mean_kw_base = float(fc.forecast_frame(frame, curve)["predicted_power_kw"].mean())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{mean_kw:,.0f} kW", "predicted power",
            "mean over the 7-day horizon")
    with k2:
        kpi(f"{mean_kw - mean_kw_base:+,.0f} kW", "what-if effect",
            f"wind {wind_shift:+.1f} m/s, air {temp_shift:+.1f} °C")
    with k3:
        kpi(f"{float(pred['air_density'].mean()):.3f} kg/m³", "mean air density",
            f"{float(pred['density_ratio'].mean()):.3f} of the reference")
    with k4:
        kpi(f"{float(pred['predicted_power_kw'].max()):,.0f} kW", "peak predicted",
            f"curve from {len(curve)} measured bins")

    # --- chart 1: the forecast ---------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # One x-range for both charts, so the weather and the power it produced
    # line up vertically. Plotly auto-ranges each figure separately, and
    # the two can end up showing different spans.
    fx_axis = _shared_axis_range(pred["ts"])

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=pred["ts"], y=pred["wind_speed_100m"], name="wind speed at 100 m (m/s)",
        mode="lines", line=dict(width=1.8, color="#5fd39a")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pred["ts"], y=pred["air_density"], name="air density (kg/m³)",
        mode="lines", line=dict(width=1.8, color="#9fb4ff"),
        yaxis="y2"), row=1, col=1)
    fig.update_layout(
        title="Forecast: wind speed and the air density it implies",
        xaxis_title="UTC", height=380, margin=dict(l=110, r=140, t=54, b=10),
        hovermode="x unified",
        xaxis=fx_axis,
        yaxis=dict(title="wind speed (m/s)"),
        yaxis2=dict(title="air density (kg/m³)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left",
                    x=0),
    )
    _chart(fig, width="stretch")

    # --- chart 2: the predicted power -------------------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=pred["ts"], y=pred["predicted_power_kw"],
        name="predicted healthy power (kW)", mode="lines",
        line=dict(width=2.0, color="#5fd39a"), fill="tozeroy",
        fillcolor="rgba(95,211,154,0.10)"))
    fig2.add_trace(go.Scatter(
        x=pred["ts"], y=pred["predicted_power_kw"].rolling(6, center=True).mean(),
        name="6-hour mean", mode="lines",
        line=dict(width=1.0, dash="dot", color="#9fb4ff")))
    fig2.update_layout(
        title=("Predicted healthy power generation — move the sliders and "
               "the curve moves"),
        xaxis=fx_axis, xaxis_title="UTC", yaxis_title="power (kW)",
        height=380,
        margin=dict(l=110, r=140, t=54, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left",
                    x=0),
    )
    _chart(fig2, width="stretch")




# ---------------------------------------------------------------- tab 2
# ---------------------------------------------------------------- tab 3
def tab_benchmark(engie, care_score, multiv, lgbm, ae_audit, audit, thermal):
    """Model & Benchmark Comparison -- the isolated technical study.

    Everything that compares one detector with another is here: the
    deterministic arm against the learned one, the false-alarm rates by year,
    the CARE benchmark, and the audit of the published autoencoder baseline.
    """
    st.subheader("Model & Benchmark Comparison")

    _model_comparison(engie, care_score, multiv, lgbm)
    st.markdown("---")
    st.subheader("The CARE benchmark — and what the published 0.66 really is")

    # --- sub-scores across arms -------------------------------------------
    import plotly.graph_objects as go

    arms = []
    if care_score:
        arms.append(("power-only residual (baseline)", care_score["sub_scores"],
                     care_score["care_score"]))
    if multiv:
        arms.append(("multivariate bins (max z)",
                     multiv["max_combiner"]["sub_scores"],
                     multiv["max_combiner"]["care_score"]))
        arms.append(("multivariate bins (L2)", multiv["l2_combiner"]["sub_scores"],
                     multiv["l2_combiner"]["care_score"]))
    if lgbm:
        arms.append(("LightGBM thermal NBM", lgbm["sub_scores"], lgbm["care_score"]))
    ae_cfg = (ae_audit or {}).get("per_config", {}).get("published_s0")
    if ae_cfg:
        arms.append(("published AE (reproduced here)",
                     {k: ae_cfg.get(k) for k in
                      ("coverage", "accuracy", "reliability", "earliness")},
                     ae_cfg.get("care_score")))

    if arms:
        sub_metrics = ["coverage", "accuracy", "reliability", "earliness"]
        fig = go.Figure()
        for name, subs, _score in arms:
            fig.add_trace(go.Bar(
                x=[m for m in sub_metrics], y=[subs.get(m) or 0 for m in sub_metrics],
                name=name,
                hovertemplate="%{x}: %{y:.4f}<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(barmode="group", height=380,
                          title="CARE sub-scores over all 95 labelled events",
                          yaxis_title="sub-score", xaxis_title=None,
                          margin=dict(l=10, r=10, t=60, b=10))
        left, right = st.columns([3, 2])
        with left:
            _chart(fig, width="stretch")
        with right:
            rows = [{"arm": n, "CARE": sc, **{k: (subs or {}).get(k) for k in
                      ("coverage", "accuracy", "reliability", "earliness")}}
                    for n, subs, sc in arms]
            st.table(pd.DataFrame(rows).round(4))
            if ae_audit:
                st.markdown(
                    f"<div class='note'><b>The published AE figure is "
                    f"{_fmt((ae_audit.get('published_baselines') or {}).get('AE (published)'), 2)}"
                    f"</b> — but the authors' own default configuration "
                    f"reproduces "
                    f"{_fmt((ae_audit.get('per_config') or {}).get('published_s0', {}).get('care_score'))} "
                    f"here. The difference is the Optuna search, the per-farm "
                    f"adaptive thresholds and the per-farm sensitivities, none "
                    f"of which has a held-out protocol behind it.</div>",
                    unsafe_allow_html=True)
    else:
        st.warning("no CARE artefacts found.")

    st.markdown("")

    # --- what the benchmark can and cannot score --------------------------
    st.markdown("#### What the benchmark can and cannot score")
    if audit:
        pf = audit.get("per_farm", {})
        rows = [{
            "farm": farm,
            "anomaly events": v.get("n_anomaly"),
            "events with ZERO scorable rows": v.get("n_anomaly_with_zero_scorable_rows"),
            "mean % of window in normal operation": _fmt(
                (v.get("mean_frac_event_rows_normal") or 0) * 100, 1),
            "mean % of window in status 4 (fault)": _fmt(
                (v.get("mean_frac_event_rows_status4") or 0) * 100, 1),
        } for farm, v in pf.items()]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.table(pd.DataFrame(rows), hide_index=True)
        with c2:
            st.metric(f"{audit.get('all_anomaly_events_zero_scorable')} / "
                      f"{audit.get('n_anomaly_events')}",
                      "anomaly events with no scorable rows")
            st.markdown(
                "<div class='note'>The CARE score only credits rows in "
                "normal operation. Wind Farm A's fault occupies ~95 % of its "
                "event windows, so no detector can earn credit for detecting "
                "them — a property of the published labels, not of any "
                "detector.</div>",
                unsafe_allow_html=True,
            )

    # --- the scope limit ---------------------------------------------------
    if thermal:
        st.markdown("#### What a temperature model can see in CARE")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi(_fmt(thermal.get("max_z_median_in_event"), 4), "median z inside the event")
        with c2:
            kpi(_fmt(thermal.get("max_z_median_outside"), 4), "median z outside it")
        with c3:
            kpi(_fmt(thermal.get("max_z_event_higher_fraction"), 4),
                "events where the fault window scores higher",
                "a coin flip — most CARE faults are not thermal")
        st.markdown(
            "<div class='note'>CARE's faults are mostly hydraulic, pitch, "
            "converter and safety-chain failures. A thermal detector cannot "
            "see them, and no threshold recovers a signal the channels do "
            "not carry — a scope limit, not a badly chosen threshold.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- tab 4
def tab_forensics():
    st.subheader("Engineering and data forensics")
    st.markdown(
        "The numbers are only worth as much as the process behind them."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🧪 Single source of truth")
        st.markdown(
            "Every number in the README and in this dashboard is "
            "**generated** from `results/*.json`. CI re-runs the generator "
            "and fails if the committed file is not reproduced byte for "
            "byte."
        )
    with c2:
        st.markdown("#### 🧭 A pipeline, not a pile of scripts")
        st.markdown(
            "`scripts/pipeline/` produces the artefacts; "
            "`scripts/diagnostics/` are probes that write nothing. The "
            "Makefile holds the execution order; logic lives in "
            "`src/windmill/`."
        )
    with c3:
        st.markdown("#### 🤖 CI as a determinism guard")
        st.markdown(
            "CI installs the pinned dependencies on Python 3.11, "
            "syntax-checks the tree, runs the tests, **re-runs the "
            "documentation generator** and fails if the working tree "
            "changed."
        )

    st.markdown("")
    st.markdown("#### Upstream data bugs this project caught")
    bugs = [
        ("**100× air-density error**",
         "open-meteo reports surface pressure in hPa. The ideal-gas law needs "
         "Pa. Getting it wrong scales density by 100 and silently corrupts "
         "every normalised power value — no error, just a plausible-looking "
         "number."),
        ("**The cluster filter removed the documented fault**",
         "The ENGIE release's outlier filtering deleted the one documented "
         "gearbox bearing failure (R80721, 2015, 84.12 °C peak). The cleaned "
         "CSVs peak at 78.85 °C. A project that claimed to detect that event "
         "would have been claiming to detect data that is not there."),
        ("**CARE's Farm A dates are shifted by six years**",
         "To hide the year, CARE moved each Farm A file by 2,191 days (2,192 "
         "before the 2016 leap day). Mapped back, the turbines match EDP's "
         "original data reading for reading on about 170,000 shared "
         "timestamps, and each CARE fault label ends within 10 minutes of a "
         "failure in EDP's logbook. Only then were the labels used."),
        ("**The status column is the fault annotation**",
         "In Wind Farm A, `status_type_id` is built from the EDP failure "
         "logbook — the dataset authors say it is for filtering training data. "
         "Used as a feature it would be the answer key. Measured: removing it "
         "changes the autoencoder's score by nothing, because the AE trains "
         "only on normal rows where the column is constant."),
        ("**Half the benchmark cannot be scored**",
         "11 of 45 labelled anomaly events have no rows in normal operation "
         "inside their window. The published labels, not the detector, cap what "
         "Coverage can be."),
    ]
    for title, body in bugs:
        st.markdown(f"**{title}**")
        st.markdown(f"<div class='note'>{body}</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### Reproduce it")
    st.code("""make setup && make data && make pipeline && make test""", language="bash")
    st.markdown(
        "<div class='note'>~19 GB of public data, no credentials, no API "
        "key. Every dataset's SHA-256 is recorded.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- main
# ------------------------------------------------- the daily pages (1, 2, 3)
@st.cache_data(show_spinner=False)
def load_daily_series() -> dict | None:
    return _load("daily_series.json")


def _daily_view():
    """The page rules. Streamlit re-runs app.py on every change but keeps
    modules it imported from src/ in memory, so a running dashboard would
    keep the old rules after an edit; reload the module when its file is
    newer than the copy in memory."""
    import importlib
    import os

    _ensure_src()
    from windmill import daily_view
    stamp = os.path.getmtime(daily_view.__file__)
    if getattr(daily_view, "_loaded_mtime", None) != stamp:
        daily_view = importlib.reload(daily_view)
        daily_view._loaded_mtime = stamp
    return daily_view


C_ACTUAL = "#E5E7EB"     # the sensor: the quiet neutral line, as elsewhere
C_MODEL = "#00D4AA"      # the model fed the day's weather: teal, as elsewhere
C_FORECAST = "#A78BFA"   # the same model fed the forecast issued days before
C_HOT = "#E05263"
C_OLD = "#F5B841"        # an older model, not retrained: amber


def _older_label(age: int) -> str:
    return {3: "3-month-old", 12: "1-year-old", 24: "2-year-old"}.get(
        age, f"{age}-month-old")


def _older_model_control(series: dict, key: str):
    """A tick box and, when ticked, the age of the older model to show."""
    ages = series.get("older_model_ages") or []
    if not ages:
        return None
    on = st.checkbox("Show an older model (not retrained)", value=False, key=f"{key}_show_old")
    if not on:
        return None
    return st.selectbox("How old", ages, index=len(ages) - 1,
                        format_func=lambda a: f"{_older_label(a)} model", key=f"{key}_old_age")


def _y_range(frames, tolerance: float, lead: int, old_age: int | None = None) -> list[float]:
    """One temperature range for several charts, so the same tolerance band
    is drawn the same height on each of them."""
    vals = []
    for df in frames:
        chk = df[df["status"] == "checked"]
        for c in ("actual", "pred", f"f{lead}") + ((f"o{old_age}",) if old_age else ()):
            if c in chk:
                vals += [v for v in chk[c].tolist() if v == v]
        if "pred" in chk:
            vals += [v + d for v in chk["pred"].tolist() if v == v
                     for d in (tolerance, -tolerance)]
    if not vals:
        return None
    return [min(vals) - 1.0, max(vals) + 1.0]


def _window_alarms(series: dict, turbine: str, sensor: str, start: str,
                   end: str, tolerance: float):
    """The alarms of one turbine inside a window, found on the whole record so
    a hot spell that began before the window is not counted again."""
    dv = _daily_view()
    a = dv.alarms(dv.frame(series, turbine, sensor), tolerance)
    return a[(a["day"] >= pd.Timestamp(start)) & (a["day"] <= pd.Timestamp(end))]


def _three_line_figure(df, tolerance: float, lead: int, title: str,
                       y_range=None, alarms=None, show_pred: bool = True,
                       show_forecast: bool = True, limit_c: float | None = None,
                       x_range=None, old_age: int | None = None):
    """The forecast, the on-day prediction and the sensor, one point per day,
    with the tolerance as a band around the on-day prediction. The sensor is
    always drawn; the prediction (with its band) and the forecast can be
    switched off, so a presenter can show the real data first."""
    import plotly.graph_objects as go

    fc = f"f{lead}"
    chk = df[df["status"] == "checked"]
    fig = go.Figure()
    if show_pred:
        fig.add_trace(go.Scatter(
            x=chk["day"], y=chk["pred"] + tolerance, mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=chk["day"], y=chk["pred"] - tolerance, mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(0,212,170,0.12)",
            name=f"tolerance ±{tolerance:g} K", hoverinfo="skip"))
    if show_forecast and fc in df:
        fig.add_trace(go.Scatter(
            x=chk["day"], y=chk[fc], mode="lines+markers",
            name=f"forecast, issued {lead} day{'s' if lead > 1 else ''} before",
            line=dict(width=2, dash="dot", color=C_FORECAST),
            marker=dict(size=6), connectgaps=False))
    old = f"o{old_age}" if old_age else None
    if old and old in chk and chk[old].notna().any():
        # An older model, fed the same day's weather: what a model that was
        # not retrained would have said, and the days it would call hot.
        name = _older_label(old_age)
        fig.add_trace(go.Scatter(
            x=chk["day"], y=chk[old], mode="lines+markers",
            name=f"predicted by the {name} model", connectgaps=False,
            line=dict(width=2.4, dash="dashdot", color=C_OLD), marker=dict(size=6)))
        oh = chk[(chk["actual"] - chk[old]) > tolerance]
        if len(oh):
            fig.add_trace(go.Scatter(
                x=oh["day"], y=oh["actual"], mode="markers",
                name=f"hot day under the {name} model",
                marker=dict(size=17, symbol="square-open", color=C_OLD,
                            line=dict(width=3, color=C_OLD))))
    if show_pred:
        fig.add_trace(go.Scatter(
            x=chk["day"], y=chk["pred"], mode="lines+markers",
            name="predicted on the day (actual weather)",
            line=dict(width=2.4, dash="dash", color=C_MODEL), marker=dict(size=6)))
    fig.add_trace(go.Scatter(
        x=chk["day"], y=chk["actual"], mode="lines+markers",
        name="sensor (daily mean)", line=dict(width=2.6, color=C_ACTUAL),
        marker=dict(size=7)))
    dv = _daily_view()
    hot = chk[dv.lights(chk, tolerance) == "hot"]
    if not show_pred:
        # The hot days and the call-outs are the model's verdicts: without
        # the prediction on screen they would have no visible reason.
        hot, alarms = hot.iloc[0:0], None
    if len(hot):
        # The same language as the daily grid: orange says the sensor ran hot,
        # only an alarm marker says whether the call-out was real.
        fig.add_trace(go.Scatter(
            x=hot["day"], y=hot["actual"], mode="markers",
            name=f"hot day: over {tolerance:g} K above the prediction",
            marker=dict(size=13, color=dv.LIGHT_COLOURS["hot"], symbol="square")))
    if alarms is not None and len(alarms):
        on = chk.set_index("day")["actual"]
        for real, name, marker in (
                (True, "model call-out (first hot day of a run), in the abnormal period",
                 dict(symbol="circle", size=22, color="rgba(0,0,0,0)",
                      line=dict(width=4, color=dv.ALARM_COLOURS["real"]))),
                (False, "model call-out (first hot day of a run), outside the abnormal period",
                 dict(symbol="circle-open", size=22,
                      color=dv.ALARM_COLOURS["false"],
                      line=dict(width=3, color=dv.ALARM_COLOURS["false"])))):
            sel = alarms[alarms["real"] == real]
            sel = sel[sel["day"].isin(on.index)]
            if len(sel):
                fig.add_trace(go.Scatter(
                    x=sel["day"], y=on.reindex(sel["day"]).to_numpy(),
                    mode="markers", name=name, marker=marker))
    fig.update_layout(
        title=title, height=560, hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=150),
        legend=dict(orientation="h", yref="container", y=0, yanchor="bottom",
                    x=0, xanchor="left"))
    if limit_c is not None:
        # The old rule, on the same axis: what a fixed alarm limit would see.
        fig.add_hline(y=limit_c, line=dict(color=_daily_view().ALARM_COLOURS["real"],
                                           width=2, dash="dash"),
                      annotation_text=f"fixed alarm limit {limit_c:g} °C",
                      annotation_position="top left")
    fig.update_yaxes(title_text="°C")
    if y_range:
        fig.update_yaxes(range=y_range)
    if x_range:
        # The period in the title, exactly: charts side by side then share
        # their days, whatever day each turbine was first judged.
        fig.update_xaxes(range=x_range)
    return fig


def _demo_cards(card: dict, badge: dict) -> None:
    cols = st.columns(4)
    with cols[0]:
        kpi(f"{card['days_hot']} of {card['days_checked']}", "days flagged hot")
    with cols[1]:
        kpi(card["first_hot"] or "none", "first call-out")
    with cols[2]:
        kpi(f"{card['mae_on_day']:.1f} K" if card["mae_on_day"] is not None else "–",
            "sensor vs on-day prediction")
    with cols[3]:
        kpi(f"{card['mae_forecast']:.1f} K" if card["mae_forecast"] is not None else "–",
            "sensor vs forecast")
    st.markdown(f"<div class='note'>{badge['text']}</div>", unsafe_allow_html=True)


def page_demo(series: dict | None) -> None:
    """Page 1: one month, a healthy turbine beside the faulty one."""
    st.subheader("How it works: one month, two turbines")
    st.markdown("<div class='note'>Data: ENGIE's La Haute Borne wind farm in France, "
                "four Senvion MM82 turbines (2,050 kW), open SCADA readings every "
                "10 minutes, 2013 to 2018, joined to Copernicus ERA5 weather.</div>",
                unsafe_allow_html=True)
    if not series:
        st.info("results/daily_series.json is missing: run `make daily`.")
        return
    dv = _daily_view()
    tol = series["tolerance_k"]
    demo = dv.DEMO
    st.markdown(
        f"At the start of each month a model is trained on a month of **healthy** "
        f"data and then frozen. Each day it is fed only that day's power and "
        f"weather and predicts what a healthy bearing should read. If the sensor "
        f"runs more than **{tol:g} K** hotter on average, or holds the fixed "
        f"**80 °C** limit for an hour, a technician is called out that day. "
        f"Days ahead, the same model fed the **weather forecast** says what to expect.")
    with st.container(key="demo_controls"):
        st.markdown("**Show on both charts:** tick a box to compare the sensor "
                    "with the model")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_pred = st.checkbox("Show the daily prediction (actual weather)",
                                    value=False, key="demo_show_pred")
        with c2:
            show_fc = st.checkbox("Show the forecast prediction (up to 7 days ahead)",
                                  value=False, key="demo_show_forecast")
        with c3:
            old_age = _older_model_control(series, "demo")
        lead = demo["lead"]
        if show_fc:
            lead = st.select_slider("Forecast issued how many days before",
                                    options=series["leads"], value=demo["lead"],
                                    key="demo_lead")
    window = f"{demo['start']} to {demo['end']}"
    roles = (("healthy", "healthy"), ("faulty", "developing a fault"))
    frames = {r: dv.frame(series, demo[r], series["primary"], demo["start"],
                          demo["end"]) for r, _ in roles}
    # Both charts on one temperature scale: the same ±tolerance band, drawn
    # the same height, so the two turbines are compared like for like.
    y_range = _y_range(frames.values(), tol, lead, old_age)
    _ensure_src()
    from windmill import daily_check as dck
    limit_c = dck.HOT_C          # the fixed-limit baseline, 80 C for an hour
    if y_range:
        y_range = [y_range[0], max(y_range[1], limit_c + 2.0)]
    for role, label in roles:
        t = demo[role]
        df = frames[role]
        card = dv.demo_summary(df, tol, lead)
        _chart(_three_line_figure(
            df, tol, lead,
            f"{t}, {label}: gearbox bearing temperature, {window} "
            f"(tolerance ±{tol:g} K on both)", y_range=y_range,
            alarms=_window_alarms(series, t, series["primary"], demo["start"],
                                  demo["end"], tol),
            show_pred=show_pred, show_forecast=show_fc, limit_c=limit_c, old_age=old_age,
            x_range=[pd.Timestamp(demo["start"]) - pd.Timedelta(hours=12), pd.Timestamp(demo["end"]) + pd.Timedelta(hours=12)]),
            width="stretch")
        _demo_cards(card, dv.badge(series, t, demo["end"], _month_of_readings(series)))
    st.markdown(
        "<div class='note'>With the prediction shown, an orange square is a hot day "
        "(the sensor more than the tolerance above the healthy prediction) and a "
        "circle is the model's call-out: the first hot day of a run, when a "
        "technician is sent. The following hot days of the same run get no new "
        "call-out. Both can fall below the fixed 80 °C limit.</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='note'>The weather forecast is simulated: ERA5 reanalysis plus "
        "an error that grows with lead time, because archived forecasts for the "
        "site are not in the data. The call-out is made on the day, from the "
        "actual weather; the forecast is for planning.</div>",
        unsafe_allow_html=True)


def _month_of_readings(series: dict) -> int:
    return int(series.get("month_of_readings", 30 * 144))


def _strip_figure(rows, tolerance: float, dv, alarms=None):
    """The traffic light: one square per day per turbine, and under each row
    a circle on every alarm (the first day of a run of hot days, when the
    technician is called out): filled red when the alarm was real, open
    amber when it was false."""
    import plotly.graph_objects as go

    alarms = alarms or {}
    n = len(rows)
    ypos = {t: n - 1 - i for i, (t, _) in enumerate(rows)}   # first row on top
    fig = go.Figure()
    for kind in dv.LIGHTS:
        xs, ys = [], []
        for t, df in rows:
            sel = df[dv.lights(df, tolerance) == kind]
            xs += list(sel["day"])
            ys += [ypos[t]] * len(sel)
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=dv.LIGHT_LABELS[kind],
                legendgroup="day", legendgrouptitle_text="The day",
                hovertemplate="%{x|%d %b %Y}<extra>" + dv.LIGHT_LABELS[kind] + "</extra>",
                marker=dict(symbol="square", size=22, color=dv.LIGHT_COLOURS[kind])))
    real_c, false_c = dv.ALARM_COLOURS["real"], dv.ALARM_COLOURS["false"]
    for real, name, marker in (
            (True, "model call-out (first hot day of a run), in the abnormal period",
             dict(symbol="circle", size=16, color=real_c,
                  line=dict(width=2, color=real_c))),
            (False, "model call-out (first hot day of a run), outside the abnormal period",
             dict(symbol="circle-open", size=16, color=false_c,
                  line=dict(width=3, color=false_c)))):
        xs, ys = [], []
        for t, _ in rows:
            a = alarms.get(t)
            if a is not None and len(a):
                sel = a[a["real"] == real]
                xs += list(sel["day"])
                ys += [ypos[t] - 0.42] * len(sel)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name, marker=marker,
            legendgroup="alarm", legendgrouptitle_text="Model call-out",
            hovertemplate="%{x|%d %b %Y}<extra>alarm, " + name + "</extra>"))

    def count(t):
        """The row's alarm count, as its own label right of the row. Kept out
        of the axis label: a two-line tick label is drawn with its lines on top
        of each other at the dashboard's font size."""
        a = alarms.get(t)
        if a is None:
            return None
        r = int(a["real"].sum())
        return f"{len(a)} model call-out{'s' if len(a) != 1 else ''}, {r} in the abnormal period"

    for t, _ in rows:
        text = count(t)
        if text:
            # One line, a smaller font, centred on its row: two lines at the
            # chart font are taller than the gap between rows.
            fig.add_annotation(xref="paper", x=1.0, xanchor="left", xshift=12,
                               y=ypos[t], yref="y", yanchor="middle", text=text,
                               showarrow=False, align="left",
                               font=dict(size=STRIP_COUNT_FONT_PX))
    fig.update_layout(
        title="Every day of every turbine, and every alarm",
        height=200 + 95 * n,
        margin=dict(l=10, r=300 if alarms else 10, t=30, b=130),
        legend=dict(orientation="h", yref="container", y=0, yanchor="bottom",
                    x=0, xanchor="left"))
    fig.update_yaxes(tickmode="array", tickvals=[ypos[t] for t, _ in rows],
                     ticktext=[f"<b>{t}</b>" for t, _ in rows],
                     range=[-0.75, n - 0.5], showgrid=False, zeroline=False)
    return fig


def page_explore(series: dict | None) -> None:
    """Page 2: any month, any sensor, the four turbines side by side."""
    st.subheader("Explore: the four turbines, day by day")
    st.markdown("<div class='note'>Data: ENGIE's La Haute Borne wind farm in France, "
                "four Senvion MM82 turbines (2,050 kW), open SCADA readings every "
                "10 minutes, 2013 to 2018, joined to Copernicus ERA5 weather.</div>",
                unsafe_allow_html=True)
    if not series:
        st.info("results/daily_series.json is missing: run `make daily`.")
        return
    dv = _daily_view()
    tol = series["tolerance_k"]
    names = series["sensor_names"]
    all_t = list(series["turbines"])
    months = sorted({d[:7] for t in all_t for d in series["turbines"][t]["day"]})
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        chosen = st.multiselect("Turbines", all_t, default=all_t, key="ex_turbines")
    with c2:
        month = st.selectbox("Month", months,
                             index=months.index("2017-09") if "2017-09" in months else 0,
                             key="ex_month")
    with c3:
        sensors = [c for c in names if any(c in series["turbines"][t]["sensors"]
                                           for t in all_t)]
        sensor = st.selectbox("Sensor", sensors, format_func=lambda c: names[c],
                              key="ex_sensor")
    if not chosen:
        st.info("Pick at least one turbine.")
        return
    start = f"{month}-01"
    end = str((pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date())
    rows = [(t, dv.frame(series, t, sensor, start, end)) for t in chosen]
    # Alarms are found on the whole record and then cut to the month, so a
    # run of hot days that began last month is not counted as a new alarm.
    alarms = {}
    for t in chosen:
        a = dv.alarms(dv.frame(series, t, sensor), tol)
        alarms[t] = a[(a["day"] >= pd.Timestamp(start)) & (a["day"] <= pd.Timestamp(end))]
    _chart(_strip_figure(rows, tol, dv, alarms), width="stretch")
    st.markdown(
        "<div class='note'>The squares say what the sensor did: orange is a day "
        "hotter than predicted. The circles under a row are the model's call-outs: "
        "the first hot day of a run, when a technician is called out, whatever the "
        "temperature, so they can fall well below the fixed 80 °C limit. The circle's colour "
        "says where it fell: <b>filled red</b>, the day lies in the turbine's "
        "abnormal period; <b>open white</b>, it does not. The data have no "
        "inspection records, so the abnormal period (set in this project from the "
        "sibling turbines) stands in for what an inspection would find.</div>",
        unsafe_allow_html=True)

    cols = st.columns(len(chosen))
    for col, t in zip(cols, chosen):
        b = dv.badge(series, t, end, _month_of_readings(series))
        label = {"monitoring": "MONITORING", "frozen": "FAULT · MODEL FROZEN",
                 "hold": "MAINTENANCE HOLD", "none": "NOT CHECKED"}[b["kind"]]
        extra = ""
        if b["kind"] == "hold":
            extra = (f"<div class='hold-bar'><div style='width:{b['share']:.0%}'>"
                     "</div></div>")
        with col:
            st.markdown(
                f"<div class='kpi status-{b['kind']}'><div class='v'>{t}</div>"
                f"<div class='l'>{label}</div>{extra}"
                "<div style='font-size:.8rem;color:var(--muted-foreground)'>"
                f"{b['text']}</div></div>",
                unsafe_allow_html=True)

    c4, c5, c6 = st.columns([1, 2, 1])
    with c4:
        focus = st.selectbox("Plot turbine", chosen, key="ex_focus")
    with c5:
        leads = series["leads"] if sensor == series["primary"] else series["series_leads"]
        lead = st.select_slider("Forecast issued how many days before", options=leads,
                                value=leads[0], key="ex_lead")
    with c6:
        old_age = _older_model_control(series, "ex")
    df = dict(rows)[focus]
    if (df["status"] == "checked").any():
        # The scale of all the chosen turbines this month, so switching the
        # plotted turbine never changes how big the tolerance looks.
        y_range = _y_range([d for _, d in rows], tol, lead, old_age)
        _chart(_three_line_figure(df, tol, lead,
                                  f"{focus}: {names[sensor]}, {month} "
                                  f"(tolerance ±{tol:g} K)", y_range=y_range,
                                  alarms=alarms.get(focus), old_age=old_age,
                                  x_range=[pd.Timestamp(start) - pd.Timedelta(hours=12), pd.Timestamp(end) + pd.Timedelta(hours=12)]),
               width="stretch")
    else:
        st.info(f"{focus} has no checked day in {month}.")
    if sensor == series["primary"] and st.checkbox(
            "Compare with the siblings at the same moment", key="ex_sib"):
        import plotly.graph_objects as go

        fig = go.Figure()
        for t, d in rows:
            fig.add_trace(go.Scatter(x=d["day"], y=d["sibling"], mode="lines+markers",
                                     name=t))
        fig.add_hline(y=2.0, line_dash="dash", line_color=C_HOT,
                      annotation_text="+2 K")
        fig.update_layout(
            title="Change in the gap to the siblings (wind-direction aware, 5-day median)",
            height=460, margin=dict(l=10, r=10, t=30, b=110),
            legend=dict(orientation="h", yref="container", y=0, yanchor="bottom",
                        x=0, xanchor="left"))
        fig.update_yaxes(title_text="K")
        _chart(fig, width="stretch")
        st.markdown(
            "<div class='note'>A gap in a line is a day with nothing to compare: the "
            "turbine and its siblings did not run at similar power at the same "
            "moment for at least 6 hours, usually because the wind was too low or a "
            "turbine was stopped. They are mostly the days shown grey (ran too little to "
            "check) in the strip above.</div>", unsafe_allow_html=True)


def page_performance(dcj: dict | None, series: dict | None, live: bool) -> None:
    """Page 3: what the model adds to the fixed limit, why a frozen model is
    judged with last year's same month, the false alarms over the whole
    record, and R80711's timeline."""
    st.subheader("Performance over the whole record")
    _limit_vs_model_view(load_limit_vs_model())
    _frozen_fallback_view(load_frozen_fallback())
    if dcj and dcj.get("fleet"):
        _daily_check_view(dcj)
        _case_timeline(series)


def _forecast_accuracy_view(dcj: dict) -> None:
    import plotly.graph_objects as go

    rows = [r for r in dcj["by_turbine"] if r.get("forecast_accuracy")]
    if not rows:
        return
    leads = sorted({int(L) for r in rows for L in r["forecast_accuracy"]["by_lead"]})

    def mean(key, L):
        v = [r["forecast_accuracy"]["by_lead"][str(L)][key] for r in rows
             if str(L) in r["forecast_accuracy"]["by_lead"]]
        return sum(v) / len(v)

    same = sum(r["forecast_accuracy"]["same_day_mae_vs_sensor_k"] for r in rows) / len(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=leads, y=[mean("mae_vs_sensor_k", L) for L in leads],
                             mode="lines+markers", name="forecast vs sensor",
                             line=dict(color=C_FORECAST, width=3)))
    fig.add_trace(go.Scatter(x=leads, y=[mean("mae_vs_same_day_k", L) for L in leads],
                             mode="lines+markers", name="forecast vs on-day prediction",
                             line=dict(color=C_FORECAST, width=2, dash="dot")))
    fig.add_hline(y=same, line_dash="dash", line_color=C_MODEL,
                  annotation_text=f"on-day prediction vs sensor {same:.2f} K")
    fig.update_layout(title="How far ahead the bearing can be predicted (healthy days, "
                            "four turbines)", height=480,
                      margin=dict(l=10, r=10, t=30, b=110),
                      legend=dict(orientation="h", yref="container", y=0,
                                  yanchor="bottom", x=0, xanchor="left"))
    fig.update_xaxes(title_text="forecast issued, days before", dtick=1)
    fig.update_yaxes(title_text="mean absolute error (K)", rangemode="tozero")
    _chart(fig, width="stretch")


def _sibling_view(dcj: dict) -> None:
    sib = dcj.get("siblings")
    if not sib:
        return
    st.markdown("#### The sibling comparison, and the multi-sensor rule")
    rows = []
    for t, s in sib.items():
        for mode, label in (("direction", "per wind sector"), ("overall", "one normal")):
            b = s[mode]["by_threshold"]["2"]
            rows.append({"turbine": t, "normal gap": label,
                         "false call-outs / year at +2 K":
                             f"{b['false_call_outs_per_year']:.1f}",
                         "first call-out on the fault":
                             b["first_fault_call_out"] or "–"})
    for r in dcj["by_turbine"]:
        if r["tolerance"] == 2.0:
            rows.append({"turbine": r["turbine"],
                         "normal gap": "model: bearing sensor alone",
                         "false call-outs / year at +2 K":
                             f"{r['false_call_outs_per_year']:.1f}",
                         "first call-out on the fault":
                             r["first_fault_call_out"] or "–"})
        g = r.get("gearbox_rule")
        if g and r["tolerance"] == 2.0:
            rows.append({"turbine": r["turbine"],
                         "normal gap": "model: 2 of 4 gearbox sensors",
                         "false call-outs / year at +2 K":
                             f"{g['false_call_outs_per_year']:.1f}",
                         "first call-out on the fault":
                             g["first_fault_call_out"] or "–"})
    rows.sort(key=lambda r: r["turbine"])
    st.table(pd.DataFrame(rows))
    st.markdown(
        "<div class='note'>The siblings are compared at the same moment and at "
        "similar power, so weather and load cancel out. Learning the normal gap per "
        "wind sector was meant to remove a direction effect; on the healthy "
        "turbines it did not lower the false call-outs, and it called the fault "
        "later. Requiring two gearbox sensors to agree raised false call-outs. "
        "The bearing's own model stays the alarm; the siblings and the other "
        "sensors are a second opinion for the technician.</div>",
        unsafe_allow_html=True)


def _case_timeline(series: dict | None) -> None:
    """R80711 from May 2017: the daily difference from its healthy model."""
    if not series or "R80711" not in series["turbines"]:
        return
    import plotly.graph_objects as go

    dv = _daily_view()
    tol = series["tolerance_k"]
    df = dv.frame(series, "R80711", series["primary"], "2017-05-01", "2018-01-31")
    chk = df[df["status"] == "checked"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chk["day"], y=chk["diff"], name="sensor − prediction",
                         marker_color=[C_HOT if v > tol else "#7f8ea3"
                                       for v in chk["diff"]]))
    fig.add_trace(go.Scatter(x=df["day"], y=df["sibling"], mode="lines",
                             name="change against the siblings",
                             line=dict(color=C_FORECAST, width=2.5)))
    fig.add_hline(y=tol, line_dash="dash", line_color=C_HOT,
                  annotation_text=f"+{tol:g} K")
    for a, b in series["anomaly_spans"].get("R80711", []):
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(224,82,99,0.08)", line_width=0)
    fig.update_layout(title="R80711, May 2017 to January 2018: the fault period shaded",
                      height=520, margin=dict(l=10, r=10, t=30, b=110),
                      legend=dict(orientation="h", yref="container", y=0,
                                  yanchor="bottom", x=0, xanchor="left"))
    fig.update_yaxes(title_text="K")
    _chart(fig, width="stretch")


def page_evidence(engie, care_score, multiv, lgbm, ae_audit, audit, thermal,
                  live: bool) -> None:
    """Page 4: the research behind the pages before it."""
    _hero(engie, load_backtest(), load_snapshot(), load_operating_curve())
    e1, e2, e3, e4 = st.tabs(["Weather Forecast", "Benchmark Comparison",
                              "Engineering & Forensics", "Data Explorer"])
    dcj = load_daily_check()
    with e1:
        tab_forecast()
        if dcj and dcj.get("fleet"):
            _forecast_accuracy_view(dcj)
    with e2:
        tab_benchmark(engie, care_score, multiv, lgbm, ae_audit, audit, thermal)
    with e3:
        tab_forensics()
        if dcj and dcj.get("fleet"):
            _sibling_view(dcj)
    with e4:
        tab_explorer(live)


def _header() -> None:
    """The page header: what this dashboard is, before any tab says what it
    shows. The sidebar carries the brand; this carries the subject."""
    st.markdown(
        f"<div class='app-header'><div class='app-mark'>{_TURBINE_ICON_SVG}</div>"
        "<div class='app-text'><div class='app-title'>Drift-Resilient Wind "
        "Turbine Condition Monitoring</div>"
        "<div class='app-sub'>A healthy-turbine model, retrained every month, "
        "that flags a component running hot for its conditions before it "
        "reaches the alarm limit, and explains every alarm</div>"
        "<div class='chips'><span class='chip'>Python</span>"
        "<span class='chip'>pandas</span><span class='chip'>LightGBM</span>"
        "<span class='chip'>Plotly</span><span class='chip'>Streamlit</span>"
        "<span class='chip'>CI-verified</span></div></div>"
        "<a class='app-link' href='https://github.com/abhisakh' "
        "target='_blank' rel='noopener'>GitHub ↗</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def _hero(engie, backtest, snapshot, curve=None) -> None:
    """Above the evidence: why the healthy model must be retrained every
    month, and the turbine the demo is built on. Every figure is read from
    results/*.json through windmill.scorecard; none is computed here."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from windmill import scorecard

    card = scorecard.goal_scorecard(backtest, engie, snapshot, curve)
    ages = card["false_alarm"]["ratio_by_reference_age"]
    ratio = card["false_alarm"]["ratio_frozen_2014"]
    peak = card["r80711"]["peak_c"]
    peers = card["r80711"]["peer_max_c"]

    st.markdown("<div class='eyebrow'>Why the model is retrained every month: "
                "false alarms grow with its age</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, key, label in ((c1, "three_months", "3 months old"),
                            (c2, "one_year", "1 year old"),
                            (c3, "two_years", "2 years old")):
        with col:
            v = ages.get(key)
            kpi(f"{_fmt(v, 1)}×" if v else "—", f"False alarms, model {label}",
                "against one refreshed on the last 30 days, same threshold rule")
    with c4:
        kpi(f"{peak:.2f} °C" if peak is not None else "—",
            "R80711 bearing peak, 2017",
            (f"siblings peak at {peers:.2f} °C" if peers is not None else ""))
    if ratio:
        st.markdown(
            f"<div class='note'>A model frozen in 2014 raised {_fmt(ratio, 1)}× the "
            f"false alarms of one refreshed monthly: the same effect at its "
            f"extreme. That figure lets each model keep its own threshold rule; "
            f"the cards above hold the rule fixed.</div>",
            unsafe_allow_html=True)

def _footer() -> None:
    """Provenance and credit, at the foot of the page."""
    st.markdown(
        "<div class='app-footer'>Data: ENGIE La Haute Borne SCADA (CC BY 4.0) "
        "· CARE to Compare (CC BY-SA 4.0) · Copernicus ERA5 via open-meteo. "
        "Every figure on this page is read from <code>results/*.json</code>, "
        "generated by the pipeline in <code>scripts/pipeline/</code>. "
        "<a href='https://github.com/abhisakh' target='_blank' "
        "rel='noopener'>github.com/abhisakh</a></div>",
        unsafe_allow_html=True,
    )


def main():
    _style()

    live = live_available()
    engie = load_engie()
    care_score = load_care_score()
    multiv = load_multivariate()
    lgbm = load_lightgbm()
    ae_audit = load_ae_audit()
    audit = load_audit()
    thermal = load_thermal()

    # --- sidebar -----------------------------------------------------------
    with st.sidebar:
        st.markdown(
            f"<div class='brand'>{_TURBINE_ICON_SVG}"
            "<span>Windmill · Condition Monitoring</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("*Drift-resilient condition monitoring for wind turbines*")
        if live:
            st.markdown("<span class='badge live'>LIVE SCADA ENGINE (Local 19 GB)</span>",
                        unsafe_allow_html=True)
            st.markdown("The R80711 monitor reads the raw ENGIE SCADA directly.")
        else:
            st.markdown("<span class='badge cloud'>CLOUD SNAPSHOT MODE (Verified JSONs)</span>",
                        unsafe_allow_html=True)
            st.markdown(
                "The 19 GB of raw SCADA is not present. Everything here is read "
                "from the committed artefacts in `results/`.")
        st.markdown("---")
        st.markdown("**Data**")
        st.markdown(
            "- ENGIE La Haute Borne SCADA (Mendeley v2, CC BY 4.0)\n"
            "- EDP wind farm, Portugal: alarm log + failure logbook (Mendeley v2, CC BY 4.0)\n"
            "- CARE to Compare (Zenodo, CC BY-SA 4.0)\n"
            "- ERA5 reanalysis via open-meteo (no key)")
        st.markdown("---")
        st.markdown(
            "<div class='note'>Scored on CARE with the benchmark authors' own "
            "implementation.</div>",
            unsafe_allow_html=True)

    _header()

    series = load_daily_series()
    t1, t2, t3, t4 = st.tabs(["Demo", "Explore", "Performance", "Evidence & method"])
    with t1:
        page_demo(series)
    with t2:
        page_explore(series)
    with t3:
        page_performance(load_daily_check(), series, live)
    with t4:
        page_evidence(engie, care_score, multiv, lgbm, ae_audit, audit, thermal, live)

    _footer()


if __name__ == "__main__":
    main()
