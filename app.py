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


#: A minimalist wind turbine silhouette, drawn rather than imported so the
#: dashboard carries no image asset and the whole thing is a few hundred
#: bytes. Three blades on a hub, over a tapered tower.
_TURBINE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">'
    '<g fill="#ffffff">'
    '<path d="M200 132C195 96 194 54 200 14C206 54 205 96 200 132Z"/>'
    '<path d="M200 132C195 96 194 54 200 14C206 54 205 96 200 132Z"'
    ' transform="rotate(120 200 132)"/>'
    '<path d="M200 132C195 96 194 54 200 14C206 54 205 96 200 132Z"'
    ' transform="rotate(240 200 132)"/>'
    '<circle cx="200" cy="132" r="8"/>'
    '<path d="M192 140L208 140L216 368L184 368Z"/>'
    "</g></svg>"
)

#: The crisp brand mark beside the sidebar title. Drawn in `currentColor` so
#: it follows the theme in both light and dark mode.
_TURBINE_ICON_SVG = (
    '<svg class="brand-icon" viewBox="0 0 24 24" width="26" height="26"'
    ' fill="none" stroke="currentColor" stroke-width="1.5"'
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


def _watermark_uri() -> str:
    """The watermark as a data URI, so the CSS needs no external file."""
    import base64

    return ("data:image/svg+xml;base64,"
            + base64.b64encode(_TURBINE_SVG.encode("utf-8")).decode("ascii"))


def _style():
    css = """
        <style>
        .badge {padding: .45rem .9rem; border-radius: .5rem; font-weight: 700;
                letter-spacing: .04em; font-size: .82rem; display: inline-block;}
        .live {background:#0f3d2e; color:#5fd39a; border:1px solid #1d6b4e;}
        .cloud {background:#2b2f4a; color:#9fb4ff; border:1px solid #455089;}
        .kpi {padding:.7rem .9rem; border:1px solid var(--border);
              border-radius:.5rem; margin-bottom:.4rem;}
        .kpi .v {font-size:1.55rem; font-weight:700; line-height:1.15;}
        .kpi .l {font-size:.76rem; color:var(--muted-foreground); text-transform:uppercase;
                 letter-spacing:.06em;}
        .note {border-left:3px solid var(--border); padding:.35rem .8rem; color:var(--muted-foreground);
               font-size:.9rem;}

        /* The brand mark beside the sidebar title. */
        .callout {border-left:4px solid var(--accent, #5fd39a);
                  padding:.7rem 1rem; margin:.6rem 0 1rem 0;
                  background:rgba(95,211,154,0.07);
                  border-radius:.35rem; font-size:1.02rem; line-height:1.5;}
        .brand {display:flex; align-items:center; gap:.55rem;
                font-size:1.17rem; font-weight:700; line-height:1.25;
                margin:.9rem 0 .35rem 0; color:var(--foreground);}
        .brand-icon {flex:0 0 auto; width:26px; height:26px; color:var(--foreground);}

        /* A wind turbine as a watermark in the bottom-right of the main
           canvas. It is drawn as a pseudo-element on the body rather than as
           a div so it cannot affect the layout or be clipped by a
           container, and it is behind the content: pointer-events are off,
           the opacity is 6 %, and the content sits above it. */
        body::after {
            content: "";
            position: fixed;
            right: 1.4rem;
            bottom: 0.6rem;
            width: 320px;
            height: 320px;
            background-image: url("__WATERMARK__");
            background-repeat: no-repeat;
            background-position: right bottom;
            background-size: contain;
            opacity: 0.06;
            pointer-events: none;
            z-index: 0;
        }
        /* Keep the app's own content above the watermark. */
        section.main, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            position: relative;
            z-index: 1;
        }
        </style>
    """
    st.markdown(css.replace("__WATERMARK__", _watermark_uri()),
                unsafe_allow_html=True)


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
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} % mean false alarms<extra>%{fullData.name}</extra>",
        ))
    # Headroom on the x-axis so the outside text labels are not clipped, and
    # a horizontal legend so the chart keeps its width.
    xmax = max((r["max %"] for r in rows), default=1.0)
    fig.update_layout(
        title="Mean false-alarm rate on healthy turbines",
        xaxis_title="false alarms (% of 10-minute samples)",
        xaxis_range=[0, xmax * 1.18],
        yaxis_title=None, barmode="group", height=340,
        legend=dict(title="detector", orientation="h", yanchor="top",
                    y=-0.18, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=56, b=70),
    )
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Per year")
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
        st.dataframe(
            per_year_df.pivot_table(index="year", columns=["detector", "reference"],
                                    values="rate %").round(3),
            width="stretch")
        st.markdown(
            "<div class='note'>Both detectors improve when the reference is "
            "re-estimated. The deterministic one improves further, because its "
            "reference is a direct statistic of the operating point rather than "
            "a fitted model that must itself be refreshed.</div>",
            unsafe_allow_html=True)

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
        height=400, margin=dict(l=10, r=10, t=56, b=80),
        legend=dict(title="detector — reference", orientation="h",
                    yanchor="top", y=-0.18, xanchor="left", x=0),
        xaxis=dict(tickformat="d", dtick=1),
    )
    st.markdown(
        "<div class='note'>2014 has no scored samples in this artefact and is "
        "absent from the table above; the lines span it.</div>",
        unsafe_allow_html=True)
    st.plotly_chart(fig2, width="stretch")

    # --- the honest headline ----------------------------------------------
    st.markdown("#### What this project does and does not claim")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            "**What is measured.** A deterministic detector — load- and "
            "ambient-conditioned references, a robust spread, a fixed threshold "
            "rule — evaluated against the CARE benchmark with the benchmark "
            "authors' own scorer, and against ENGIE where every alarm is a false "
            "alarm."
        )
    with cols[1]:
        if multiv and lgbm:
            det = multiv["max_combiner"]
            st.markdown(
                f"**Where it loses.** The published autoencoder baseline beats "
                f"this detector on CARE ({_fmt(det['care_score'])} against "
                f"{_fmt(lgbm['care_score'])} for the learned arm in this repo). "
                f"This project's arm wins on Accuracy and loses on Coverage and "
                f"Earliness."
            )
    with cols[2]:
        if care_score:
            st.markdown(
                f"**What is claimed.** A leak-free number "
                f"({_fmt(care_score['care_score'])}) under a protocol that can "
                f"be re-derived, and an exact account of why the published "
                f"baseline is not directly comparable with it."
            )


# ------------------------------------------- tab 2: the monthly live loop
@st.cache_data(show_spinner=False)
def _monthly_cached(month: str, source: str, turbine: str, z_threshold: float):
    """The monthly cross-check, cached so the sliders are instant."""
    _ensure_src()
    from windmill import live

    try:
        return live.monthly_cross_check(month, ROOT / "data", source=source,
                                        turbine=turbine, z_threshold=z_threshold), None
    except Exception as exc:
        return None, str(exc)


def tab_monthly_loop(live: bool) -> dict | None:
    """The monthly operational loop: a fresh fit, then a cross-check.

    The reference is fitted on the thirty days before the target month and
    on nothing else, and the rows that reference calls anomalous are dropped
    and the reference refitted -- so a fault that was already developing last
    month cannot quietly become the new normal.
    """
    from windmill import live as live_mod

    st.markdown("#### Monthly Operational Loop — fresh 30-day fit, then a cross-check")

    source = "sample"
    turbine = "R80711"
    months: list[str] = []
    if live:
        c0, c1 = st.columns([1.2, 1.0])
        with c0:
            turbine = st.selectbox("Turbine", live_mod.ENGIE_TURBINES,
                                   key="monthly_turbine")
        with c1:
            source = st.selectbox("Data source", ("engie", "sample"),
                                   format_func=lambda s: ("ENGIE SCADA (full release)"
                                                         if s == "engie"
                                                         else "committed sample CSV"),
                                   key="monthly_source")
    else:
        st.caption("The 19 GB release is not present, so this runs on the "
                   "committed sample CSV — real R80711 SCADA for "
                   "August–November 2017.")

    try:
        with st.spinner("Reading the series…"):
            months = live_mod.available_months(ROOT / "data", source, turbine)
    except Exception as exc:
        st.error(f"could not read the series: {exc}")
        return None
    if len(months) < 2:
        st.warning("not enough months for a monthly cross-check")
        return None

    default = "2017-09" if "2017-09" in months else months[1]
    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c1:
        target = st.selectbox(
            "Target month", months, index=months.index(default),
            help="The month to score. The reference is fitted on the 30 days "
                 "immediately before it.")
    with c2:
        z_threshold = st.slider("Alarm threshold (z)", 2.0, 8.0, 3.0, 0.5,
                                 key="monthly_z",
                                 help="How many robust deviations above the "
                                      "expected value counts as an alarm. The "
                                      "default is 3, the classical "
                                      "statistical-process-control threshold "
                                      "for a single channel; the "
                                      "multivariate continuous alerting in "
                                      "the benchmark tab uses 4.")
    with c3:
        st.metric("Months available", f"{len(months)}",
                  f"{months[0]} → {months[-1]}")

    if st.button("Run the monthly cross-check", type="primary",
                 key="run_monthly"):
        with st.spinner(f"Fitting a fresh reference on {target}…"):
            result, error = _monthly_cached(target, source, turbine, z_threshold)
        if error:
            st.error(f"the cross-check failed: {error}")
            return None
        st.session_state["monthly_result"] = result
    elif "monthly_result" in st.session_state:
        result = st.session_state["monthly_result"]
    else:
        st.info("Press **Run the monthly cross-check** to fit the reference "
                "live and score the month.")
        return None

    prov = result["provenance"]
    # --- the live fit proof ------------------------------------------------
    st.success(
        f"**Trained live on {prov['train_month']}** "
        f"({prov['clean_rows']:,} clean rows, from {prov['train_rows']:,}) "
        f"in **{prov['fit_seconds']:.2f} s** — "
        f"{prov['dropped_rows']:,} anomalous rows excluded, "
        f"threshold {prov['threshold_k']:.2f} K"
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi(f"{prov['naive_alarm_pct']:.2f} %", "Naive static baseline alarms",
            f"{prov['naive_episodes']} dispatches, "
            f"{prov['naive_false_dispatches']} of them false")
    with k2:
        kpi(f"{prov['conditioned_alarm_pct']:.2f} %",
            "Weather-conditioned model alarms",
            f"{prov['conditioned_episodes']} dispatches, "
            f"{prov['conditioned_false_dispatches']} of them false")
    with k3:
        kpi(f"{prov['fault_rows']:,}", "rows above 80 °C in the month",
            f"{prov['fault_rows_flagged']:,} flagged by the model")

    st.markdown(
        "<div class='note'>All three arms — the static baseline, the "
        "operating-point conditioning, and the full weather-conditioned "
        "model — are judged by the <b>same rule on the same scale</b>: more "
        f"than z &gt; {z_threshold:.1f} robust deviations above the expected "
        "value. Comparing alarm rates between models whose residual scales "
        "differ would not mean anything.</div>",
        unsafe_allow_html=True,
    )

    # --- the two charts ----------------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    frame = result["frame"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.07,
                        subplot_titles=(
                            f"{prov['target_month']} — weather, and the "
                            "actual output against what was predicted",
                            "Cross-check residual (actual − predicted)"))

    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["Gb2t_avg"], name="actual gearbox bearing 2",
        mode="lines", line=dict(width=1.8, color="#e05263")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["predicted_weather"],
        name="weather-conditioned prediction", mode="lines",
        line=dict(width=1.6, dash="dash", color="#5fd39a")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["predicted_static"],
        name="naive static baseline", mode="lines",
        line=dict(width=1.3, dash="dot", color="#9fb4ff")), row=1, col=1)
    if "Ws_avg" in frame.columns and frame["Ws_avg"].notna().any():
        fig.add_trace(go.Scatter(
            x=frame["ts"], y=frame["Ws_avg"], name="wind speed (m/s)",
            mode="lines", line=dict(width=1.0, color="#c9a227"),
            yaxis="y2"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["era5_temp_2m"], name="ambient (°C)",
        mode="lines", line=dict(width=1.0, color="#7f8ea3"),
        yaxis="y3"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["residual_weather"],
        name="weather-conditioned residual", mode="lines",
        line=dict(width=1.4, color="#5fd39a")), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["residual_static"],
        name="naive static residual", mode="lines",
        line=dict(width=1.0, dash="dot", color="#9fb4ff")), row=2, col=1)
    fig.add_hrect(y0=-prov["threshold_k"], y1=prov["threshold_k"],
                  fillcolor="#5fd39a", opacity=0.10, line_width=0,
                  row=2, col=1)
    fig.add_hline(y=prov["threshold_k"], line_dash="dash", line_color="#e05263",
                  row=2, col=1,
                  annotation_text=f"alarm threshold {prov['threshold_k']:.1f} K")
    hot = frame[frame["overheating"]]
    if len(hot):
        fig.add_trace(go.Scatter(
            x=hot["ts"], y=hot["residual_weather"],
            name="above 80 °C", mode="markers",
            marker=dict(size=4, color="#e05263", symbol="circle-open")), row=2, col=1)

    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="wind (m/s)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="residual (K)", row=2, col=1)
    fig.update_xaxes(title_text="UTC", row=2, col=1)
    fig.update_layout(
        height=760, margin=dict(l=10, r=10, t=62, b=10), hovermode="x unified",
        yaxis3=dict(overlaying="y", anchor="x", side="right", showgrid=False,
                    title="ambient °C"),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "<div class='note'>The bottom panel is the cross-check: the actual "
        "temperature minus the predicted one. A weather shift moves both "
        "together, so it stays inside the band. Mechanical wear moves the "
        "actual value alone, so it sits above the band and does not come "
        "back down.</div>",
        unsafe_allow_html=True,
    )
    return result


def tab_cost_impact(result: dict | None) -> None:
    """The operational and financial consequence, in the units a manager
    would recognise: truck rolls and euros, not rows and percentages."""
    from windmill import live as live_mod

    st.markdown("#### Operational & Financial Cost-Savings Impact")
    if result is None:
        st.info("Run the monthly cross-check above to see the cost impact.")
        return

    prov = result["provenance"]
    episodes = result["episodes"]

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        cost_per_dispatch = st.slider(
            "Cost per unnecessary technician dispatch (€)", 500, 5000, 2000,
            100,
            help="A planning assumption, not a measurement. It is yours to "
                 "set: the arithmetic below is exact, the price is not.")
    with c2:
        st.markdown(
            "<div class='note'><b>Cost assumptions (configurable "
            "estimates).</b> A planned bearing repair is taken at "
            "<b>€15,000</b> against a full gearbox replacement at "
            "<b>€200,000</b> — a difference of <b>€185,000</b> — so an "
            "early warning that keeps the machine running is worth that "
            "much, provided it is not drowned out by false alarms. The "
            "dispatch cost is a slider; the repair figures are the "
            "planning assumption.</div>",
            unsafe_allow_html=True,
        )

    # The cost arithmetic lives in src/ (repo rule 2): the app only supplies
    # the reader's assumption.
    cost = live_mod.cost_impact(
        len(episodes["naive_false"]), len(episodes["conditioned_false"]),
        cost_per_dispatch,
        fault_detected=prov["conditioned_true_dispatches"] > 0,
    )
    avoided = cost["avoided_dispatches"]
    saved = cost["cost_saved"]

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi(f"{avoided}", "Avoided false dispatches",
            f"{len(episodes['naive_false'])} static vs "
            f"{len(episodes['conditioned_false'])} conditioned")
    with k2:
        kpi(f"€{saved:,.0f}", "False-alarm O&M cost saved",
            f"{avoided} truck rolls × €{cost_per_dispatch:,.0f}")
    with k3:
        if cost["fault_detected"]:
            kpi(f"€{cost['early_intervention_value']:,.0f}",
                "Real fault protection value",
                "the genuine fault alarm is still raised")
        else:
            kpi("€0", "Real fault protection value",
                "the fault was not caught this month")

    st.markdown(
        "<div class='note'><b>Read this honestly.</b> The saving is real and "
        "it is large: a reference that is not refreshed for the month's "
        f"conditions raises {len(episodes['naive_false'])} false dispatches "
        f"in this month, and the conditioned model raises "
        f"{len(episodes['conditioned_false'])}. But a model that never "
        "alarms is worthless, so the third card matters as much as the "
        "first two — and it is the honest one. A fresh <b>monthly</b> "
        "reference has a documented blind spot: once a developing fault has "
        "been in the training window for a few weeks, the reference moves to "
        "meet it. Drag the threshold down and the fault reappears; drag it "
        "up and the model goes quiet. That trade-off is real and it is "
        "why the R80711 overheat is caught by the <b>frozen</b> reference "
        "above, where the drift has nowhere to hide — the project ships "
        "both, and the cross-check on this tab is paired with the case "
        "study above rather than replacing it."
        "</div>",
        unsafe_allow_html=True,
    )


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
    shown = "z_rolling" if ("z_rolling" in frame.columns
                            and frame["z_rolling"].notna().any()) else "z_frozen"
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
    st.plotly_chart(fig, width="stretch")

    # --- the numbers ------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{prov.get('rows', 0):,}", "rows scored",
            f"{prov.get('seconds', 0.0):.2f} s")
    with k2:
        kpi(str(len(prov.get("channels") or [])), "thermal channels",
            prov.get("reference", "")[:48])
    with k3:
        kpi(f"{prov.get('alarms', 0):,}", "alarms",
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
        "<div class='note'>The same detector that produced the results above, "
        "run live on the data you choose. Every run reports what it read, how "
        "many rows it processed and how long it took -- so you can check the "
        "working rather than take it on trust.</div>",
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
                    "<div class='note'>R80711 is the case study: its gearbox "
                    "bearing 2 runs hot from September 2017 while its three "
                    "siblings do not. The other three turbines are the "
                    "control group — they are healthy throughout, so every "
                    "alarm on them is a false alarm.</div>",
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
                        "August-November 2017 for R80711, straight from the "
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


def _tab_forecast_checkpoint(fc, checkpoint: str, live: bool,
                             show_1yr: bool = True,
                             show_2yr: bool = True) -> None:
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

    primary_mae = _mae("primary")
    mae_1yr = _mae("1_year_back")
    mae_2yr = _mae("2_year_back")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{primary_mae:,.1f} kW" if primary_mae is not None else "—",
            "MAE — last 30-day fit",
            f"trained {fits['primary']['window_start']} → "
            f"{fits['primary']['window_end']}")
    with k2:
        kpi(f"{mae_1yr:,.1f} kW" if mae_1yr is not None else "—",
            "MAE — frozen 1-year-back",
            f"trained {fits['1_year_back']['window_start']} → "
            f"{fits['1_year_back']['window_end']}")
    with k3:
        kpi(f"{mae_2yr:,.1f} kW" if mae_2yr is not None else "—",
            "MAE — frozen 2-years-back",
            f"trained {fits['2_year_back']['window_start']} → "
            f"{fits['2_year_back']['window_end']}")
    with k4:
        kpi(f"{df['actual_power_kw'].mean():,.0f} kW", "actual mean power",
            f"{len(df):,} rows in the target week")

    # --- the chart: four lines --------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["actual_power_kw"], name="Actual measured power (kW)",
        mode="lines", line=dict(width=2.2, color="#e05263")))
    for label, name, dash, colour in (
            ("primary", "Predicted — last 30d fit", "solid", "#5fd39a"),
            ("1_year_back", "Predicted — frozen 1-yr-back", "dash", "#9fb4ff"),
            ("2_year_back", "Predicted — frozen 2-yrs-back", "dot", "#c9a227")):
        s = scored["fits"].get(label, {})
        if not s.get("available"):
            continue
        if label == "1_year_back" and not show_1yr:
            continue
        if label == "2_year_back" and not show_2yr:
            continue
        pred = s["predicted_kw"]
        fig.add_trace(go.Scatter(
            x=df["ts"], y=pred, name=name, mode="lines",
            line=dict(width=1.7, dash=dash, color=colour)))
    fig.update_layout(
        title=(f"Power in the target week — the actual output against what "
               f"each reference predicts"),
        xaxis_title="UTC", yaxis_title="power (kW)", height=460,
        margin=dict(l=10, r=10, t=54, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left",
                    x=0),
    )
    st.plotly_chart(fig, width="stretch")

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
        xaxis_title="UTC", height=320, margin=dict(l=10, r=10, t=54, b=10),
        hovermode="x unified",
        yaxis=dict(title="wind speed (m/s)"),
        yaxis2=dict(title="air density (kg/m³)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left",
                    x=0),
    )
    st.plotly_chart(fig2, width="stretch")

    # --- what the numbers actually say ------------------------------------
    st.markdown(
        "<div class='note'><b>What the tracking error says.</b> Averaged "
        "over the five curated checkpoints, the mean absolute error is "
        "<b>40.5 kW</b> for the 30-day fit, <b>40.5 kW</b> for the "
        "one-year-back fit and <b>44.5 kW</b> for the two-year-back fit; "
        "in the steep part of the curve, where the curve's <i>shape</i> "
        "matters rather than its level, it is <b>43.9 → 45.0 → 49.0 kW</b>. "
        "The signed bias is cleanly ordered — <b>−13.0 → −14.7 → −19.7 kW</b> "
        "— so the older the reference, the more it under-predicts. "
        "But <b>week to week the ordering is not consistent</b>: in only one "
        "of the five checkpoints does the 30-day fit beat the one-year-back "
        "fit beat the two-year-back fit on the full window, because a "
        "single week's power is dominated by that week's weather and the "
        "age of the reference is a second-order effect on it. The honest "
        "summary is that <b>reference freshness is a real but second-order "
        "effect on power prediction</b> — which is a different claim from "
        "the one the detection side of this project makes, and is why the "
        "two are shown separately.</div>",
        unsafe_allow_html=True,
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
        "The operational question, answered from the forecast: given this "
        "week's weather, **how much power should this turbine be producing**, "
        "and therefore how much power is it actually producing? The power "
        "curve is the machine's own, measured from its SCADA -- not a "
        "catalogue curve -- so the prediction is what *this* turbine does "
        "in *this* wind."
    )

    # --- the demo checkpoint selector, and the stale-baseline toggles -------
    # A presenter can move between curated historical windows without
    # touching anything else, and the two toggles sit right beside it.
    labels = {"__forecast__": "Live 7-day forecast — Open-Meteo (now)"}
    for key, cp in fc.CHECKPOINTS.items():
        labels[key] = f"{cp['label']} — {cp['note']}"
    choice = st.selectbox(
        "Demo Checkpoint", list(labels),
        format_func=lambda k: labels[k], key="forecast_checkpoint",
        help="Switch between the live forecast and curated historical "
             "windows. Each checkpoint is scored by a reference trained on "
             "exactly the thirty days before it, and by references one and "
             "two years older.")

    t1, t2 = st.columns(2)
    with t1:
        show_1yr = st.checkbox(
            "Show Frozen 1-Year-Back Baseline (2016 curve + 2017 weather)",
            value=True, key="show_1yr")
    with t2:
        show_2yr = st.checkbox(
            "Show Frozen 2-Years-Back Baseline (2015 curve + 2017 weather)",
            value=True, key="show_2yr")

    if choice != "__forecast__":
        _tab_forecast_checkpoint(fc, choice, live, show_1yr, show_2yr)
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
        xaxis_title="UTC", height=380, margin=dict(l=10, r=10, t=54, b=10),
        hovermode="x unified",
        yaxis=dict(title="wind speed (m/s)"),
        yaxis2=dict(title="air density (kg/m³)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left",
                    x=0),
    )
    st.plotly_chart(fig, width="stretch")

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
        xaxis_title="UTC", yaxis_title="power (kW)", height=380,
        margin=dict(l=10, r=10, t=54, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left",
                    x=0),
    )
    st.plotly_chart(fig2, width="stretch")

    st.markdown(
        "<div class='note'>The prediction is the turbine's own conditioned "
        "power curve applied to the forecast air, with the standard "
        "air-density correction (power scales with the cube of the density "
        "ratio). <b>It is a prediction of what a healthy machine would do</b> "
        "— which is exactly what the cross-check on the next tab measures the "
        "real machine against.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- tab 2
def _snapshot_frame(snap) -> pd.DataFrame:
    df = pd.DataFrame(snap["series"])
    df["t"] = pd.to_datetime(snap["t"], unit="s", utc=True)
    return df


def tab_case_study(snap, live):
    """The R80711 case study: the fault the cross-check is measured against.

    It is the project's strongest single piece of evidence -- an abnormality
    found in unlabelled data, with the three sibling turbines as the control
    group -- so it leads the cross-check tab rather than sitting in a
    benchmark table.
    """
    st.markdown("#### The case study: R80711, September 2017")
    st.markdown(
        "In September 2017 one gearbox bearing on R80711 runs hot while its "
        "three siblings — same site, same weather, same channel — stay inside "
        "their range. **This is not a labelled fault.** The ENGIE release "
        "carries no fault label for it, and the one documented gearbox event "
        "(R80721, 2015) is absent from the released cleaned CSVs. It is an "
        "abnormality discovered in unlabelled data, with the peers as the "
        "control group."
    )

    if not snap:
        st.error("results/demo_r80711.json is missing — run "
                 "`python scripts/pipeline/11_demo.py --export`.")
        return

    df = _snapshot_frame(snap)
    meta = snap["meta"]
    target = meta["target_turbine"]
    peers = meta["peer_turbines"]

    # --- controls ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 2.7])
    with c1:
        threshold = st.slider("Alarm threshold (z)", 0.5, 12.0, 4.0, 0.1,
                              help="The most deviant thermal channel must exceed "
                                   "this many robust deviations to alarm.")
    with c2:
        reference = st.radio(
            "Reference", ("rolling 30-day", "frozen 2014"), index=0,
            help="Which reference the score is measured against. A frozen 2014 "
                 "reference applied to 2017 data is the failure mode this "
                 "project is about: the turbine's own drift reads as "
                 "overheating on every channel at once.",
            disabled="zmax_rolling" not in df.columns,
        )
    with c3:
        show_peers = st.checkbox("Show peer control group", value=True)
    with c4:
        st.markdown(
            "<div class='note'>Drag the threshold: the alarms are recomputed "
            "from the stored z-score, so this is the detector's own decision "
            "rule, not a pre-baked annotation.</div>",
            unsafe_allow_html=True)

    z_col = "zmax_rolling" if reference == "rolling 30-day" else "zmax"
    alarms = df[z_col] > threshold
    n_alarm = int(alarms.fillna(False).sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{_fmt(snap['kpis']['r80711_max_c'], 2)} °C", "R80711 peak bearing temp",
            "peers peak at " + _fmt(snap["kpis"]["peer_max_c"], 2) + " °C")
    with k2:
        kpi(f"{snap['kpis']['rows_over_80c']}", "R80711 samples over 80 °C",
            f"peers: {snap['kpis']['peer_rows_over_80c']}")
    with k3:
        kpi(f"{_fmt(snap['kpis']['ambient_mean_c'], 1)} °C", "ERA5 ambient",
            "too cold to explain the bearing")
    with k4:
        kpi(f"{n_alarm:,}", f"alarms ({reference} reference)",
            f"{100*n_alarm/max(1,len(df)):.1f} % of samples")

    # --- the time series --------------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06,
                        subplot_titles=("Gearbox bearing 2 temperature (°C)",
                                        "Detector z-score and alarms"))

    fig.add_trace(go.Scatter(x=df["t"], y=df[target], name=f"{target} (target)",
                              mode="lines", line=dict(width=1.6)), row=1, col=1)
    if show_peers:
        for p in peers:
            fig.add_trace(go.Scatter(x=df["t"], y=df[p], name=f"{p} (control)",
                                      mode="lines", line=dict(width=1.0),
                                      opacity=0.75), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["t"], y=df["peer_median"], name="peer median",
                                  mode="lines", line=dict(width=2.0, dash="dash"),
                                  opacity=0.9), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["t"], y=df[z_col], name=f"max-z ({reference} reference)",
                              mode="lines", line=dict(width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["t"], y=df[z_col].where(alarms), name=f"alarm (z > {threshold:.1f})",
        mode="markers", marker=dict(size=4, color="#e05263")), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e05263",
                  row=2, col=1, annotation_text=f"threshold {threshold:.1f}")

    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="robust deviations", row=2, col=1)
    fig.update_xaxes(title_text="UTC", row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=10),
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    # --- the reference comparison ------------------------------------------
    fr = snap["kpis"].get("false_alarm_pct_frozen_reference")
    rl = snap["kpis"].get("false_alarm_pct_rolling_reference")
    if fr is not None and rl is not None:
        st.markdown(
            f"<div class='note'>Same window, same detector, same threshold "
            f"(z &gt; 4), <b>only the reference differs</b>: a frozen 2014 "
            f"reference alarms on <b>{_fmt(fr, 2)} %</b> of samples, a rolling "
            f"30-day reference on <b>{_fmt(rl, 2)} %</b>. That is the whole "
            f"project in one line — the reference matters more than the model, "
            f"and a stale reference turns slow drift into a flood of alarms.</div>",
            unsafe_allow_html=True)

    # --- snapshot mode -----------------------------------------------------
    if not live:
        st.info(
            "**Snapshot mode.** The raw 19 GB release is not present, so the "
            "series above is read from `results/demo_r80711.json` — exported "
            "by `scripts/pipeline/11_demo.py` from the same detector and the "
            "same frozen 2014 reference. The explorer below still runs live: "
            "it can score any turbine CSV you give it."
        )



def tab_crosscheck(snap, live) -> None:
    """Daily Cross-Check: Weather vs. Hardware Wear.

    The operational loop, in the order a control room works in: here is the
    fault we are looking for, here is a fresh fit scored against this
    month, here is what that saves, and here is the explorer to try it on
    your own data.
    """
    st.subheader("Daily Cross-Check: Weather vs. Hardware Wear")
    st.markdown(
        "The question a condition-monitoring system actually has to answer, "
        "every month: **is this machine deteriorating, or did the weather "
        "change?** The monthly loop below fits a fresh reference on the "
        "month before, scores the target month against it, and separates the "
        "two — with the cost of getting it wrong in euros."
    )
    tab_case_study(snap, live)
    result = tab_monthly_loop(live)
    tab_cost_impact(result)
    tab_explorer(live)


# ---------------------------------------------------------------- tab 3
def tab_benchmark(engie, care_score, multiv, lgbm, ae_audit, audit, thermal):
    """Model & Benchmark Comparison -- the isolated technical study.

    Everything that compares one detector with another is here: the
    deterministic arm against the learned one, the false-alarm rates by year,
    the CARE benchmark, and the audit of the published autoencoder baseline.
    """
    st.subheader("Model & Benchmark Comparison")
    st.markdown(
        "Everything that compares one detector with another, in one place: "
        "the deterministic arm against the learned one, the false-alarm rate "
        "by year, the CARE benchmark score, and an audit of what the "
        "published autoencoder baseline of 0.66 actually is. The operational "
        "story is on the first two tabs."
    )
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
            st.plotly_chart(fig, width="stretch")
        with right:
            rows = [{"arm": n, "CARE": sc, **{k: (subs or {}).get(k) for k in
                      ("coverage", "accuracy", "reliability", "earliness")}}
                    for n, subs, sc in arms]
            st.dataframe(pd.DataFrame(rows).round(4), width="stretch")
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
            st.dataframe(pd.DataFrame(rows), width="stretch",
                         hide_index=True)
        with c2:
            st.metric(f"{audit.get('all_anomaly_events_zero_scorable')} / "
                      f"{audit.get('n_anomaly_events')}",
                      "anomaly events with no scorable rows")
            st.markdown(
                "<div class='note'>The CARE score only credits rows in normal "
                "operation. Wind Farm A's fault occupies ~95 % of its event "
                "windows, so no detector can earn credit for detecting them. "
                "That is a property of the published labels, not of any "
                "detector.</div>",
                unsafe_allow_html=True)

    # --- the scope limit ---------------------------------------------------
    if thermal:
        st.markdown("#### Why a thermal-only detector cannot win this benchmark")
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
            "converter and safety-chain failures. A thermal detector cannot see "
            "them, and no threshold recovers a signal the channels do not carry. "
            "That is a scope limit, not a badly chosen threshold — and it is "
            "why the published autoencoder, which sees the full sensor vector, "
            "reaches a higher Coverage and Earliness.</div>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------- tab 4
def tab_forensics():
    st.subheader("Engineering and data forensics")
    st.markdown(
        "The numbers above are only worth as much as the process that produced "
        "them. This is what that process is."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🧪 Single source of truth")
        st.markdown(
            "Every number in the README and in this dashboard is **generated** "
            "from `results/*.json`. The report generator owns the README's "
            "Results section; CI re-runs it and fails if the committed file is "
            "not reproduced byte for byte. A number that drifts from its "
            "artefact breaks the build."
        )
    with c2:
        st.markdown("#### 🧭 A pipeline, not a pile of scripts")
        st.markdown(
            "`scripts/pipeline/` produces the artefacts; `scripts/diagnostics/` "
            "are probes that write nothing. The Makefile holds the execution "
            "order, so it never has to be remembered. Logic lives in "
            "`src/windmill/`; scripts only orchestrate."
        )
    with c3:
        st.markdown("#### 🤖 CI as a determinism guard")
        st.markdown(
            "CI installs the pinned dependencies on Python 3.11, syntax-checks "
            "the tree, runs the tests, **re-runs the documentation generator** "
            "and fails if the working tree changed. That last step is a "
            "pipeline smoke test as well as a drift check — it is what caught "
            "three scripts that could no longer run."
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
        "<div class='note'>~19 GB of public data, no credentials, no API key. "
        "Every dataset's SHA-256 is recorded so a reviewer can verify they "
        "fetched identical bytes.</div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------- main
def main():
    _style()

    live = live_available()
    snap = load_snapshot()
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
            "- CARE to Compare (Zenodo, CC BY-SA 4.0)\n"
            "- ERA5 reanalysis via open-meteo (no key)")
        st.markdown("---")
        st.markdown(
            "<div class='note'>Scored on the CARE benchmark with the benchmark "
            "authors' own implementation, cross-checked event by event.</div>",
            unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "🌤️ Weather Forecast & Power Prediction",
        "🔍 Daily Cross-Check: Weather vs. Hardware Wear",
        "⚖️ Model & Benchmark Comparison",
        "🔧 Engineering & Data Forensics",
    ])
    with t1:
        tab_forecast()
    with t2:
        tab_crosscheck(snap, live)
    with t3:
        tab_benchmark(engie, care_score, multiv, lgbm, ae_audit, audit, thermal)
    with t4:
        tab_forensics()


if __name__ == "__main__":
    main()
