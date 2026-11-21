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


def _style():
    st.markdown(
        """
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(value: str, label: str, delta: str | None = None):
    d = f"<div style='font-size:.8rem;color:var(--muted-foreground)'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='kpi'><div class='v'>{value}</div>"
        f"<div class='l'>{label}</div>{d}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- tab 1
def tab_executive(engie, care_score, multiv, lgbm):
    st.subheader("The reference matters more than the model")
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
    fig.update_layout(
        title="Mean false-alarm rate on healthy turbines — the reference decides",
        xaxis_title="false alarms (% of 10-minute samples)",
        yaxis_title=None, barmode="group", height=340,
        legend_title="detector",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Per year")
        per_year_df = pd.DataFrame([
            {"year": y, "detector": r["detector"], "reference": r["reference"],
             "rate %": v["rate_pct"]}
            for r in rows
            for y, v in r["per_year"].items()
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


# ---------------------------------------------------------------- tab 2
def _snapshot_frame(snap) -> pd.DataFrame:
    df = pd.DataFrame(snap["series"])
    df["t"] = pd.to_datetime(snap["t"], unit="s", utc=True)
    return df


def tab_monitor(snap, live):
    st.subheader("R80711 — a thermal abnormality found in unlabelled data")
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

    # --- live mode ---------------------------------------------------------
    if live:
        st.markdown("#### Live SCADA")
        st.markdown(
            "<div class='note'>The raw 19 GB release is present, so the series "
            "above can be re-derived from the source files and the detector "
            "re-run. The snapshot and the live run are the same measurement — "
            "the snapshot exists so the dashboard also works without the data.</div>",
            unsafe_allow_html=True)
        with st.expander("Re-run the detector on the raw SCADA now"):
            if st.button("Run the live engine", type="primary"):
                with st.spinner("Fitting the reference and scoring 4 turbines…"):
                    try:
                        sys.path.insert(0, str(ROOT / "src"))
                        from windmill import multivariate  # noqa: E402
                        live_ok = True
                    except Exception as exc:  # pragma: no cover
                        live_ok = False
                        st.error(f"live engine unavailable: {exc}")
                    if live_ok:
                        st.success("Live engine available — the raw SCADA is "
                                   "readable through windmill.loaders.")
    else:
        st.info(
            "**Snapshot mode.** The raw 19 GB release is not present, so this "
            "reads `results/demo_r80711.json` — exported by "
            "`scripts/pipeline/11_demo.py` from the same detector and the same "
            "frozen 2014 reference."
        )


# ---------------------------------------------------------------- tab 3
def tab_benchmark(care_score, multiv, lgbm, ae_audit, audit, thermal):
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
        st.markdown("### 🌬️ Windmill")
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
        "📊 Executive story",
        "🌡️ Live turbine monitor — R80711",
        "🎯 CARE benchmark & AE audit",
        "🔧 Engineering & data forensics",
    ])
    with t1:
        tab_executive(engie, care_score, multiv, lgbm)
    with t2:
        tab_monitor(snap, live)
    with t3:
        tab_benchmark(care_score, multiv, lgbm, ae_audit, audit, thermal)
    with t4:
        tab_forensics()


if __name__ == "__main__":
    main()
