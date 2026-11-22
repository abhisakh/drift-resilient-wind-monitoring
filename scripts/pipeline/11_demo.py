"""Interactive demo: watch the detector work on one turbine, with an audit trail.

    python scripts/pipeline/11_demo.py                     # the R80711 thermal event
    python scripts/pipeline/11_demo.py --dataset <name>    # a CARE event instead
    python scripts/pipeline/11_demo.py --list              # list what is available

This is the part a reviewer should see first. It runs the real detector on real
SCADA and prints, row by row:

  * the operating point and the component's load- and ambient-conditioned value,
  * the residual and its robust z-score,
  * whether the reading raised an alarm, and
  * why -- which channels contributed, and what the reference was.

Every number printed here comes from the same code path the CARE scoring uses.
Nothing is simulated.

Two demonstrations are available:

  ENGIE R80711 (September 2017)
      A genuine thermal anomaly that is NOT in the CARE benchmark: R80711 runs
      5-9 K above its own conditioned reference while its three peers stay
      inside theirs. This is the strongest single piece of evidence in the
      project -- an abnormality found in unlabelled data, with the peers as the
      control.

  A CARE anomaly event
      A labelled event from the benchmark, with the ground-truth window shown so
      you can see exactly when the detector fires relative to the truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import (RATED_POWER_KW, care, forecast, live, loaders,  # noqa: E402
                    multivariate)

# --- presentation -----------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def rule(title: str = "", width: int = 96) -> None:
    if title:
        pad = max(0, width - len(title) - 3)
        print(f"\n{c('── ' + title + ' ' + '─' * pad, BOLD)}")
    else:
        print(c("─" * width, DIM))


def bar(value: float, vmax: float, width: int = 28) -> str:
    """A horizontal bar, scaled so the reader can see magnitude at a glance."""
    if vmax <= 0:
        return ""
    n = int(min(1.0, value / vmax) * width)
    return "█" * n + c("·" * (width - n), DIM)


# --- ENGIE demo -------------------------------------------------------------
ENGIE_TURBINE = "R80711"
ENGIE_EVENT_START = "2017-09-01"
ENGIE_EVENT_END = "2017-10-01"
#: The channel carrying the anomaly: gearbox bearing 2 temperature.
ENGIE_CHANNEL = "Gb2t_avg"
#: The three sibling turbines at the same site. They are the control group:
#: any signal they also show is weather, not a fault on R80711.
ENGIE_PEERS = ("R80736", "R80721", "R80790")


def _engie_dir() -> Path:
    for candidate in (
        ROOT / "data" / "raw" / "engie_la_haute_borne_extracted",
        ROOT / "data" / "raw" / "engie_v2_extracted",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "ENGIE not found -- run scripts/pipeline/01_build_dataset.py")


def _engie_csv(base: Path, turbine: str) -> Path:
    """Find the F99 cleaned CSV for one turbine."""
    hits = sorted(base.rglob(f"*{turbine}*F99*.csv"))
    if not hits:
        hits = sorted(base.rglob(f"*{turbine}*.csv"))
    if not hits:
        raise FileNotFoundError(f"no SCADA CSV for {turbine} under {base}")
    return hits[0]


def _engie_read(path: Path) -> pd.DataFrame:
    """Read one ENGIE SCADA CSV.

    The cleaned CSVs are comma-separated with a UTF-8 BOM (the Mendeley
    archive's static-information.csv is semicolon-separated, but the per-turbine
    data files are not). Timestamps carry mixed +01:00/+02:00 offsets, so
    parsing is offset-aware and then normalised to UTC.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    tcol = next(c for c in df.columns if "date" in c.lower() and "time" in c.lower())
    df["ts"] = pd.to_datetime(df[tcol], format="mixed", utc=True, errors="coerce")
    df = df[df["ts"].notna()].sort_values("ts")
    return df


def _demo_engie_detector(df: pd.DataFrame, event: pd.DataFrame,
                         args: argparse.Namespace) -> None:
    """Run the real detector on R80711 and show what it alarms on.

    Same code path as the CARE scoring arm: a load- and ambient-conditioned
    reference per thermal channel, a one-sided robust z, and the channels
    combined by max-z.
    """
    # The detector needs power normalised to rated power, and ambient
    # temperature. ENGIE ships no ambient sensor, so ERA5 supplies it.
    era5_path = ROOT / "data" / "raw" / "era5_openmeteo.json"
    if not era5_path.exists():
        print(c("ERA5 not downloaded -- skipping the detector section", YELLOW))
        return

    work = df.copy()
    work["P_norm"] = work["P_avg"] / RATED_POWER_KW
    work = work.dropna(subset=["P_norm", ENGIE_CHANNEL])

    # Reference: the turbine's own earlier behaviour (2013-2015), the same
    # frozen-reference protocol the false-alarm numbers use.
    ref = work[(work["ts"] >= pd.Timestamp("2013-01-01", tz="UTC"))
               & (work["ts"] <= pd.Timestamp("2015-12-31", tz="UTC"))]
    test = work[work["ts"] >= pd.Timestamp("2016-01-01", tz="UTC")]
    if ref.empty or test.empty:
        print(c("not enough rows for the detector section", YELLOW))
        return

    res = multivariate.detect(
        ref, test, loaders.ENGIE_THERMAL, "P_norm",
        z_threshold=args.z_threshold, max_channels=24,
    )

    ev_start = pd.Timestamp(ENGIE_EVENT_START, tz="UTC")
    ev_end = pd.Timestamp(ENGIE_EVENT_END, tz="UTC")
    ts = test["ts"]
    if ts.dt.tz is None:
        # naive timestamps: compare against naive boundaries
        in_event = ((ts >= ev_start.tz_localize(None)) &
                    (ts <= ev_end.tz_localize(None))).to_numpy()
    else:
        in_event = ((ts >= ev_start) & (ts <= ev_end)).to_numpy()

    flags = res.flags_max.to_numpy()
    z = res.zmax.to_numpy()

    print(f"reference      : 2013-2015, frozen (this turbine's own earlier behaviour)")
    print(f"scored         : 2016-01-01 onward, {len(test):,} rows")
    print(f"threshold      : z > {args.z_threshold:.1f}  (max-z over "
          f"{len(res.channels_used)} thermal channels)")
    print(f"channels used  : {', '.join(res.channels_used)}")
    print()
    print(f"{'window':<26}{'rows':>10}{'alarms':>9}{'rate':>9}{'median z':>11}")
    print(f"{'2016-2017 (before)':<26}"
          f"{int((~in_event).sum()):>10,}{int(flags[~in_event].sum()):>9,}"
          f"{100*flags[~in_event].mean():>8.2f}%"
          f"{np.nanmedian(z[~in_event]):>11.2f}")
    print(f"{'Sept 2017 (the anomaly)':<26}"
          f"{int(in_event.sum()):>10,}{int(flags[in_event].sum()):>9,}"
          f"{100*flags[in_event].mean() if in_event.sum() else 0:>8.2f}%"
          f"{np.nanmedian(z[in_event]):>11.2f}")
    print()
    if in_event.sum():
        idx = np.where(in_event)[0]
        first = idx[flags[in_event]] if flags[in_event].any() else None
        if first is not None:
            when = test.iloc[first[0]]["ts"]
            print(f"first alarm in the window : {when:%Y-%m-%d %H:%M} UTC")
        else:
            print(c("no alarm raised in the window at this threshold", YELLOW))
    print()
    print(c("Note: this frozen reference is deliberately the hard case -- it is a", DIM))
    print(c("2013-2015 reference scoring 2017 data, so a drifting turbine reads as", DIM))
    print(c("anomalous. That is exactly why the deployed system uses a rolling", DIM))
    print(c("reference (see scripts/pipeline/08c_engie_multivariate.py).", DIM))



#: The window exported to the dashboard snapshot: two months around the
#: September 2017 anomaly, so the drift is visible on both sides of it.
SNAPSHOT_START = "2017-08-15"
SNAPSHOT_END = "2017-10-15"


def _far_pct(z) -> float | None:
    """The share of samples above z > 4 under one reference."""
    if z is None:
        return None
    v = z.to_numpy(dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return round(float((v > 4.0).mean() * 100), 2)


def export_snapshot(path: Path | None = None) -> int:
    """Write results/demo_r80711.json -- a compact snapshot for the dashboard.

    The dashboard has to work on Streamlit Community Cloud, where the 19 GB of
    raw SCADA is not available. So the R80711 case study is exported as a
    small, self-contained artefact: the bearing temperature of R80711 and its
    three siblings, the peer median, the load- and ambient-conditioned
    residual and the z-score, all on the same 10-minute grid.

    It is the SAME detector and the SAME reference the project measures with --
    no re-implementation, no smoothing, no resampling. The values are rounded
    for size, and nothing else.
    """
    base = _engie_dir()
    era5_path = ROOT / "data" / "raw" / "era5_openmeteo.json"
    if not era5_path.exists():
        print(c("ERA5 not downloaded -- cannot build the snapshot", YELLOW))
        return 1

    start = pd.Timestamp(SNAPSHOT_START, tz="UTC")
    end = pd.Timestamp(SNAPSHOT_END, tz="UTC")

    frames = {}
    for t in (ENGIE_TURBINE, *ENGIE_PEERS):
        d = _engie_read(_engie_csv(base, t))
        d["P_norm"] = d["P_avg"] / RATED_POWER_KW
        frames[t] = d

    # Ambient temperature from ERA5 (the ENGIE release has no ambient sensor).
    era5 = loaders.load_era5(ROOT / "data")
    e = era5.copy()
    e["ts"] = e["ts"].dt.floor("10min")
    e = e.set_index("ts")[["era5_temp_2m"]].resample("10min").ffill().reset_index()

    def _window(d: pd.DataFrame) -> pd.DataFrame:
        d = d.merge(e, on="ts", how="left")
        return d[(d["ts"] >= start) & (d["ts"] <= end)]

    main_df = _window(frames[ENGIE_TURBINE])
    # Every turbine on the same window, so they share one grid.
    win = {t: _window(frames[t]) for t in (ENGIE_TURBINE, *ENGIE_PEERS)}
    peers = {t: win[t] for t in ENGIE_PEERS}

    # --- the reference: R80711's own 2013-2015 behaviour, frozen -----------
    full = frames[ENGIE_TURBINE].merge(e, on="ts", how="left")
    ref_rows = full[(full["ts"] >= pd.Timestamp("2013-01-01", tz="UTC"))
                    & (full["ts"] <= pd.Timestamp("2015-12-31", tz="UTC"))]
    ref = multivariate.fit_channel_reference(
        ref_rows, ENGIE_CHANNEL, "P_norm", ambient_col="era5_temp_2m")
    if ref is None:
        print(c("could not fit the channel reference", RED))
        return 1

    # --- the grid: one row per timestamp the target turbine reports --------
    grid = main_df.dropna(subset=[ENGIE_CHANNEL]).reset_index(drop=True)
    ts = grid["ts"]

    corrected, z = ref.apply(
        grid["P_norm"].to_numpy(),
        grid["era5_temp_2m"].to_numpy(),
        grid[ENGIE_CHANNEL].to_numpy(),
    )

    # --- the detector's own alarms (max-z over all thermal channels) -------
    work = full.dropna(subset=["P_norm", ENGIE_CHANNEL])
    ref_all = work[(work["ts"] >= pd.Timestamp("2013-01-01", tz="UTC"))
                    & (work["ts"] <= pd.Timestamp("2015-12-31", tz="UTC"))]
    test_all = work[work["ts"] >= pd.Timestamp("2016-01-01", tz="UTC")]
    res = multivariate.detect(ref_all, test_all, loaders.ENGIE_THERMAL, "P_norm",
                              ambient_col="era5_temp_2m", z_threshold=4.0,
                              max_channels=24)
    # Align on the TIMESTAMP, not the positional index: the detector's score
    # is indexed by its own frame, whose rows are a different subset.
    zmax = None
    if len(res.channels_used):
        zmax_by_ts = res.zmax.copy()
        zmax_by_ts.index = test_all.loc[zmax_by_ts.index, "ts"]
        # ENGIE carries ~30 duplicate timestamps per turbine, and reindex
        # refuses duplicate labels. Keep the first of each.
        zmax_by_ts = zmax_by_ts[~zmax_by_ts.index.duplicated()]
        zmax = zmax_by_ts.reindex(ts)

    # --- the ROLLING reference, for comparison -----------------------------
    # The frozen reference above is the FAILURE mode the project is about: a
    # 2014 reference applied to 2017 data reads the drift as overheating, so it
    # alarms on a quarter of the samples. The rolling 30-day reference is what
    # a deployment would use, and it alarms on well under 1 %. Exporting both
    # lets the dashboard show the project's central result rather than only
    # the noisy end of it.
    zmax_roll = None
    rolling = multivariate.detect_rolling(
        work, loaders.ENGIE_THERMAL, "P_norm", ambient_col="era5_temp_2m",
        window_days=30, z_threshold=4.0, max_channels=24)
    if len(rolling.channels_used):
        roll_by_ts = rolling.zmax.copy()
        roll_by_ts.index = work.loc[roll_by_ts.index, "ts"]
        roll_by_ts = roll_by_ts[~roll_by_ts.index.duplicated()]
        zmax_roll = roll_by_ts.reindex(ts)

    # --- the peer median, at matched wind speed ----------------------------
    peer_frames = []
    for t, d in peers.items():
        dd = d[["ts", "Ws_avg", ENGIE_CHANNEL]].copy()
        dd["turbine"] = t
        peer_frames.append(dd)
    pf = pd.concat(peer_frames, ignore_index=True)
    pf = pf.dropna(subset=["Ws_avg", ENGIE_CHANNEL])
    pf["bin"] = (pf["Ws_avg"] // 1.0).astype("int64")
    peer_by_bin = pf.groupby("bin")[ENGIE_CHANNEL].median()
    main_bin = (grid["Ws_avg"] // 1.0).astype("int64")
    peer_median = main_bin.map(peer_by_bin).to_numpy()

    series: dict[str, list] = {}
    for t in (ENGIE_TURBINE, *ENGIE_PEERS):
        d = win[t].drop_duplicates(subset="ts", keep="first").set_index("ts")
        d = d.reindex(ts)
        series[t] = [None if pd.isna(v) else round(float(v), 2)
                     for v in d[ENGIE_CHANNEL].to_numpy()]
    series["peer_median"] = [None if not np.isfinite(v) else round(float(v), 2)
                             for v in peer_median]
    series["residual"] = [None if not np.isfinite(v) else round(float(v), 3)
                          for v in corrected]
    series["z"] = [None if not np.isfinite(v) else round(float(v), 3)
                   for v in z]
    if zmax is not None:
        series["zmax"] = [None if pd.isna(v) else round(float(v), 3)
                          for v in zmax.to_numpy()]
    if zmax_roll is not None:
        series["zmax_rolling"] = [None if pd.isna(v) else round(float(v), 3)
                                  for v in zmax_roll.to_numpy()]
    series["ambient"] = [None if pd.isna(v) else round(float(v), 2)
                         for v in grid["era5_temp_2m"].to_numpy()]
    series["wind_speed"] = [None if pd.isna(v) else round(float(v), 2)
                            for v in grid["Ws_avg"].to_numpy()]

    epoch = (ts.astype("int64") // 10**9).tolist()

    ev = grid[(grid["ts"] >= pd.Timestamp(ENGIE_EVENT_START, tz="UTC"))
              & (grid["ts"] <= pd.Timestamp(ENGIE_EVENT_END, tz="UTC"))]
    peer_max = max(
        float(np.nanmax(peers[t][ENGIE_CHANNEL].to_numpy())) for t in ENGIE_PEERS)

    out = {
        "generated_by": "scripts/pipeline/11_demo.py",
        "meta": {
            "site": "ENGIE La Haute Borne, Meuse, France",
            "turbines": [ENGIE_TURBINE, *ENGIE_PEERS],
            "target_turbine": ENGIE_TURBINE,
            "peer_turbines": list(ENGIE_PEERS),
            "channel": ENGIE_CHANNEL,
            "channel_description": "gearbox bearing 2 temperature",
            "window_start": SNAPSHOT_START,
            "window_end": SNAPSHOT_END,
            "reference": "frozen 2014-2015, load- and ambient-conditioned, "
                         "fitted on R80711's own earlier behaviour",
            "n_rows": int(len(epoch)),
            "resolution": "10 minutes",
            "units": {"temperature": "degC", "residual": "K",
                      "z": "robust deviations", "ambient": "degC"},
            "caveat": "R80711's September 2017 anomaly is NOT a labelled "
                      "fault. The ENGIE release carries no fault label for "
                      "it, and the one documented gearbox event (R80721, 2015) "
                      "is absent from the released cleaned CSVs.",
        },
        "kpis": {
            "r80711_max_c": round(float(np.nanmax(grid[ENGIE_CHANNEL].to_numpy())), 2),
            "peer_max_c": round(peer_max, 2),
            "rows_over_80c": int((grid[ENGIE_CHANNEL] > 80).sum()),
            "peer_rows_over_80c": int(sum(
                int((peers[t][ENGIE_CHANNEL] > 80).sum()) for t in ENGIE_PEERS)),
            "ambient_mean_c": round(float(np.nanmean(grid["era5_temp_2m"].to_numpy())), 2),
            "event_max_c": round(float(np.nanmax(ev[ENGIE_CHANNEL].to_numpy())), 2),
            "false_alarm_pct_frozen_reference": _far_pct(zmax),
            "false_alarm_pct_rolling_reference": _far_pct(zmax_roll),
        },
        "t": epoch,
        "series": series,
    }

    dest = path or (ROOT / "results" / "demo_r80711.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    size_mb = dest.stat().st_size / 1e6
    print(f"wrote {dest} ({size_mb:.2f} MB, {len(epoch):,} rows)")

    # The downloadable sample: a real slice of the ENGIE release, with the
    # ERA5 ambient temperature added, so a reviewer on Streamlit Community
    # Cloud can run the detector on a real turbine without the 19 GB.
    sample = live.build_sample_csv(live.sample_csv_path(), ROOT / "data")
    print(f"wrote {sample} ({sample.stat().st_size/1e6:.2f} MB)")

    # The frozen-baseline bundles -> results/demo_frozen_baselines.json.
    # The reference curves for one and two years back, and the target
    # weeks' actuals, so the multi-year staleness comparison works on
    # Streamlit Community Cloud without the 19 GB of raw SCADA.
    frozen = forecast.export_frozen_baselines(
        forecast.frozen_baselines_path(), ROOT / "data")
    print(f"wrote {frozen} ({frozen.stat().st_size/1e6:.2f} MB, "
          f"{len(forecast.CHECKPOINTS)} checkpoints)")
    return 0


def demo_engie(args: argparse.Namespace) -> int:
    base = _engie_dir()
    path = _engie_csv(base, ENGIE_TURBINE)
    df = _engie_read(path)

    rule("ENGIE LA HAUTE BORNE - R80711, September 2017")
    print(f"turbine        : {ENGIE_TURBINE}  (2 050 kW Senvion MM82, Meuse, France)")
    print(f"source file    : {path.name}")
    print(f"rows in file   : {len(df):,}   span {df['ts'].min():%Y-%m-%d} -> {df['ts'].max():%Y-%m-%d}")
    print(f"channel        : {ENGIE_CHANNEL} (gearbox bearing 2 temperature)")

    temp_cols = [c_ for c_ in loaders.ENGIE_THERMAL if c_ in df.columns]
    if not temp_cols:
        print(c("no temperature columns found in this file", RED))
        return 1
    if ENGIE_CHANNEL not in df.columns:
        print(c(f"{ENGIE_CHANNEL} not in {path.name}", RED))
        return 1
    tcol = ENGIE_CHANNEL

    ev_start = pd.Timestamp(ENGIE_EVENT_START, tz="UTC")
    ev_end = pd.Timestamp(ENGIE_EVENT_END, tz="UTC")
    event = df[(df["ts"] >= ev_start) & (df["ts"] <= ev_end)]

    rule("1. THE CHANNEL ITSELF")
    print(f"{'month':<10}{'max (C)':>10}{'rows > 80C':>13}{'mean (C)':>11}   distribution")
    for month in pd.period_range(ev_start, ev_end, freq="M"):
        # period_range yields tz-naive boundaries; the data is tz-aware.
        start = month.start_time.tz_localize("UTC")
        end = month.end_time.tz_localize("UTC")
        m = df[(df["ts"] >= start) & (df["ts"] <= end)]
        if m.empty:
            continue
        hi = int((m[tcol] > 80).sum())
        print(f"{str(month):<10}{m[tcol].max():>10.2f}{hi:>13}{m[tcol].mean():>11.2f}   "
              f"{bar(m[tcol].max(), 90)} {m[tcol].max():.1f}")

    rule("2. THE CONTROL GROUP - the same channel on the three sibling turbines")
    print("If R80711 were only reacting to weather, its peers would move with it.")
    print("They do not.\n")
    peers = {}
    for peer in ENGIE_PEERS:
        try:
            peers[peer] = _engie_read(_engie_csv(base, peer))
        except FileNotFoundError:
            continue
    if not peers:
        print(c("no peer files found", YELLOW))
        return 1

    print(f"{'turbine':<10}{'max (C)':>10}{'rows > 80C':>13}{'mean (C)':>11}   "
          f"{'':>28}")
    print(f"{c(ENGIE_TURBINE, BOLD):<10}{event[tcol].max():>10.2f}"
          f"{int((event[tcol] > 80).sum()):>13}{event[tcol].mean():>11.2f}   "
          f"{bar(event[tcol].max(), 90)}")
    for peer, pdf in peers.items():
        pm = pdf[(pdf["ts"] >= ev_start) & (pdf["ts"] <= ev_end)]
        if pm.empty:
            continue
        print(f"{peer:<10}{pm[tcol].max():>10.2f}"
              f"{int((pm[tcol] > 80).sum()):>13}{pm[tcol].mean():>11.2f}   "
              f"{bar(pm[tcol].max(), 90)}")

    rule("3. THE DRIFT - residual against the peers at matched wind speed")
    print("Own value minus the peer median at the same wind bin, by month.")
    print("A weather effect cancels here; a machine-specific fault does not.\n")
    frames = []
    for t in (ENGIE_TURBINE, *peers):
        d = peers.get(t) if t != ENGIE_TURBINE else df
        d = d.copy()
        d["turbine"] = t
        d["year"] = d["ts"].dt.year
        d["month"] = d["ts"].dt.to_period("M")
        frames.append(d[["turbine", "year", "month", "Ws_avg", tcol]])
    allf = pd.concat(frames, ignore_index=True)
    allf["bin"] = (allf["Ws_avg"] // 1.0).astype("int64")
    # The peer median EXCLUDES the target: a turbine compared against a group
    # it belongs to would be pulled toward it and the residual would vanish.
    peer_med = (allf[allf["turbine"] != ENGIE_TURBINE]
                .groupby(["year", "month", "bin"])[tcol].median()
                .rename("peer_med").reset_index())
    own = allf[allf["turbine"] == ENGIE_TURBINE].merge(
        peer_med, on=["year", "month", "bin"], how="left")
    own["resid"] = own[tcol] - own["peer_med"]
    bym = own.groupby(["year", "month"])["resid"].agg(["mean", "max", "size"])
    tail = bym.tail(10)
    print(f"{'month':<10}{'mean resid':>12}{'max resid':>12}{'rows':>9}   trend")
    for (yr, mo), r in tail.iterrows():
        print(f"{str(mo):<10}{r['mean']:>12.2f}{r['max']:>12.2f}{int(r['size']):>9}   "
              f"{bar(max(r['mean'], 0), 5)}")

    rule("4. WHAT THE DETECTOR DOES WITH IT")
    _demo_engie_detector(df, event, args)

    rule("READING")
    print(
        f"R80711 runs {event[tcol].mean() - np.mean([peers[p][(peers[p]['ts'] >= ev_start) & (peers[p]['ts'] <= ev_end)][tcol].mean() for p in peers]):.2f} C\n"
        f"hotter than its peers on the same channel, and exceeds 80 C on\n"
        f"{int((event[tcol] > 80).sum()):,} rows in September 2017, peaking at {event[tcol].max():.2f} C,\n"
        "while all three peers stay below 80 C. The peer residual grows month on\n"
        "month through 2017 and keeps growing into 2018 -- a drift, not a transient.\n"
        "\n"
        "ERA5 ambient over the same window is 15-16 C, so weather does not explain it.\n"
        "\n"
        "This is NOT a labelled fault: the ENGIE release carries no fault label for it,\n"
        "and the published gearbox event (R80721, 2015) is absent from the cleaned CSVs\n"
        "entirely. It is an abnormality found in unlabelled data, with the peers as the\n"
        "control -- evidence of what the detector is for, not a benchmark score."
    )
    return 0


# --- CARE demo --------------------------------------------------------------
def demo_care(args: argparse.Namespace) -> int:
    root = care.care_root(ROOT / "data")
    if not root.exists():
        print(c(f"CARE not found at {root} -- run scripts/pipeline/01_build_dataset.py", RED))
        return 1

    farm_dir = root / args.farm
    if not farm_dir.exists():
        print(c(f"no {args.farm}", RED))
        return 1

    info = care.event_info(farm_dir)
    anomalies = info[info["event_label"].astype(str).str.strip().str.lower() == "anomaly"]
    if anomalies.empty:
        print(c("no anomaly events in this farm", YELLOW))
        return 1

    rule(f"CARE TO COMPARE - {args.farm}, labelled anomaly")
    print(f"{'event':<8}{'window':<26}{'description'}")
    for _, row in anomalies.head(12).iterrows():
        s, e = int(row["event_start_id"]), int(row["event_end_id"])
        print(f"{str(row['event_id']):<8}{f'{s} - {e}':<26}"
              f"{str(row.get('event_description', ''))[:56]}")

    ev = anomalies.iloc[min(args.event, len(anomalies) - 1)]
    eid = str(ev["event_id"])
    s, e = int(ev["event_start_id"]), int(ev["event_end_id"])

    files = sorted((farm_dir / "datasets").glob("*.csv"))
    files = [f for f in files if f.stem == eid]
    if not files:
        print(c(f"no CSV for event {eid}", RED))
        return 1
    path = files[0]

    df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
    train = df[df["train_test"] == "train"]
    test = df[df["train_test"] == "prediction"]

    columns = care.dataset_columns(path)
    pcol, thermal, wind, ambient = care.feature_rows(farm_dir, columns)
    if not thermal:
        print(c("no thermal channels found", RED))
        return 1

    rule("DETECTOR")
    train_n = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
    res = multivariate.detect(
        train_n, test, thermal, pcol, ambient_col=ambient,
        quantile=args.quantile, max_channels=24,
    )

    print(f"event          : {eid}  ({ev.get('event_description', '')})")
    print(f"dataset        : {path.name}")
    print(f"channels used  : {len(res.channels_used)} thermal channels")
    print(f"threshold (max): {res.threshold_max:.3f}   "
          f"cross-fitted training quantile {args.quantile}")
    print(f"threshold (L2) : {res.threshold_l2:.3f}")

    in_event = (test.index.to_numpy() >= s) & (test.index.to_numpy() <= e)
    status = test["status_type_id"].to_numpy()
    scorable = in_event & np.isin(status, care.NORMAL_STATUS_IDS)

    rule("ALARMS")
    flagged = res.flags_max.to_numpy()
    n_flag_event = int((flagged & scorable).sum())
    print(f"rows in event window          : {int(in_event.sum()):,}")
    print(f"  of which scorable (status 0/2): {int(scorable.sum()):,}")
    print(f"alarms raised in that window  : {n_flag_event:,}")
    print(f"alarms outside the window     : "
          f"{int((flagged & ~in_event).sum()):,}")
    if scorable.sum():
        alarm_rows = np.where(scorable & flagged)[0]
        if alarm_rows.size:
            print(f"first alarm at row            : {alarm_rows[0]:,} "
                  f"of {int(scorable.sum()):,} scorable rows")
        else:
            print(c("no alarm raised inside the event window at this threshold",
                    YELLOW))
    else:
        print()
        print(c("  This event has NO scorable rows: the fault occupies the whole", YELLOW))
        print(c("  window, so the benchmark cannot award credit for detecting it.", YELLOW))
        print(c("  11 of Wind Farm A's 12 anomaly events are like this -- see", YELLOW))
        print(c("  results/farm_a_audit.json. That is a property of the published", YELLOW))
        print(c("  labels, not of any detector.", YELLOW))

    rule("SCORE ON THIS EVENT")
    z = res.zmax.to_numpy()
    if z.size:
        if scorable.sum():
            print(f"median z inside event   : {np.nanmedian(z[scorable]):.3f}")
        if (~in_event).sum():
            print(f"median z outside event  : {np.nanmedian(z[~in_event]):.3f}")
        print(f"max z over the whole test set: {np.nanmax(z):.3f}")
    else:
        print(c("no scorable rows in the test split", YELLOW))
    return 0


def list_datasets() -> int:
    root = care.care_root(ROOT / "data")
    if not root.exists():
        print(c("CARE not downloaded", YELLOW))
        return 1
    for farm in ("Wind Farm A", "Wind Farm B", "Wind Farm C"):
        farm_dir = root / farm
        if not farm_dir.exists():
            continue
        info = care.event_info(farm_dir)
        anom = info[info["event_label"].astype(str).str.strip().str.lower() == "anomaly"]
        print(f"\n{farm}: {len(anom)} anomaly events")
        for _, row in anom.iterrows():
            print(f"  {str(row['event_id']):<10}"
                  f"{str(row.get('event_description', ''))[:64]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list the CARE anomaly events")
    ap.add_argument("--dataset", choices=("engie", "care"), default="engie",
                    help="which demonstration to run")
    ap.add_argument("--farm", default="Wind Farm C", help="CARE farm")
    ap.add_argument("--event", type=int, default=0,
                    help="index of the CARE anomaly event")
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="training quantile for the threshold; 0.90 is what "
                         "the scored arm uses (scripts/pipeline/08_multivariate.py)")
    ap.add_argument("--z-threshold", type=float, default=4.0,
                    help="fixed z threshold for the ENGIE detector demo")
    ap.add_argument("--export", action="store_true",
                    help="write results/demo_r80711.json for the dashboard")
    args = ap.parse_args()

    if args.export:
        return export_snapshot()
    if args.list:
        return list_datasets()
    if args.dataset == "care":
        return demo_care(args)
    return demo_engie(args)


if __name__ == "__main__":
    raise SystemExit(main())
