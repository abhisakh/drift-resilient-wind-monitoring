"""Alarm triage: is an alarm real excess heat, or load and weather?

For every alarm the same accounting. On the alarm day, the component's hottest
readings are split into four parts, using the healthy reference of the
component (trained on healthy data with the weather, as in the daily check):

    reading = typical + load + weather + excess

* typical -- what the component reads at its usual load in usual weather;
* load    -- what running harder than usual adds (the reference's load curve);
* weather -- what the day's ambient temperature adds (the reference's ambient
             correction);
* excess  -- what neither explains: the component is hotter than its healthy
             self under the same load and weather.

The verdict follows the excess, checked against two references when there are:
the component's own healthy model, and its sibling turbines at the same moment
and similar power (they share the weather and the load, so their gap is excess
by construction).

* "real excess heat"  -- the day's 95th percentile of the excess, by either
  reference, is above `REAL_HEAT_K`;
* "explained by load" / "explained by weather" -- it is not, and the larger of
  the load and weather parts names the cause;
* "no verdict" -- there was no healthy reference that day.

The own reference is the daily check's frozen monthly model; while that is
blank after a recorded repair, a provisional one fitted on at least
`SHORT_HOLD_READINGS` healthy readings since the repair. Days the daily check
flagged are never trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import daily_check as dc
from . import multivariate

REAL_HEAT_K = 5.0
PEAK_SHARE = 0.05                  # the hottest 5 % of the day's readings
SHORT_HOLD_READINGS = 14 * 144     # two weeks of 10-minute readings
SIBLING_PRIOR_DAYS = 30
SIBLING_MIN_PRIOR = 7 * dc.MIN_DAY_ROWS
VERDICTS = ("real excess heat", "explained by load", "explained by weather", "no verdict")


def decompose(ref, load, ambient, actual, typical_load: float) -> dict:
    """Split readings into typical + load + weather + excess (arrays)."""
    load = np.asarray(load, dtype="float64")
    ambient = np.asarray(ambient, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    with np.errstate(invalid="ignore"):
        bins = np.floor(load / multivariate.LOAD_BIN)
        abins = np.floor(ambient / multivariate.AMBIENT_BIN)
    base = ref.medians.reindex(pd.Series(bins).astype("Int64")).to_numpy(
        dtype="float64", na_value=np.nan)
    off = ref.ambient_offsets.reindex(pd.Series(abins).astype("Int64")).to_numpy(
        dtype="float64", na_value=np.nan)
    off = np.where(np.isfinite(off), off, 0.0)
    idx = ref.medians.index.to_numpy()
    tb = int(np.floor(typical_load / multivariate.LOAD_BIN))
    typical = float(ref.medians.loc[idx[np.argmin(np.abs(idx - tb))]])
    return {"typical": np.full(len(load), typical), "load": base - typical,
            "weather": off, "excess": actual - base - off}


def verdict(own_p95, sib_p95, load_part, weather_part, real_heat_k=REAL_HEAT_K) -> str:
    vals = [v for v in (own_p95, sib_p95) if v is not None and np.isfinite(v)]
    if not vals:
        return "no verdict"
    if max(vals) > real_heat_k:
        return "real excess heat"
    lp = load_part if load_part is not None and np.isfinite(load_part) else -np.inf
    wp = weather_part if weather_part is not None and np.isfinite(weather_part) else -np.inf
    if lp == -np.inf and wp == -np.inf:
        return "explained by load"      # siblings only: shared load and weather
    return "explained by load" if lp >= wp else "explained by weather"


class Component:
    """One component of one turbine: its daily check, its references and its
    sibling gap, ready to judge any day."""

    def __init__(self, frame: pd.DataFrame, turbine: str, channel: str,
                 repairs=(), sibling_gap: pd.Series | None = None,
                 tolerance: float = REAL_HEAT_K):
        self.channel = channel
        self.repairs = sorted(pd.Timestamp(r) for r in repairs)
        self.sim = dc.simulate(frame, turbine, tolerance, spans={}, channel=channel,
                               maintenance_extra=[pd.Timestamp(r, tz="UTC") for r in self.repairs])
        m, _ = dc.mark_restarts(frame)
        m["tsn"] = m["ts"].dt.tz_convert(None)
        m["dayn"] = m["day"].dt.tz_convert(None)
        self.m = m
        self.usable = (m["running"] & ~m["warmup"]).to_numpy()
        self.days = {d["day"]: d for d in self.sim["days"]}
        self.months = {mo["month"]: mo for mo in self.sim["months"]}
        self.flagged = {pd.Timestamp(k) for k, d in self.days.items() if d.get("flag") == "hot"}
        self.sibling_gap = sibling_gap
        self._refs: dict = {}

    def _fit(self, sel):
        key = (int(np.argmax(sel)) if sel.any() else -1, int(sel.sum()))
        if key not in self._refs:
            rows = self.m[sel]
            ref = multivariate.fit_channel_reference(rows, self.channel, "P_norm",
                                                     ambient_col="era5_temp_2m")
            run = rows[rows["P_norm"] > dc.RUNNING]
            self._refs[key] = (ref, float(run["P_norm"].median()) if len(run) else 0.5)
        return self._refs[key]

    def reference(self, day: pd.Timestamp):
        """(reference, typical load, kind) for a day, or None."""
        d = self.days.get(str(day)[:10])
        healthy = self.usable & ~self.m["dayn"].isin(self.flagged).to_numpy()
        if d and d.get("status") in ("checked", "thin") and d.get("state"):
            mo = self.months[str(day)[:7]]
            lo, hi = pd.Timestamp(mo["model_train_start"]), pd.Timestamp(mo["model_train_end"])
            sel = healthy & (self.m["tsn"] >= lo).to_numpy() & (self.m["tsn"] < hi).to_numpy()
            ref, tl = self._fit(sel)
            return (ref, tl, "monthly") if ref is not None else None
        before = [r for r in self.repairs if r < day]
        if before:
            sel = (healthy & (self.m["tsn"] >= before[-1]).to_numpy()
                   & (self.m["tsn"] < day).to_numpy())
            if sel.sum() >= SHORT_HOLD_READINGS:
                ref, tl = self._fit(sel)
                return (ref, tl, "provisional after repair") if ref is not None else None
        return None

    def judge_day(self, day: pd.Timestamp) -> dict:
        day = pd.Timestamp(day).normalize()
        out = {"day": str(day)[:10], "own_p95": None, "sib_p95": None,
               "reference": None, "typical": None, "load": None, "weather": None,
               "excess": None, "reading": None}
        today = self.usable & (self.m["dayn"] == day).to_numpy()
        if today.sum() >= dc.MIN_DAY_ROWS:
            r = self.reference(day)
            if r is not None:
                ref, tl, kind = r
                rows = self.m[today]
                parts = decompose(ref, rows["P_norm"], rows["era5_temp_2m"], rows[self.channel], tl)
                ok = np.isfinite(parts["excess"])
                if ok.sum() >= dc.MIN_DAY_ROWS:
                    actual = rows[self.channel].to_numpy(dtype="float64")[ok]
                    hot = actual >= np.quantile(actual, 1 - PEAK_SHARE)
                    out.update(reference=kind,
                               own_p95=float(np.quantile(parts["excess"][ok], 0.95)),
                               reading=float(actual[hot].mean()),
                               **{k: float(parts[k][ok][hot].mean())
                                  for k in ("typical", "load", "weather", "excess")})
        g = self.sibling_gap
        if g is not None:
            t0, t1 = day, day + pd.Timedelta(days=1)
            now = g[(g.index >= t0) & (g.index < t1)]
            prior = g[(g.index >= day - pd.Timedelta(days=SIBLING_PRIOR_DAYS)) & (g.index < day)]
            prior = prior[~prior.index.floor("D").isin(self.flagged)]
            if len(now) >= dc.MIN_DAY_ROWS and len(prior) >= SIBLING_MIN_PRIOR:
                out["sib_p95"] = float(np.quantile(now - prior.median(), 0.95))
        return out

    def judge_spell(self, start, end) -> dict:
        """The spell's verdict: its worst day decides, and that day's split is reported."""
        days = [self.judge_day(d) for d in pd.date_range(pd.Timestamp(start), pd.Timestamp(end))]
        judged = [d for d in days if d["own_p95"] is not None or d["sib_p95"] is not None]
        if not judged:
            return {"verdict": "no verdict", "days_judged": 0}

        def worst(d):
            return max(v for v in (d["own_p95"], d["sib_p95"]) if v is not None)
        w = max(judged, key=worst)
        return {"verdict": verdict(max((d["own_p95"] for d in judged if d["own_p95"] is not None), default=None),
                                   max((d["sib_p95"] for d in judged if d["sib_p95"] is not None), default=None),
                                   w["load"], w["weather"]),
                "days_judged": len(judged), "worst_day": w["day"],
                **{k: (round(w[k], 2) if isinstance(w[k], float) else w[k])
                   for k in ("reference", "own_p95", "sib_p95", "reading", "typical",
                             "load", "weather", "excess")}}


def sibling_gap(temps: pd.DataFrame, loads: pd.DataFrame, turbine: str,
                match: float = dc.SIBLING_LOAD_MATCH) -> pd.Series:
    """The component minus its siblings' median, at moments of similar power."""
    peers = [p for p in temps.columns if p != turbine]
    gap = temps[turbine] - temps[peers].median(axis=1)
    ok = (((loads[turbine] - loads[peers].median(axis=1)).abs() < match)
          & (loads[turbine] > dc.RUNNING) & gap.notna())
    return gap[ok]


def summarise(spells: pd.DataFrame, label: str = "followed_thermal") -> dict:
    """The triage table: verdict x (followed by a failure / not)."""
    out = {}
    for v in VERDICTS:
        s = spells[spells["verdict"] == v]
        out[v] = {"alarms": int(len(s)), "followed_by_failure": int(s[label].sum()),
                  "no_failure": int((~s[label]).sum())}
    out["all"] = {"alarms": int(len(spells)), "followed_by_failure": int(spells[label].sum()),
                  "no_failure": int((~spells[label]).sum())}
    return out
