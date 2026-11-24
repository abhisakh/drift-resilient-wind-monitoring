"""The project's goal, stated as measurements and read from results/*.json.

THE GOAL
--------
Cut the cost of unnecessary maintenance call-outs by raising fewer false
alarms, while still warning early enough of a developing bearing fault that
the turbine can be stopped on purpose rather than tripping on its own.

That is three questions, and each is answered from an artefact rather than
asserted:

  1. How many fewer false alarms does a fresh reference give?
     (the backtest's B.4: the same rule, the same weeks, only the age of
     the reference changed, every arm calibrated to the same 1 % rate)
  2. Is a developing fault caught, and how fast?
     (C.6: a ramp injected into a healthy test week)
  3. How many days of warning before the 80 C crossing?
     (C.7: the unlabelled R80711 anomaly of September 2017)

The dashboard's headline cards and the README's goal section both read
this module, so the two cannot disagree with each other or with
`results/`. Nothing here computes a new number: it selects and arranges
figures that the pipeline already wrote. A missing artefact degrades to
None, never to an exception, so the dashboard still renders without it.
"""

from __future__ import annotations

#: The arm the project deploys, and the learned comparison, as the backtest
#: names them in its results.
DEPLOYED_ARM = "load+ambient"
LEARNED_ARM = "lightgbm"

#: The ramp the headline detection figure is quoted at, in kelvin.
HEADLINE_RAMP_K = 4.0

#: The reference ages quoted in the headline, as the backtest keys them.
AGE_KEYS = {"three_months": "90d", "one_year": "365d", "two_years": "730d"}


def _get(d, *path):
    """Walk a nested dict; None if any step is missing."""
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def as_fraction(x):
    """A rate as a fraction in [0, 1], whichever way the artefact stored it.

    The backtest stores detection rates as fractions, but a value above 1
    can only be a percentage, so it is divided down rather than shown as
    "9000 %".
    """
    v = _as_float(x)
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


#: The standing operating point the dashboard's dispatch rule implies: a 1 %
#: sensitivity, sustained for an hour of ten-minute samples.
STANDING_RATE = 0.01
STANDING_PERSISTENCE = 6


def curve_reading(curve, arm: str = DEPLOYED_ARM, rate: float = STANDING_RATE,
                  persistence: int = STANDING_PERSISTENCE,
                  ramp_k: float = HEADLINE_RAMP_K):
    """One setting of the operating curve, for one arm, or None.

    The curve (results/operating_curve.json) measures false dispatches per
    turbine-year and the time to catch an injected ramp at every setting of
    an arm's dial, with a 0 K placebo as the chance level. This picks one
    setting out of it; nothing is recomputed.
    """
    for p in (curve or {}).get("points") or []:
        if (p.get("arm") == arm and p.get("target_rate") == rate
                and p.get("persistence_samples") == persistence):
            ramps = p.get("ramps") or {}
            ramp = ramps.get(f"{ramp_k:g}") or {}
            placebo = ramps.get("0") or {}
            return {
                "false_dispatches_per_turbine_year":
                    _as_float(p.get("false_dispatches_per_turbine_year")),
                "detection_rate": _as_float(ramp.get("detection_rate")),
                "median_days": _as_float(ramp.get("median_days_to_dispatch")),
                "chance_rate": _as_float(placebo.get("detection_rate")),
            }
    return None


def curve_budget_reading(curve, budget: int, arm: str):
    """An arm's best setting within a false-call-out budget, or None."""
    row = ((curve or {}).get("budgets_ramp_4k") or {}).get(str(budget)) or {}
    r = row.get(arm)
    return dict(r) if isinstance(r, dict) else None


def anomaly_reading(curve, arm: str = DEPLOYED_ARM,
                    rate: float = STANDING_RATE,
                    persistence: int = STANDING_PERSISTENCE):
    """One setting of the dial on the real R80711 anomaly, or None.

    `days_before_hot_spell` is how many days BEFORE the first sustained hot
    spell the first dispatch fell; negative means after it. `None` for the
    first dispatch means the setting never dispatched during the anomaly.
    """
    block = (curve or {}).get("r80711_lead_time") or {}
    for r in block.get("rows") or []:
        if (r.get("arm") == arm and r.get("target_rate") == rate
                and r.get("persistence_samples") == persistence):
            return {
                "first_dispatch": r.get("first_dispatch_after_onset"),
                "days_after_onset": _as_float(r.get("days_after_onset")),
                "days_before_hot_spell": _as_float(r.get("days_before_hot_spell")),
                "dispatches_before_onset": r.get("dispatches_before_onset"),
            }
    return None


def early_clean_settings(curve, arm: str = DEPLOYED_ARM):
    """(settings that warned early AND cleanly, settings swept) for one arm.

    A setting counts when its first dispatch fell before the first sustained
    hot spell AND it raised no dispatch in the quiet stretch before the onset.
    A jumpy setting that dispatches early only because it dispatches at
    everything does not count.
    """
    block = (curve or {}).get("r80711_lead_time") or {}
    rows = [r for r in (block.get("rows") or []) if r.get("arm") == arm]
    if not rows:
        return None
    ok = [r for r in rows
          if r.get("dispatches_before_onset") == 0
          and (_as_float(r.get("days_before_hot_spell")) or 0.0) > 0.0]
    return len(ok), len(rows)


def goal_scorecard(backtest, engie=None, snapshot=None, curve=None) -> dict:
    """The three measurements of the goal, plus the R80711 case study.

    `backtest` is results/backtest.json, `engie` is
    results/engie_multivariate.json, `snapshot` is results/demo_r80711.json
    and `curve` is results/operating_curve.json; any of them may be None.
    """
    b = backtest or {}

    # 1. false alarms: the age of the reference, at a matched threshold.
    sweep = _get(b, "B4_reference_age", "sweep_over_reference_age") or {}
    ratios = {name: _as_float(_get(sweep, key, "ratio_over_30d"))
              for name, key in AGE_KEYS.items()}

    # ...and the extreme case: a reference frozen in 2014, each arm at its
    # own threshold rule. The same effect at its largest, not a like-for-like
    # figure, so it is kept apart.
    strategies = (engie or {}).get("strategies", {})
    frozen = _as_float(_get(strategies, "frozen_2014", "mean_rate_pct"))
    rolling = _as_float(_get(strategies, "rolling_30d", "mean_rate_pct"))
    frozen_ratio = frozen / rolling if frozen and rolling else None

    # 2. detection: an injected ramp, per arm.
    detection: dict = {}
    for x in (_get(b, "C6_fault_injection", "summary") or {}).values():
        if not isinstance(x, dict):
            continue
        magnitude = _as_float(x.get("magnitude_k"))
        if magnitude is None or abs(magnitude - HEADLINE_RAMP_K) > 1e-9:
            continue
        detection[x.get("arm")] = {
            "rate": as_fraction(x.get("detection_rate")),
            "median_days": _as_float(x.get("median_days_to_first_alarm")),
        }

    # 3. lead time on the R80711 anomaly, per arm, at two false-alarm rates.
    lead = b.get("C7_lead_time") or {}
    lead_time = {
        "first_80c_crossing": lead.get("first_80c_crossing"),
        "first_sibling_divergence_5k": lead.get("first_sibling_divergence_5k"),
        "days_before_80c_1pct": lead.get("days_before_80c_1pct") or {},
        "days_before_80c_5pct": lead.get("days_before_80c_5pct") or {},
    }

    kpis = (snapshot or {}).get("kpis") or {}
    curve_arms = ((curve or {}).get("protocol") or {}).get("arms") or []
    return {
        "false_alarm": {
            "ratio_by_reference_age": ratios,
            "ratio_frozen_2014": frozen_ratio,
        },
        "detection": detection,
        "operating_point": {arm: curve_reading(curve, arm) for arm in curve_arms},
        "anomaly": {
            "first_hot_spell": ((curve or {}).get("r80711_lead_time") or {}).get(
                "first_hot_spell"),
            "onset": ((curve or {}).get("r80711_lead_time") or {}).get("onset"),
            "standing": {arm: anomaly_reading(curve, arm) for arm in curve_arms},
            "early_clean_settings": {
                arm: early_clean_settings(curve, arm) for arm in curve_arms},
        },
        "lead_time": lead_time,
        "r80711": {
            "peak_c": _as_float(kpis.get("r80711_max_c")),
            "peer_max_c": _as_float(kpis.get("peer_max_c")),
        },
    }


def lead_days(card: dict, arm: str = DEPLOYED_ARM):
    """(days at a 1 % false-alarm rate, days at 5 %) for one arm, or None."""
    lt = card.get("lead_time", {})
    one = _as_float((lt.get("days_before_80c_1pct") or {}).get(arm))
    five = _as_float((lt.get("days_before_80c_5pct") or {}).get(arm))
    return one, five
