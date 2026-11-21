"""Detection and deterministic gating.

Design rule: the ML/statistical layer produces a *score*; the deterministic layer
decides whether an alert is allowed to fire. Every alert must be reproducible from
an audit record (which gates fired, on what evidence, under which reference).

This mirrors how real condition monitoring is licensed to operators: an algorithm
suggests, a rule set disposes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --- alarm budget (ISA-18.2 / IEC 62682 style) -------------------------------
ALARM_BUDGET_PER_10MIN = 1.0
FLOOD_FRACTION_MAX = 0.01  # 1 % of the reporting period


@dataclass
class Gate:
    """One deterministic precondition for raising an alarm."""

    name: str
    passed: bool
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def operating_state_gate(
    df: pd.DataFrame,
    power_col: str = "P_avg",
    wind_col: str = "Ws_avg",
    rated_kw: float = 2050.0,
    min_power_frac: float = 0.05,
    min_wind: float = 2.0,
) -> Gate:
    """Reject samples that are not in a generating state.

    Standstill and curtailment produce low power that would otherwise look like
    underperformance. The ENGIE release has already filtered most standstill rows
    (P==0 count is 0 across all turbines), but partial-curtailment and low-wind rows
    remain and must be gated.
    """
    p = df[power_col]
    v = df[wind_col]
    ok = (p >= rated_kw * min_power_frac) & (v >= min_wind)
    frac_in = float(ok.mean())
    return Gate(
        name="operating_state",
        passed=frac_in >= 0.50,
        detail=f"{frac_in*100:.1f}% of samples in a generating state",
        evidence={
            "frac_generating": round(frac_in, 4),
            "n_power_zero": int((p == 0).sum()),
            "n_power_negative": int((p < 0).sum()),
            "n_below_power_gate": int((~ok).sum()),
        },
    )


def persistence_gate(
    df: pd.DataFrame,
    flag_col: str = "flag",
    consecutive: int = 3,
) -> Gate:
    """Require the condition to persist for N consecutive samples.

    A single-sample excursion is noise; a fault is persistent. This is the
    'fleeting alarm' suppression ISA-18.2 requires.
    """
    flags = df[flag_col].fillna(False).to_numpy()
    run = 0
    best = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    longest = best
    return Gate(
        name="persistence",
        passed=longest >= consecutive,
        detail=f"longest sustained excursion = {longest} samples (need {consecutive})",
        evidence={"longest_run": int(longest), "required": int(consecutive)},
    )


def climatology_consistency_gate(
    df: pd.DataFrame,
    wind_col: str = "Ws_avg",
    clim_mean: float | None = None,
    clim_sd: float | None = None,
    k: float = 3.0,
    period_col: str = "year",
) -> Gate:
    """Reject a period whose wind regime is outside the climatological envelope.

    If the period's own wind is far from the site's climatology, a residual in that
    period cannot be attributed to the turbine with any confidence -- the weather
    has swamped the signal. This is the gate that makes the false-alarm claim
    measurable rather than rhetorical.
    """
    if clim_mean is None or clim_sd is None or clim_sd == 0:
        return Gate(
            name="climatology_consistency",
            passed=True,
            detail="no climatology supplied -- gate skipped",
            evidence={},
        )
    per = df.groupby(period_col)[wind_col].mean()
    z = (per - clim_mean) / clim_sd
    worst = float(z.abs().max()) if len(z) else 0.0
    worst_period = str(z.abs().idxmax()) if len(z) else ""
    return Gate(
        name="climatology_consistency",
        passed=worst <= k,
        detail=f"worst period z = {worst:.2f} ({worst_period}), limit {k:.1f}",
        evidence={
            "worst_abs_z": round(worst, 3),
            "worst_period": worst_period,
            "k": k,
            "period_means": {str(i): round(float(v), 3) for i, v in per.items()},
        },
    )


def alarm_budget_gate(
    df: pd.DataFrame,
    flag_col: str = "flag",
    period_minutes: int = 10,
) -> Gate:
    """Check the raised alarms against an ISA-18.2 / IEC 62682 style budget.

    Two criteria: sustained rate below one alarm per 10 minutes on average, and the
    'flood' fraction of the reporting period below 1 %.
    """
    n = len(df)
    if n == 0:
        return Gate(name="alarm_budget", passed=False, detail="empty period", evidence={})
    alarms = int(df[flag_col].fillna(False).sum())
    periods = n * period_minutes / 10.0
    rate = alarms / periods if periods else float("inf")
    flood = alarms / n
    return Gate(
        name="alarm_budget",
        passed=(rate <= ALARM_BUDGET_PER_10MIN) and (flood <= FLOOD_FRACTION_MAX),
        detail=(
            f"{alarms} alarms in {n} samples = {rate:.3f}/10min "
            f"(limit {ALARM_BUDGET_PER_10MIN}), flood {flood*100:.2f}% "
            f"(limit {FLOOD_FRACTION_MAX*100:.0f}%)"
        ),
        evidence={
            "n_alarms": alarms,
            "n_samples": n,
            "rate_per_10min": round(rate, 4),
            "flood_fraction": round(flood, 5),
        },
    )


def apply_gates(
    df: pd.DataFrame,
    gates: list[Gate],
) -> pd.DataFrame:
    """AND the gates: an alarm is allowed only if every gate passes."""
    ok = all(g.passed for g in gates)
    out = df.copy()
    out["alarm_allowed"] = ok
    out["flag_final"] = out.get("flag", pd.Series(False, index=out.index)).fillna(False) & ok
    return out


def audit_record(
    alarm: pd.Series,
    gates: list[Gate],
    reference: dict,
    extra: dict | None = None,
) -> dict:
    """A reproducible record of why an alarm was (or was not) raised."""
    rec = {
        "alarm_raised": bool(alarm),
        "gates": [g.to_dict() for g in gates],
        "reference": reference,
    }
    if extra:
        rec.update(extra)
    return rec


def write_audit(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, default=str) + "\n")
    return path
