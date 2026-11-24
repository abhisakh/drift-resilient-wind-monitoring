"""The cost of a false call-out, priced against the value of an early warning.

The project's goal has two halves that pull against each other. A detector
can always be made quieter by making it less sensitive, and made to warn
earlier by making it more sensitive. So neither half can be priced alone:
a saving on false call-outs means nothing without saying what happens to
the faults, and a caught fault means nothing without saying how many false
call-outs were paid for it.

THE MODEL
---------
Per turbine, per year, a detector:

  * raises `n` false dispatches, each costing `D` (a technician, a vehicle,
    a few hours: the cost of going to look at a machine that was fine);
  * meets faults at a rate `lam` per turbine-year, and catches a share `p`
    of them early enough for a planned repair rather than a forced one,
    each catch being worth `V` (a planned repair against a failure that
    trips the turbine: the difference, not either figure alone).

    net value  =  lam * p * V  -  n * D

Only the DIFFERENCE between a forced failure and a planned repair appears,
because the cost of a fault that happens regardless cancels between any two
detectors being compared.

WHAT IS MEASURED AND WHAT IS ASSUMED
------------------------------------
`n` and `p` come from the project's own measurements: `n` from running the
monthly cross-check over every month of the four turbines, `p` from the
evaluation suite's fault injection. `D`, `V` and `lam` are NOT measured.
There is no maintenance log in the public data, so they are assumptions,
with placeholder defaults that the reader is expected to replace. The
sensitivity grid exists so that no conclusion has to rest on one guess.

This module is pure arithmetic: no data, no Streamlit, no I/O. The
dashboard, the README generator and the pipeline script all read it, so
they cannot disagree about what a number means.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: What one false call-out costs, in currency units. The dashboard's slider
#: opens on this figure; a test keeps the two in step.
DEFAULT_DISPATCH_COST = 2_000.0

#: What an early intervention is worth: a planned repair set against a
#: failure that trips the turbine. The same planning figure `cost_impact`
#: in `live.py` has always used, so the two views cannot disagree.
DEFAULT_EARLY_INTERVENTION_VALUE = 185_000.0

#: How often a developing fault turns up, per turbine per year. A
#: placeholder: the public data has too few faults to estimate it.
DEFAULT_FAULTS_PER_TURBINE_YEAR = 0.05

#: The axes of the default sensitivity grid. The dispatch costs span the
#: dashboard slider's range.
DISPATCH_COSTS = (500.0, 1_000.0, 2_000.0, 3_500.0, 5_000.0)
FAULT_RATES = (0.01, 0.02, 0.05, 0.10, 0.20)
EARLY_VALUES = (50_000.0, 185_000.0, 400_000.0)


@dataclass(frozen=True)
class CostAssumptions:
    """The three numbers that are assumed rather than measured."""

    dispatch_cost: float = DEFAULT_DISPATCH_COST
    early_intervention_value: float = DEFAULT_EARLY_INTERVENTION_VALUE
    faults_per_turbine_year: float = DEFAULT_FAULTS_PER_TURBINE_YEAR

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")

    def as_dict(self) -> dict:
        return asdict(self)


def _check_rate(n: float, name: str = "false dispatches") -> float:
    n = float(n)
    if n < 0:
        raise ValueError(f"{name} cannot be negative, got {n}")
    return n


def _check_probability(p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"a detection probability is in [0, 1], got {p}")
    return p


# ------------------------------------------------------------- the pieces
def false_dispatch_cost(false_dispatches_per_year: float,
                        a: CostAssumptions = CostAssumptions()) -> float:
    """What a detector's false call-outs cost per turbine-year."""
    n = _check_rate(false_dispatches_per_year)
    return n * a.dispatch_cost


def false_dispatch_saving(n_baseline: float, n_candidate: float,
                          dispatch_cost: float = DEFAULT_DISPATCH_COST) -> float:
    """The saving on false call-outs of the candidate over the baseline.

    Negative when the candidate raises MORE false dispatches. This prices
    one half of the goal only; see `annual_net_value` for both halves.
    """
    return (_check_rate(n_baseline) - _check_rate(n_candidate)) * float(dispatch_cost)


def expected_detection_value(detection_probability: float,
                             a: CostAssumptions = CostAssumptions()) -> float:
    """What the faults a detector catches are worth per turbine-year."""
    p = _check_probability(detection_probability)
    return a.faults_per_turbine_year * p * a.early_intervention_value


def annual_net_value(false_dispatches_per_year: float,
                     detection_probability: float,
                     a: CostAssumptions = CostAssumptions()) -> float:
    """What a detector is worth per turbine-year, against no monitoring.

    The value of the faults it catches early, less the cost of the false
    call-outs it makes along the way. Negative means it costs more than it
    saves, at these assumptions.
    """
    return (expected_detection_value(detection_probability, a)
            - false_dispatch_cost(false_dispatches_per_year, a))


# ---------------------------------------------------------- the break-evens
def break_even_fault_rate(false_dispatches_per_year: float,
                          detection_probability: float,
                          a: CostAssumptions = CostAssumptions()):
    """How often faults must occur, per turbine-year, for the detector to pay.

    Below this rate the false call-outs cost more than the early warnings
    save. None when the detector catches nothing (it can never pay) or an
    early intervention is worth nothing.
    """
    n = _check_rate(false_dispatches_per_year)
    p = _check_probability(detection_probability)
    if p <= 0.0 or a.early_intervention_value <= 0.0:
        return None
    return n * a.dispatch_cost / (p * a.early_intervention_value)


def break_even_dispatch_cost(n_a: float, n_b: float, p_a: float, p_b: float,
                             a: CostAssumptions = CostAssumptions()):
    """The cost of a false call-out at which two detectors are worth the same.

    The usual case: B is the quieter detector (n_b < n_a) and A the more
    sensitive (p_a > p_b). Then B is the better buy when a false call-out
    costs MORE than this, and A is when it costs less. This is the number
    that answers "how much must a wasted trip cost before it is worth
    missing some faults to avoid it?".

    None when the two raise the same number of false dispatches, because no
    dispatch cost can then separate them.
    """
    n_a, n_b = _check_rate(n_a), _check_rate(n_b)
    p_a, p_b = _check_probability(p_a), _check_probability(p_b)
    if n_a == n_b:
        return None
    return (a.faults_per_turbine_year * a.early_intervention_value
            * (p_a - p_b) / (n_a - n_b))


# ----------------------------------------------------------- sensitivity
def sensitivity_grid(false_dispatches_per_year: float,
                     detection_probability: float,
                     dispatch_costs=DISPATCH_COSTS,
                     fault_rates=FAULT_RATES,
                     early_values=EARLY_VALUES) -> list[dict]:
    """The annual net value over a grid of the three assumptions.

    One row per combination, so a reader can see how much of any conclusion
    rests on a guess. Nothing here is a finding about a currency amount; it
    is a map of where the answer changes sign.
    """
    rows = []
    for v in early_values:
        for lam in fault_rates:
            for d in dispatch_costs:
                a = CostAssumptions(dispatch_cost=d, early_intervention_value=v,
                                    faults_per_turbine_year=lam)
                rows.append({
                    "dispatch_cost": d,
                    "early_intervention_value": v,
                    "faults_per_turbine_year": lam,
                    "net_value": annual_net_value(
                        false_dispatches_per_year, detection_probability, a),
                })
    return rows


def share_positive(rows: list[dict]) -> float | None:
    """The share of a grid's cells in which the detector pays for itself."""
    if not rows:
        return None
    return sum(1 for r in rows if r["net_value"] > 0) / len(rows)


# ---------------------------------------------- pricing the operating curve
#: The dial the dashboard's dispatch rule implies, for "at the standing dial".
STANDING_RATE = 0.01
STANDING_PERSISTENCE = 6

#: One sample is ten minutes.
_SAMPLE_HOURS = 10.0 / 60.0


def _evaluate(p: dict, a: CostAssumptions, ramp_k: float):
    """One point of the operating curve, priced. None if it lacks the figures."""
    n = p.get("false_dispatches_per_turbine_year")
    ramps = p.get("ramps") or {}
    ramp = ramps.get(f"{ramp_k:g}") or {}
    det = ramp.get("detection_rate")
    if n is None or det is None:
        return None
    placebo = ramps.get("0") or {}
    return {
        "arm": p.get("arm"),
        "target_rate": p.get("target_rate"),
        "persistence_samples": p.get("persistence_samples"),
        "persistence_hours": round(
            p.get("persistence_samples", 0) * _SAMPLE_HOURS, 3),
        "false_dispatches_per_turbine_year": n,
        "detection_rate": det,
        "median_days_to_dispatch": ramp.get("median_days_to_dispatch"),
        "chance_rate": placebo.get("detection_rate"),
        "false_dispatch_cost": round(false_dispatch_cost(n, a), 2),
        "expected_detection_value": round(
            expected_detection_value(det, a), 2),
        "net_value": round(annual_net_value(n, det, a), 2),
        "break_even_fault_rate": break_even_fault_rate(n, det, a),
    }


def grid_extent(points: list[dict], arm: str):
    """The (lowest, highest) sensitivity and persistence an arm was swept over."""
    rates = sorted({p["target_rate"] for p in points if p.get("arm") == arm})
    pers = sorted({p["persistence_samples"] for p in points
                   if p.get("arm") == arm})
    if not rates or not pers:
        return None
    return rates[0], rates[-1], pers[0], pers[-1]


def setting_at(points: list[dict], arm: str, rate: float, persistence: int,
               a: CostAssumptions = CostAssumptions(), ramp_k: float = 4.0):
    """One named setting of an arm, priced; None if it was not swept."""
    for p in points:
        if (p.get("arm") == arm and p.get("target_rate") == rate
                and p.get("persistence_samples") == persistence):
            return _evaluate(p, a, ramp_k)
    return None


def best_setting(points: list[dict], arm: str,
                 a: CostAssumptions = CostAssumptions(), ramp_k: float = 4.0):
    """The setting of an arm that is worth the most under these assumptions.

    The dial is chosen to maximise annual net value, so an arm is judged at
    its own most profitable setting rather than at one setting shared by
    every arm. The choice is made on the same weeks it is scored on, so the
    figure is optimistic, equally for every arm.

    `at_grid_edge` names any edge of the swept grid the choice sits on: an
    optimum at the edge means the sweep may have been too narrow to find the
    real one.
    """
    best = None
    for p in points:
        if p.get("arm") != arm:
            continue
        cand = _evaluate(p, a, ramp_k)
        if cand is None:
            continue
        key = (cand["net_value"], -cand["false_dispatches_per_turbine_year"])
        if best is None or key > best[0]:
            best = (key, cand)
    if best is None:
        return None
    out = best[1]
    ext = grid_extent(points, arm)
    edges = []
    if ext:
        lo_r, hi_r, lo_m, hi_m = ext
        if out["target_rate"] == hi_r:
            edges.append("most sensitive rate swept")
        if out["target_rate"] == lo_r:
            edges.append("least sensitive rate swept")
        if out["persistence_samples"] == hi_m:
            edges.append("longest persistence swept")
        if out["persistence_samples"] == lo_m:
            edges.append("shortest persistence swept")
    out["at_grid_edge"] = edges
    return out


def net_value_curve(points: list[dict], arm: str, fault_rates,
                    a: CostAssumptions = CostAssumptions(),
                    ramp_k: float = 4.0) -> list[dict]:
    """An arm's best net value as the fault rate varies.

    At each fault rate the arm is given its own most profitable setting, so
    the curve shows where monitoring starts to pay (it crosses zero) and how
    quickly it pays more.
    """
    rows = []
    for lam in fault_rates:
        b = best_setting(points, arm, CostAssumptions(
            a.dispatch_cost, a.early_intervention_value, lam), ramp_k)
        if b is None:
            continue
        rows.append({
            "faults_per_turbine_year": lam,
            "net_value": b["net_value"],
            "target_rate": b["target_rate"],
            "persistence_samples": b["persistence_samples"],
        })
    return rows


def share_of_assumptions_where_best(points: list[dict], arms,
                                    dispatch_costs=DISPATCH_COSTS,
                                    fault_rates=FAULT_RATES,
                                    early_values=EARLY_VALUES,
                                    ramp_k: float = 4.0) -> dict:
    """For each arm, the share of the assumption grid in which it is the most
    profitable of all the arms, each at its own best setting.

    Any arm can be turned quiet enough that it beats no monitoring at all, so
    that is not a useful test. Whether the RANKING of the arms survives a
    change of assumptions is. Ties are shared out equally.
    """
    wins = {arm: 0.0 for arm in arms}
    total = 0
    for v in early_values:
        for lam in fault_rates:
            for d in dispatch_costs:
                a = CostAssumptions(d, v, lam)
                nets = {}
                for arm in arms:
                    b = best_setting(points, arm, a, ramp_k)
                    if b is not None:
                        nets[arm] = b["net_value"]
                if not nets:
                    continue
                total += 1
                top = max(nets.values())
                leaders = [arm for arm, n in nets.items() if n == top]
                for arm in leaders:
                    wins[arm] += 1.0 / len(leaders)
    if total == 0:
        return {arm: None for arm in arms}
    return {arm: round(w / total, 4) for arm, w in wins.items()}


def share_where_best_in_the_worse_case(shapes: dict[str, list[dict]], arms,
                                       dispatch_costs=DISPATCH_COSTS,
                                       fault_rates=FAULT_RATES,
                                       early_values=EARLY_VALUES,
                                       ramp_k: float = 4.0) -> dict:
    """For each arm, the share of the assumption grid in which it has the best
    WORSE-CASE value across the fault shapes.

    The shape of a real fault is not known, and the order of the arms depends
    on it. So each arm is scored by the smaller of its net values across the
    shapes (each at its own best setting for that shape), and the arm with
    the best such score leads. This is the choice that holds up whichever
    shape the fault turns out to have.
    """
    wins = {arm: 0.0 for arm in arms}
    total = 0
    for v in early_values:
        for lam in fault_rates:
            for d in dispatch_costs:
                a = CostAssumptions(d, v, lam)
                worst = {}
                for arm in arms:
                    nets = []
                    for pts in shapes.values():
                        b = best_setting(pts, arm, a, ramp_k)
                        if b is not None:
                            nets.append(b["net_value"])
                    if len(nets) == len(shapes):
                        worst[arm] = min(nets)
                if not worst:
                    continue
                total += 1
                top = max(worst.values())
                leaders = [arm for arm, n in worst.items() if n == top]
                for arm in leaders:
                    wins[arm] += 1.0 / len(leaders)
    if total == 0:
        return {arm: None for arm in arms}
    return {arm: round(w / total, 4) for arm, w in wins.items()}
