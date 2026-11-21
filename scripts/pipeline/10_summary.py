"""One summary table across every arm.

    python scripts/10_summary.py

Brings the two datasets and every arm into a single table, so the trade-off is
visible at a glance:

  * ENGIE: false-alarm rate on healthy turbines (the weather-shift arm)
  * CARE: coverage, accuracy, reliability, earliness and the final CARE score
    (the detection arm)

Outputs results/summary.json and prints the table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"


def load(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def engie_far(schemes: dict | None, multivariate: dict | None) -> dict:
    """(spread in pp, mean alarms/10 min) for the ENGIE arm."""
    out = {}
    if schemes:
        d = schemes.get("D_perturbine_density_band")
        a = schemes.get("A_pooled_static")
        if d:
            out["power_only_conditioned"] = {
                "spread_pp": d["spread_pp"],
                "mean_alarms_per_10min": d["mean_alarms_per_10min"],
                "min_rate_pct": d["min_rate_pct"],
                "max_rate_pct": d["max_rate_pct"],
            }
        if a:
            out["power_only_unconditioned"] = {
                "spread_pp": a["spread_pp"],
                "mean_alarms_per_10min": a["mean_alarms_per_10min"],
                "min_rate_pct": a["min_rate_pct"],
                "max_rate_pct": a["max_rate_pct"],
            }
    if multivariate:
        for name, s in (multivariate.get("strategies") or {}).items():
            out[name] = {
                "spread_pp": s["spread_pp"],
                "mean_rate_pct": s["mean_rate_pct"],
                "min_rate_pct": s["min_rate_pct"],
                "max_rate_pct": s["max_rate_pct"],
            }
    return out


def main() -> int:
    care = load("care_score.json")
    detection = load("detection.json")
    mv = load("multivariate.json")
    lg = load("lightgbm_nbm.json")
    sweep = load("threshold_sweep.json")

    arms = []

    if care:
        arms.append({
            "arm": "power-only residual (baseline)",
            "kind": "power-only",
            "engie_far": engie_far(detection.get("schemes") if detection else None, None)
            .get("power_only_conditioned"),
            "care": care["sub_scores"],
            "care_score": care["care_score"],
            "beats_AE": care["beats_published_AE"],
            "n_events": care["n_datasets_scored"],
        })
    if mv:
        for key, label, kind in (("max_combiner", "multivariate bins (max z)", "multivariate"),
                                 ("l2_combiner", "multivariate bins (L2)", "multivariate")):
            v = mv[key]
            arms.append({
                "arm": label,
                "kind": kind,
                "engie_far": engie_far(None, load("engie_multivariate.json")).get(
                    "rolling_30d"),
                "care": v["sub_scores"],
                "care_score": v["care_score"],
                "beats_AE": v.get("beats_published_AE"),
                "n_events": v["n_events"],
            })
    if lg:
        arms.append({
            "arm": lg["arm"],
            "kind": "lightgbm",
            # The learned arm's ENGIE false-alarm rate comes from its own
            # rolling-reference run, so the two detectors are compared on the
            # same footing rather than one having no ENGIE number at all.
            "engie_far": engie_far(None, load("engie_multivariate.json")).get(
                "rolling_30d_lightgbm"),
            "care": lg["sub_scores"],
            "care_score": lg["care_score"],
            "beats_AE": lg.get("beats_published_AE"),
            "n_events": lg["n_datasets"],
        })

    best = None
    if sweep:
        pool = sweep.get("leak_free") or sweep.get("per_threshold") or []
        if pool:
            best = max(pool, key=lambda r: r["care_score"])

    out = {
        "arms": arms,
        "best_multivariate_setting": best,
        "published_baselines": {"AE (published)": 0.66, "random": 0.50,
                                "iForest": "< 0.5"},
        "beats_published_AE": any(a["beats_AE"] for a in arms if a.get("beats_AE") is not None),
    }
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    # --- print --------------------------------------------------------------
    print("=" * 108)
    print("SUMMARY: every arm, both datasets")
    print("=" * 108)
    print(f"{'arm':<32}{'ENGIE FAR %':>14}{'alarms/10m':>12}"
          f"{'coverage':>10}{'accuracy':>10}{'reliab.':>9}{'early':>8}{'CARE':>8}{'events':>9}")
    print("-" * 108)
    for a in arms:
        far = a.get("engie_far")
        far_s = (f"{far['min_rate_pct']:.2f}-{far['max_rate_pct']:.2f}" if far else "-")
        if far and "mean_alarms_per_10min" in far:
            alm = f"{far['mean_alarms_per_10min']:.4f}"
        elif far:
            alm = f"{far['mean_rate_pct']:.3f}%"
        else:
            alm = "-"
        c = a["care"]
        print(f"{a['arm']:<32}{far_s:>14}{alm:>12}{c['coverage']:>10.4f}{c['accuracy']:>10.4f}"
              f"{c['reliability']:>9.4f}{c['earliness']:>8.4f}{a['care_score']:>8.4f}"
              f"{a.get('n_events', '?'):>9}")
    print("-" * 108)
    print(f"published baselines: AE 0.66 | random 0.50 | iForest < 0.5")
    if best:
        key = ("training quantile " + str(best["train_quantile"])
               if "train_quantile" in best else "z " + str(best["z_threshold"]))
        print(f"best leak-free multivariate setting: {best['combiner']} at {key} "
              f"-> CARE {best['care_score']:.4f}")
    print(f"\nwrote {RESULTS/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
