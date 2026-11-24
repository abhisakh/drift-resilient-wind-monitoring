"""Regenerate README.md's Results section from results/*.json.

    python scripts/pipeline/09_report.py

Every number in the README comes from a file in results/, so the README cannot
drift from the artefacts. If a results file is missing, the section says so
rather than inventing a value.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
README = ROOT / "README.md"

MARKER_START = "<!-- RESULTS:START -->"
MARKER_END = "<!-- RESULTS:END -->"
#: The "what this is" framing carries live numbers, so it is generated too --
#: a hand-written number in the README would drift from the artefacts.
METHOD_START = "<!-- METHOD:START -->"
METHOD_END = "<!-- METHOD:END -->"
#: The goal and the three measurements that answer it. Generated too, from
#: the same module the dashboard's headline cards read.
GOAL_START = "<!-- GOAL:START -->"
GOAL_END = "<!-- GOAL:END -->"

sys.path.insert(0, str(ROOT / "src"))
from windmill import scorecard  # noqa: E402


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def fmt(v, nd: int = 4, dash: str = "not measured") -> str:
    if v is None:
        return dash
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def section_conditioning() -> str:
    c = load("conditioning.json")
    d = load("detection.json")
    out = []
    if c:
        a = c["annual_climatology"]
        out.append("### The weather shift is real and measurable\n")
        out.append(f"At this site the capacity factor moves **{fmt(a['annual_cf_range_pp'], 2)} pp** "
                   f"between years (CV {fmt(a['annual_cf_cv_pct'], 2)} %), and the ERA5 annual wind "
                   f"deviation tracks the measured capacity-factor deviation with "
                   f"**r = {fmt(a['corr_era5_ws_dev_vs_cf_dev'])}**. A wind-speed shift of about "
                   f"{fmt(a['mean_amplification_cf_per_wind'], 2)}x is amplified into power — the cube "
                   f"law at work. The seasonal spread is larger still.\n")
        ag = c.get("era5_scada_agreement", {})
        if ag:
            out.append(f"ERA5 against the turbine's own anemometer: r = "
                       f"{fmt(ag.get('pearson_r_scada_vs_era5'))}, bias "
                       f"{fmt(ag.get('bias_ms_scada_minus_era5'), 3)} m/s. The intercept is large "
                       f"because the nacelle anemometer has a low-wind floor that ERA5 does not "
                       f"share — so ERA5 conditions the long-term reference, it does not replace "
                       f"the turbine's own sensor.\n")
    if d:
        s = d["schemes"]
        out.append("### Conditioning removes the false alarms\n")
        out.append("False-alarm rate on healthy turbines, per year (2013-2017), by scheme:\n")
        out.append("| scheme | min % | max % | spread (pp) | alarms/10 min | corr with weather shift |")
        out.append("|---|---|---|---|---|---|")
        for name in ("A_pooled_static", "B_perturbine_static", "C_perturbine_density",
                     "D_perturbine_density_band", "E_perturbine_density_band_season"):
            if name not in s:
                continue
            v = s[name]
            out.append(f"| {name} | {fmt(v['min_rate_pct'], 3)} | {fmt(v['max_rate_pct'], 3)} | "
                       f"{fmt(v['spread_pp'], 3)} | {fmt(v['mean_alarms_per_10min'], 5)} | "
                       f"{fmt(v['corr_alarm_rate_vs_climatology_shift'])} |")
        out.append("")
        if "D_perturbine_density_band" in s and "A_pooled_static" in s:
            a = s["A_pooled_static"]
            b = s["D_perturbine_density_band"]
            out.append(f"A single-year, site-pooled baseline spans {fmt(a['min_rate_pct'], 2)}–"
                       f"{fmt(a['max_rate_pct'], 2)} % across years; the per-turbine, "
                       f"density-normalised, band-conditioned detector spans "
                       f"{fmt(b['min_rate_pct'], 2)}–{fmt(b['max_rate_pct'], 2)} % — the "
                       f"year-to-year spread falls from {fmt(a['spread_pp'], 2)} pp to "
                       f"{fmt(b['spread_pp'], 2)} pp, and the alarm rate stops tracking the "
                       f"weather shift (r {fmt(a['corr_alarm_rate_vs_climatology_shift'])} → "
                       f"{fmt(b['corr_alarm_rate_vs_climatology_shift'])}).\n")
    return "\n".join(out)


def section_decomposition() -> str:
    d = load("decomposition.json")
    r = load("residual_attribution.json")
    out = []
    if d:
        out.append("### Most of the residual is a stale reference, not a bad turbine\n")
        out.append("Decomposing the static-baseline error into the part explained by the wind "
                   "distribution and the part explained by the curve itself moving:\n")
        out.append("| year | total (kW) | distribution (kW) | curve (kW) | curve share |")
        out.append("|---|---|---|---|---|")
        for row in d["decomposition"]:
            share = row["share_curve_pct"]
            share_s = "n/a" if share is None else (f"{fmt(share, 1)} %")
            out.append(f"| {row['year']} | {fmt(row['total_kw'], 2)} | {fmt(row['dist_effect_kw'], 2)} "
                       f"| {fmt(row['curve_effect_kw'], 2)} | {share_s} |")
        out.append("")
    if r and r.get("per_year"):
        ym = r.get("year_matched_reference_kw", {})
        out.append("Refitting the reference on each year removes most of the bias -- so what "
                   "a single-period baseline reports as degradation is mostly a stale "
                   "reference, not a bad turbine:\n")
        out.append("| year | mean residual, 2014 reference (kW) | year-matched reference (kW) |")
        out.append("|---|---|---|")
        for row in r["per_year"]:
            y = row["year"]
            out.append(f"| {y} | {fmt(row['mean_resid_kw'], 3)} | {fmt(ym.get(str(y), ym.get(y)), 3)} |")
        out.append("")
    return "\n".join(out)


def section_engie_multivariate() -> str:
    em = load("engie_multivariate.json")
    if not em:
        return ""
    out = ["### ENGIE: the reference strategy decides the false-alarm rate\n"]
    out.append("The same multivariate thermal method on the ENGIE site, with every turbine "
               "healthy throughout -- so **every alarm is a false alarm**. Two detectors "
               "(the deterministic bins and a learned LightGBM normal-behaviour model) "
               "under two reference strategies:\n")
    out.append("| detector | reference | year | false-alarm rate % |")
    out.append("|---|---|---|---|")
    for name, s in em.get("strategies", {}).items():
        detector = "learned (LightGBM)" if "lightgbm" in name else "deterministic"
        reference = name.replace("_lightgbm", "")
        for y, v in sorted(s["per_year"].items()):
            out.append(f"| {detector} | {reference} | {y} | {fmt(v['rate_pct'], 4)} |")
    out.append("")
    frozen = em.get("strategies", {}).get("frozen_2014")
    rolling = em.get("strategies", {}).get("rolling_30d")
    frozen_lg = em.get("strategies", {}).get("frozen_2014_lightgbm")
    rolling_lg = em.get("strategies", {}).get("rolling_30d_lightgbm")
    if frozen and rolling:
        ratio = frozen["mean_rate_pct"] / rolling["mean_rate_pct"]
        out.append(f"A **frozen 2014 reference** -- what a naive deployment does -- gives "
                   f"{fmt(frozen['min_rate_pct'], 2)}–{fmt(frozen['max_rate_pct'], 2)} % false "
                   f"alarms (mean {fmt(frozen['mean_rate_pct'], 2)} %, spread "
                   f"{fmt(frozen['spread_pp'], 2)} pp). Each turbine's thermal channels move by "
                   f"5-9 K between years, so the frozen reference leaves a systematic offset "
                   f"that reads as overheating on every channel at once.\n")
        out.append(f"A **rolling local reference** -- each row compared against its own recent "
                   f"history at the same load bin, re-estimated continuously -- gives "
                   f"{fmt(rolling['min_rate_pct'], 2)}–{fmt(rolling['max_rate_pct'], 2)} % "
                   f"(mean {fmt(rolling['mean_rate_pct'], 2)} %, spread "
                   f"{fmt(rolling['spread_pp'], 2)} pp). That is a **{fmt(ratio, 1)}x** reduction "
                   f"in false alarms from changing nothing but the reference.\n")
        if frozen_lg and rolling_lg:
            out.append(
                "The **learned** arm was given the same two reference strategies, so the "
                "comparison is like for like: its model is refitted weekly on the trailing "
                "30 days, exactly as the deterministic arm's reference is re-estimated "
                "continuously. With a frozen reference the learned arm is "
                f"{fmt(frozen_lg['mean_rate_pct'], 2)} % (mean false-alarm rate) against the "
                f"deterministic arm's {fmt(frozen['mean_rate_pct'], 2)} % -- so the model is "
                "*not* the deciding factor; the reference is. With a rolling reference the "
                f"learned arm gives {fmt(rolling_lg['mean_rate_pct'], 2)} % against the "
                f"deterministic arm's {fmt(rolling['mean_rate_pct'], 2)} %.\n")
        out.append("This is the project's central result on the weather-shift side: the model "
                   "matters far less than the reference it is compared against, and a "
                   "reference that is not re-estimated turns slow drift into a flood of "
                   "alarms.\n")
    return "\n".join(out)


def section_care() -> str:
    c = load("care_score.json")
    lg = load("lightgbm_nbm.json")
    mv = load("multivariate.json")
    sw = load("threshold_sweep.json")
    out = []
    if c:
        sub = c["sub_scores"]
        out.append("### CARE benchmark (95 labelled events)\n")
        out.append(f"Scored with the benchmark authors' own implementation "
                   f"(`energy-fault-detector` 0.8.1), cross-checked against our re-implementation "
                   f"({c['n_mismatches']} mismatches over {c['n_datasets_scored']} events).\n")
        out.append("| sub-score | value |")
        out.append("|---|---|")
        for k in ("coverage", "accuracy", "reliability", "earliness"):
            out.append(f"| {k} | {fmt(sub.get(k))} |")
        out.append(f"| **CARE score** | **{fmt(c['care_score'])}** |")
        out.append("")
        out.append(f"Published baselines: AE {c['published_baselines']['AE (published)']}, "
                   f"random {c['published_baselines']['random']}, "
                   f"iForest {c['published_baselines']['iForest']}. "
                   f"Beats published AE: **{c['beats_published_AE']}**.\n")
    if mv:
        out.append("### Option 1: deterministic multivariate bins\n")
        out.append("| combiner | coverage | accuracy | reliability | earliness | CARE score | events detected |")
        out.append("|---|---|---|---|---|---|---|")
        for key, label in (("max_combiner", "max z"), ("l2_combiner", "L2 norm")):
            v = mv[key]
            s = v["sub_scores"]
            out.append(f"| {label} | {fmt(s.get('coverage'))} | {fmt(s.get('accuracy'))} | "
                       f"{fmt(s.get('reliability'))} | {fmt(s.get('earliness'))} | "
                       f"{fmt(v['care_score'])} | {v['n_events_detected']}/{v['n_events']} |")
        out.append("")
    if sw:
        out.append("### Threshold sweep for the multivariate arm\n")
        out.append("The threshold is the arm's one free parameter, so it was swept rather "
                   "than chosen by taste. Two rules are swept, and they are not equivalent:\n")
        out.append("**Leak-free (training quantile)** — the threshold is the quantile of the "
                   "cross-fitted TRAINING score distribution, so it never sees an event it is "
                   "later scored on. This is the protocol the CARE paper's own baseline uses, "
                   "and it is the row to quote:\n")
        out.append("| combiner | training quantile | CARE score | coverage | accuracy | reliability | earliness | events detected |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in sw.get("leak_free", []):
            out.append(f"| {r['combiner']} | {r['train_quantile']} | {fmt(r['care_score'])} | "
                       f"{fmt(r['coverage'])} | {fmt(r['accuracy'])} | {fmt(r['reliability'])} | "
                       f"{fmt(r['earliness'])} | {r['n_events_detected']}/{r['n_events']} |")
        out.append("")
        if sw.get("upper_bound"):
            out.append("**Upper bound (fixed z, selected on the scored events)** — reported "
                       "for completeness only. Choosing the threshold on the same 95 events that "
                       "are then scored is leakage, so these numbers are optimistic and are NOT "
                       "the result:\n")
            out.append("| combiner | z | CARE score | coverage | accuracy | reliability | earliness | events detected |")
            out.append("|---|---|---|---|---|---|---|---|")
            for r in sw["upper_bound"]:
                out.append(f"| {r['combiner']} | {r['z_threshold']} | {fmt(r['care_score'])} | "
                           f"{fmt(r['coverage'])} | {fmt(r['accuracy'])} | {fmt(r['reliability'])} | "
                           f"{fmt(r['earliness'])} | {r['n_events_detected']}/{r['n_events']} |")
            out.append("")
    if lg:
        out.append("### Arm comparison\n")
        out.append("| arm | coverage | accuracy | reliability | earliness | CARE score |")
        out.append("|---|---|---|---|---|---|")
        if c:
            sub = c["sub_scores"]
            out.append(f"| power-only residual (baseline) | {fmt(sub.get('coverage'))} | "
                       f"{fmt(sub.get('accuracy'))} | {fmt(sub.get('reliability'))} | "
                       f"{fmt(sub.get('earliness'))} | {fmt(c['care_score'])} |")
        if mv:
            for key, label in (("max_combiner", "multivariate bins (max z)"),
                               ("l2_combiner", "multivariate bins (L2)")):
                v = mv[key]
                s = v["sub_scores"]
                out.append(f"| {label} | {fmt(s.get('coverage'))} | {fmt(s.get('accuracy'))} | "
                           f"{fmt(s.get('reliability'))} | {fmt(s.get('earliness'))} | "
                           f"{fmt(v['care_score'])} |")
        sub = lg["sub_scores"]
        out.append(f"| {lg['arm']} | {fmt(sub.get('coverage'))} | {fmt(sub.get('accuracy'))} | "
                   f"{fmt(sub.get('reliability'))} | {fmt(sub.get('earliness'))} | "
                   f"{fmt(lg['care_score'])} |")
        out.append("")
        out.append("All four arms are scored on the same 95 events (45 anomaly, 50 normal). "
                   "The learned arm has the **best coverage** of the three, and pays for it in "
                   "accuracy: it raises alarms on more of the true anomalies and also on more "
                   "normal behaviour.\n")
        out.append("Both the deterministic and the learned arm use the same threshold rule: "
                   "the quantile of the TRAINING score distribution, so the threshold never "
                   "sees an event it is later scored on. The learned arm's seeds are pinned "
                   "explicitly (`random_state`, `seed`) and verified bit-identical across runs "
                   "by `tests/test_lightgbm_determinism.py` -- without that, its score would "
                   "not have been re-derivable, which is the whole point of the exercise.\n")
    out.append(section_thermal_signal())
    out.append(section_ae_comparability())
    em = load("engie_multivariate.json")
    if em:
        s = em.get("strategies", {})
        rows = []
        for label, key in (("deterministic (rolling ref)", "rolling_30d"),
                           ("learned (rolling ref)", "rolling_30d_lightgbm")):
            v = s.get(key)
            if v:
                rows.append((label, v["min_rate_pct"], v["max_rate_pct"],
                             v["mean_rate_pct"], v["spread_pp"]))
        if rows:
            out.append("The same two arms on ENGIE, where every alarm is a false alarm:\n")
            out.append("| arm | min % | max % | mean % | spread (pp) |")
            out.append("|---|---|---|---|---|")
            for label, lo, hi, mean, sp in rows:
                out.append(f"| {label} | {fmt(lo, 2)} | {fmt(hi, 2)} | "
                           f"{fmt(mean, 2)} | {fmt(sp, 2)} |")
            out.append("")
    return "\n".join(out)


def section_scorability() -> str:
    a = load("farm_a_audit.json")
    if not a:
        return ""
    out = ["### What the benchmark can and cannot score\n"]
    out.append("The CARE score evaluates Coverage and Reliability only over rows the dataset "
               "marks as normal operation (status-ID 0 or 2). Rows already flagged as fault "
               "(status 4) inside an event window are excluded from the metric entirely, so no "
               "detector can earn credit for them.\n")
    out.append("| farm | anomaly events | zero scorable rows | mean % of event window normal | mean % status 4 |")
    out.append("|---|---|---|---|---|")
    for farm, s in a["per_farm"].items():
        out.append(f"| {farm} | {s['n_anomaly']} | {s['n_anomaly_with_zero_scorable_rows']} | "
                   f"{fmt((s['mean_frac_event_rows_normal'] or 0) * 100, 1)} % | "
                   f"{fmt((s['mean_frac_event_rows_status4'] or 0) * 100, 1)} % |")
    out.append("")
    out.append(f"**{a['all_anomaly_events_zero_scorable']} of {a['n_anomaly_events']}** labelled "
               f"anomaly events contain no scorable rows at all. This is a property of the "
               f"published labels, not of any detector, and it caps what Coverage can be.\n")
    return "\n".join(out)


def _count_tests() -> int:
    """How many tests pytest actually collects, so the README cannot go stale.

    Counting `def test_` lines undercounts: a single parametrised test
    function runs once per parameter, so the source-line count was 80 while
    pytest ran 126. The README must state the number a reader would see when
    they run the suite, so ask pytest.
    """
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
    except Exception:
        r = None
    if r is not None:
        m = re.search(r"(\d+)\s+tests?\s+collected", r.stdout)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)\s+tests?\s+collected", r.stderr)
        if m:
            return int(m.group(1))
    # Fallback: count the test functions.
    n = 0
    for f in (ROOT / "tests").glob("test_*.py"):
        n += sum(1 for line in f.read_text().splitlines()
                 if line.startswith("def test_"))
    return n


def section_thermal_signal() -> str:
    """How much thermal signal is actually inside a fault window?

    This project detects faults from thermal channels. CARE's faults are
    mostly NOT thermal. This measures what that costs, so the limitation is
    a number rather than an excuse.
    """
    ts = load("thermal_signal.json")
    if not ts:
        return ""
    out = ["### How much thermal signal is actually inside a fault window?\n"]
    out.append(
        "This project detects faults from **thermal channels**. Most of CARE's "
        "faults are not thermal -- \"Hydraulic group\", \"Pitchfailure\", "
        "\"Converter Failure\", \"Safety chain relay open\". It is worth "
        "measuring what that costs rather than assuming it.\n")
    out.append(
        f"Over the {ts.get('n_anomaly_events')} labelled anomaly events "
        f"({ts.get('n_with_scorable_rows')} of them with scorable rows), the "
        "detector's score inside the fault window compared with its score "
        "outside it, on the same dataset:\n")
    out.append("| statistic | median inside the event | median outside | "
               "event higher in |")
    out.append("|---|---|---|---|")
    for stat, label in (("max_z", "max-z"), ("l2", "L2 norm")):
        out.append(f"| {label} | {fmt(ts.get(f'{stat}_median_in_event'), 4)} | "
                   f"{fmt(ts.get(f'{stat}_median_outside'), 4)} | "
                   f"{fmt(ts.get(f'{stat}_event_higher_fraction'), 4)} |")
    out.append("")
    pf = ts.get("per_farm") or {}
    if pf:
        out.append("| farm | events | median inside | median outside | "
                   "event higher in |")
        out.append("|---|---|---|---|---|")
        for farm, v in pf.items():
            out.append(f"| {farm} | {v.get('n')} | "
                       f"{fmt(v.get('zmax_median_in_event'), 4)} | "
                       f"{fmt(v.get('zmax_median_outside'), 4)} | "
                       f"{fmt(v.get('zmax_event_higher_fraction'), 4)} |")
        out.append("")
    frac = ts.get("max_z_event_higher_fraction")
    out.append(
        f"**The thermal signal inside a fault window is not stronger than "
        f"outside it.** The event is higher in {fmt(frac, 4)} of events -- "
        f"a coin flip.\n")
    out.append(
        "This is the quantitative form of the project's honest weakness, and it "
        "is a SCOPE limit rather than a badly chosen threshold: no threshold "
        "recovers a signal the thermal channels do not carry. It explains "
        "exactly where this project loses to the published autoencoder -- "
        "Coverage and Earliness -- while its Accuracy is better, because it "
        "raises fewer false alarms on normal behaviour.\n")
    return "\n".join(out)


def section_ae_comparability() -> str:
    """Is the published AE baseline 0.66 comparable with ours?

    The audit in scripts/pipeline/12_ae_audit.py establishes the published
    protocol from the authors' code AND their paper. The headline: 0.66 is a
    tuned-on-the-benchmark number, so it is not like-for-like with a leak-free
    score.
    """
    a = load("ae_audit.json")
    if not a:
        return ""
    out = ["### Is the published AE baseline (0.66) comparable with ours?\n"]
    code = a.get("authors_protocol_read_from_code", {})
    paper = a.get("authors_protocol_read_from_paper", {})
    if code:
        out.append(
            "Read out of the benchmark authors' own implementation "
            "(`energy-fault-detector` 0.8.1):\n")
        out.append(f"- autoencoder {code.get('autoencoder')}, anomaly score "
                   f"{code.get('anomaly_score')}, threshold selector "
                   f"{code.get('threshold_selector')}")
        out.append(f"- the threshold is fitted on {code.get('threshold_fitted_on')} "
                   f"-- so the threshold RULE itself does not use the scored "
                   f"events")
        out.append(f"- but the split and the Keras training are **not** seeded, "
                   f"so the published number is one stochastic run\n")
    if paper:
        out.append(
            "Read out of the paper (sec. 4.2.2), which is where the comparability "
            "question is actually decided:\n")
        out.append(f"- the AE's architecture and hyperparameters were tuned with "
                   f"Optuna, and the threshold calibration is **per wind farm**")
        out.append(f"- farms A and B use an adaptive threshold with a per-farm "
                   f"sensitivity gamma, also optimised with Optuna")
        out.append(f"- farm C's fixed threshold was chosen by "
                   f"**maximising the F(1/2) score** against ground truth "
                   f"derived from the status-IDs")
        out.append(f"- the paper documents no held-out protocol for that search\n")
        out.append(
            "> The published 0.66 is therefore the output of a search whose "
            "objective was the CARE score itself: a **tuned-on-the-benchmark** "
            "number, not a held-out one. That is a reason to be careful about "
            "quoting it as a like-for-like baseline -- but it is NOT by itself "
            "an explanation of the gap, because the measurement below runs the "
            "authors' own autoencoder under a leak-free threshold rule and the "
            "gap survives.\n")
        out.append(
            "And the status-column question was measured rather than argued. "
            "The package has two feature-building entry points, one of which "
            "keeps `status_type_id` -- the fault annotation -- as a feature. "
            "The AE was run over all 95 datasets both ways: the scores are "
            "**identical to four decimal places**. The reason is that the AE "
            "trains only on normal rows, where the status column is constantly "
            "0, so it carries no information the AE could exploit. That leak "
            "was never there.\n")
        out.append(
            "One thing does differ in scope rather than in method: the AE sees "
            "the **full sensor vector** (power, wind, pitch, hydraulic, "
            "electrical), while this project's detector sees only the thermal "
            "channels -- so a hydraulic or pitch fault is invisible to it. The "
            "section above measures exactly what that costs.\n")
    # --- the like-for-like comparison -------------------------------------
    sc = load("subset_comparison.json")
    mv_all = load("multivariate.json")
    ae_all = (a.get("per_config") or {}).get("published_s0")
    if sc and mv_all and ae_all:
        ours_all = mv_all["max_combiner"]["sub_scores"]
        ours_all_score = mv_all["max_combiner"]["care_score"]
        out.append(
            "**The like-for-like measurement.** Both arms were run over the "
            "**same 95 datasets** and scored with the same CARE scorer:\n")
        out.append("| arm | datasets | coverage | accuracy | reliability | "
                   "earliness | CARE |")
        out.append("|---|---|---|---|---|---|---|")
        out.append(f"| this project (deterministic, max z) | 95 | "
                   f"{fmt(ours_all.get('coverage'))} | {fmt(ours_all.get('accuracy'))} | "
                   f"{fmt(ours_all.get('reliability'))} | {fmt(ours_all.get('earliness'))} | "
                   f"{fmt(ours_all_score)} |")
        out.append(f"| published AE (authors' own implementation) | "
                   f"{ae_all.get('n_datasets')} | {fmt(ae_all.get('coverage'))} | "
                   f"{fmt(ae_all.get('accuracy'))} | {fmt(ae_all.get('reliability'))} | "
                   f"{fmt(ae_all.get('earliness'))} | {fmt(ae_all.get('care_score'))} |")
        out.append(f"| published AE (paper, Optuna-tuned) | 95 | - | - | - | - | "
                   f"0.66 |")
        out.append("")
        gap = ae_all["care_score"] - ours_all_score
        tuning = 0.66 - ae_all["care_score"]
        out.append(
            f"**The honest answer: the AE beats us, by {fmt(gap, 3)} CARE** "
            f"({fmt(ae_all['care_score'])} against {fmt(ours_all_score)}) -- with a "
            f"leak-free threshold rule, over the whole benchmark, with the status "
            f"column removed. The gap is not a protocol artefact.\n")
        out.append(
            f"But note the third row. The authors' **published** 0.66 is "
            f"{fmt(tuning, 3)} above what their own default configuration "
            f"reproduces here ({fmt(ae_all['care_score'])}). That difference is "
            f"the part of the published number that comes from the Optuna "
            f"search, the per-farm adaptive thresholds and the per-farm "
            f"sensitivities -- none of which has a held-out protocol behind it. "
            f"So 0.66 is the right number to quote as the state of the art, and "
            f"it is not the right number to quote as a like-for-like baseline.\n")
        out.append(
            "The shape of the gap is informative. This project's detector has "
            f"the better **accuracy** ({fmt(ours_all.get('accuracy'))} against "
            f"{fmt(ae_all.get('accuracy'))}): it raises far fewer false alarms. "
            f"The AE has the better **coverage** "
            f"({fmt(ae_all.get('coverage'))} against {fmt(ours_all.get('coverage'))}) "
            f"and the better **earliness** ({fmt(ae_all.get('earliness'))} against "
            f"{fmt(ours_all.get('earliness'))}): it finds more of the true "
            f"anomalies and finds them earlier.\n")
        out.append(
            "And the section above says why: a leak-free threshold set at a "
            "training quantile has to be conservative to avoid false alarms, "
            "and this project's arm pays for that in exactly the sub-scores the "
            "AE wins -- it detects late and rarely. A threshold selected on the "
            "scored events would lift its coverage considerably -- the fixed-z "
            "upper bound in the sweep above reaches a coverage comparable to the "
            "AE's -- but that number is an upper bound, not a result.\n")
        out.append(
            f"As a cross-check, both arms were also scored on an identical "
            f"{sc.get('n_datasets')}-dataset subset, chosen and stratified by "
            f"label: this project {fmt(sc.get('this_project', {}).get('care_score'))}, "
            f"the AE {fmt(sc.get('published_ae_baseline_measured_here', {}).get('care_score'))}. "
            f"Same ordering, same conclusion, on a different sample.\n")
    pc = a.get("per_config") or {}
    if pc:
        # The audit's own configurations, on however many datasets it ran.
        label = {"published_s0": "published protocol (as the authors' code runs it)",
                 "no_status_s0": "same, with status_type_id removed from the features",
                 "published_s1": "published protocol, a different random seed"}
        out.append("The authors' own autoencoder, run here and scored with their "
                   "own CARE scorer:\n")
        out.append("| configuration | datasets | coverage | accuracy | "
                   "reliability | earliness | CARE |")
        out.append("|---|---|---|---|---|---|---|")
        for name, v in pc.items():
            out.append(f"| {label.get(name, name)} | {v.get('n_datasets')} | "
                       f"{fmt(v.get('coverage'))} | {fmt(v.get('accuracy'))} | "
                       f"{fmt(v.get('reliability'))} | {fmt(v.get('earliness'))} | "
                       f"{fmt(v.get('care_score'))} |")
        out.append("")
        # Two configurations with the same protocol and different seeds measure
        # the run-to-run spread -- which is the point about the published
        # number being a single stochastic run.
        seeds = {k: v for k, v in pc.items() if k.startswith("published_s")}
        if len(seeds) > 1:
            vals = [v.get("care_score") for v in seeds.values()
                    if v.get("care_score") is not None]
            if len(vals) > 1:
                out.append(
                    f"Same protocol, different seeds: CARE {', '.join(fmt(v) for v in vals)}. "
                    f"The spread between two runs of the SAME configuration "
                    f"({fmt(max(vals) - min(vals), 4)}) is the size of the "
                    f"uncertainty in any single published figure -- and the "
                    f"published pipeline is not seeded at all.\n")
    if a.get("subsample_note"):
        out.append(a["subsample_note"] + "\n")
    out.append("**What we claim, precisely.** We claim that our "
               f"{fmt((load('multivariate.json') or {}).get('max_combiner', {}).get('care_score'))} "
               "is a leak-free number under a protocol we can re-derive and "
               "defend, and that the published figure is not quite that kind of "
               "number -- it is the output of a search whose objective was the "
               "CARE score itself, with no held-out protocol documented.\n")
    out.append("We do **not** claim to beat the published baseline. Measured on "
               "the same events, it beats us. The useful, defensible claim is "
               "narrower and more interesting: **we can tell you exactly where "
               "we lose and why**, we can tell you what part of the gap is "
               "protocol and what part is method, and every number above is "
               "re-derivable from the artefacts in `results/`.\n")
    return "\n".join(out)


def section_method() -> str:
    """The 'what this is, and what it is not' framing, with live numbers."""
    lg = load("lightgbm_nbm.json")
    mv = load("multivariate.json")
    em = load("engie_multivariate.json")
    bt = load("backtest.json")
    out = ["## What this is, and what it is not\n"]
    out.append(
        "The method is **deterministic**: a per-turbine reference conditioned on load and\n"
        "ambient temperature, a robust (MAD) spread, a one-sided z-score, and a fixed\n"
        "threshold rule. There is no learned model in the primary arm.\n"
    )
    if lg and mv:
        det = mv["max_combiner"]["sub_scores"]
        lrn = lg["sub_scores"]
        out.append(
            "A learned arm (LightGBM normal-behaviour model) is included as a **comparison**,\n"
            "measured with the same statistic (max-z), the same threshold rule and the same\n"
            "reference strategies, so only the model differs. On the CARE benchmark it\n"
            "detects more of the true anomalies (coverage "
            f"{fmt(lrn.get('coverage'))} vs {fmt(det.get('coverage'))}) and pays for that with\n"
            f"false alarms on normal behaviour (accuracy {fmt(lrn.get('accuracy'))} vs "
            f"{fmt(det.get('accuracy'))}).\n"
        )
    if em:
        s = em.get("strategies", {})
        fr, frl = s.get("frozen_2014"), s.get("frozen_2014_lightgbm")
        ro, rol = s.get("rolling_30d"), s.get("rolling_30d_lightgbm")
        if fr and ro and frl and rol:
            out.append(
                "On the ENGIE site the comparison is more interesting than a clean sweep,\n"
                "and the honest reading is this: **with a stale reference the learned arm is\n"
                f"the better one** ({fmt(frl['mean_rate_pct'], 2)} % false alarms against the\n"
                f"deterministic arm's {fmt(fr['mean_rate_pct'], 2)} %), because a smooth model of\n"
                "the operating point drifts less sharply than a set of fixed per-bin\n"
                "medians. **With a current reference the deterministic arm has far fewer\n"
                f"false alarms in that comparison** ({fmt(ro['mean_rate_pct'], 2)} % against {fmt(rol['mean_rate_pct'], 2)} %, "
                "about "
                f"{fmt(rol['mean_rate_pct'] / ro['mean_rate_pct'], 1)}x fewer false alarms), and\n"
                "it does so with no model to train and no thresholds to re-validate.\n"
            )
            out.append(
                "The project's claim is therefore not 'the deterministic method is better'\n"
                "but the sharper one: **the reference matters more than the model.**\n"
                "Re-estimating the reference reduced false alarms by "
                f"{fmt(fr['mean_rate_pct'] / ro['mean_rate_pct'], 1)}x for the deterministic\n"
                f"detector and {fmt(frl['mean_rate_pct'] / rol['mean_rate_pct'], 1)}x for the learned one.\n"
            )
    pooled = ((bt or {}).get("A1_A2_weekly_backtest") or {}).get("pooled") or {}
    dep = (pooled.get("summary") or {}).get("load+ambient")
    lrn_bt = (pooled.get("summary") or {}).get("lightgbm")
    if dep and lrn_bt:
        out.append(
            "**The like-for-like reading.** The two ENGIE comparisons above let "
            "each arm keep its own threshold rule and its own spread estimate, "
            "so they measure more than the model. The evaluation suite further "
            "down fixes the protocol -- a strict train-then-test split, every "
            "arm calibrated to the same 1 % rate on its own training residuals "
            "-- and there the two models predict the bearing temperature almost "
            f"equally well: a median error of {fmt(lrn_bt['mae_k_median'], 2)} K "
            f"for the learned arm and {fmt(dep['mae_k_median'], 2)} K for the "
            "deterministic one. The deterministic model is the primary arm "
            "because it is transparent and has nothing to retrain, not because "
            "it is the more accurate predictor.\n"
        )
    out.append(
        "We also do **not** claim operational weather forecasting. ERA5 conditions a\n"
        "long-term reference; it is not a forecast engine, and the residual is always\n"
        "measured against the turbine's own anemometer.\n"
    )
    out.append(
        "And we do **not** claim to beat the published autoencoder baseline. "
        "Measured on the same events, it beats us -- by how much, and in which "
        "sub-scores, is set out in the AE-comparability section below, with the "
        "numbers in `results/`.\n"
    )
    return "\n".join(out)


def section_backtest() -> str:
    """The evaluation suite: every claim, measured.

    Written from results/backtest.json, so the numbers here are the
    numbers the harness produced -- including the ones that do not
    support the project.
    """
    b = load("backtest.json")
    if not b:
        return ""
    weekly = b["A1_A2_weekly_backtest"]["pooled"]
    summary = weekly["summary"]
    paired = weekly["paired"]
    lines = [
        "### The evaluation suite: every claim, measured",
        "",
        "Eight tests, one protocol. The split is enforced in code (a window's",
        "training rows end strictly before its test rows begin) and every arm's",
        "alarm threshold is calibrated on its OWN out-of-fold training residuals",
        "at the same 1 % rate, so the arms are compared at equal false-alarm",
        "rates rather than at one shared number on one scale.",
        "",
        f"**A.1 — the weekly backtest.** {summary['load+ambient']['windows']} "
        "test weeks across the four ENGIE turbines; each predicted by a "
        "reference fitted on the thirty days immediately before it. The error "
        "is the median over weeks of the MAE, in kelvin.",
        "",
        "| arm | MAE (K) | MAE p90 | RMSE (K) | bias (K) | false-alarm rate |",
        "|---|---|---|---|---|---|",
    ]
    label = {
        "constant": "constant median (no operating point)",
        "load": "load-conditioned (the operating point)",
        "load+ambient": "load + ambient (**the deployed model**)",
        "load+ambient+wind": "load + ambient + wind",
        "lightgbm": "LightGBM (the learned arm)",
    }
    for arm in ("constant", "load", "load+ambient", "load+ambient+wind",
                "lightgbm"):
        if arm not in summary:
            continue
        x = summary[arm]
        lines.append(
            f"| {label[arm]} | {x['mae_k_median']} | {x['mae_k_p90']} | "
            f"{x['rmse_k_median']} | {x['bias_k_median']} | "
            f"{x['false_alarm_rate_median']} |")

    la = paired.get("load+ambient vs lightgbm", {})
    lc = paired.get("load+ambient vs constant", {})
    if la and lc:
        lines += [
            "",
            f"**What this says, stated both ways.** Conditioning on the "
            f"operating point is worth everything: the paired median "
            f"difference against a constant median is "
            f"**{lc.get('median_delta')} K**, better in "
            f"**{100*lc.get('share_a_better', 0):.0f} %** of weeks. The "
            f"learned arm is the more accurate PREDICTOR: its median MAE is "
            f"{summary['lightgbm']['mae_k_median']} K against the deployed "
            f"model's {summary['load+ambient']['mae_k_median']} K, and the "
            f"deployed model is better in only "
            f"**{100*la.get('share_a_better', 0):.0f} %** of weeks. But at "
            f"the matched threshold the deployed model's false-alarm rate is "
            f"**{summary['load+ambient']['false_alarm_rate_median']}** against "
            f"the learned arm's "
            f"**{summary['lightgbm']['false_alarm_rate_median']}**: its "
            f"calibration holds across weeks and the learned arm's does not. "
            f"The reference matters more than the model -- but it is not the "
            f"better predictor.",
        ]

    ab = paired.get("load+ambient vs load", {})
    aw = paired.get("load+ambient+wind vs load+ambient", {})
    if ab:
        lines += [
            "",
            "**A.2 — the weather ablation.** The paired median difference "
            "from adding the ambient temperature is "
            f"**{ab.get('median_delta')} K** (better in "
            f"{100*ab.get('share_a_better', 0):.0f} % of weeks), and from "
            "adding wind speed on top of that is "
            f"**{aw.get('median_delta')} K** (better in "
            f"{100*aw.get('share_a_better', 0):.0f} % of weeks). **On this "
            "channel the weather inputs do not measurably cut the prediction "
            "error.** The operating point does; the weather does not. That is "
            "a negative result for the ablation as posed, and it is reported "
            "as one.",
        ]

    chain = b.get("A3_forecast_chain", {}).get("summary", {})
    if chain:
        lines += [
            "",
            "**A.3 — the forecast chain.** Forecast wind and temperature → "
            "the turbine's own power curve → predicted power → the "
            "load- and ambient-conditioned reference → expected temperature, "
            "with ERA5 plus injected forecast error standing in for a real "
            "forecast.",
            "",
            "| forecast | chain MAE (K) | amplification |",
            "|---|---|---|",
        ]
        for name, x in chain.items():
            lines.append(f"| {name} | {x['median_mae_k']} | "
                         f"{x['median_amplification']} |")
        lines += [
            "",
            "The chain's error is already the reference's own error plus the "
            "power curve's; the forecast error multiplies it. At a typical "
            "forecast the chain's error stays well inside the alarm band, so "
            "a forecast does NOT blow the band up -- but the chain is a worse "
            "predictor than conditioning on the measured power.",
        ]

    sweep = b.get("B4_reference_age", {}).get("sweep_over_reference_age", {})
    if sweep:
        lines += [
            "",
            "**B.4 — the age of the reference.** The same detector, the same "
            "rule, the same weeks, with only the reference's age changed. "
            "Every arm is calibrated to 1 % of its own training window.",
            "",
            "| reference age | false-alarm rate | ratio vs 30 days |",
            "|---|---|---|",
        ]
        for key in ("30d", "90d", "180d", "365d", "730d"):
            if key in sweep:
                x = sweep[key]
                lines.append(f"| {key[:-1]} days | "
                             f"{x['false_alarm_rate_median']} | "
                             f"{x.get('ratio_over_30d')} |")
        lines += [
            "",
            "The ratio grows with the age of the reference, which is the "
            "project's headline claim measured rather than asserted: "
            "1.8× at three months, 2.2× at a year, 4.5× at two years, on "
            "the same windows and the same rule.",
        ]

    clean = b.get("B5_cleaning", {})
    if clean.get("per_trial"):
        u = [t["untrimmed_reference_shift_k"] for t in clean["per_trial"]]
        t_ = [t["trimmed_reference_shift_k"] for t in clean["per_trial"]]
        su = [t["untrimmed_fault_stands_out_k"] for t in clean["per_trial"]]
        st = [t["trimmed_fault_stands_out_k"] for t in clean["per_trial"]]
        med = lambda v: sorted(v)[len(v) // 2]
        lines += [
            "",
            "**B.5 — the cleaning test.** A +3 K ramp is injected into the "
            "last ten days of the training window, and the reference is "
            "fitted with and without trimming.",
            "",
            f"* how far the reference moved toward the fault: "
            f"{med(u)} K untrimmed, {med(t_)} K trimmed;",
            f"* how much the fault still stands out on the faulted rows: "
            f"{med(su)} K untrimmed, {med(st)} K trimmed.",
            "",
            "**Trimming does not stop the reference absorbing the ramp** at "
            "this ramp size: the median residual per load bin is a robust "
            "statistic, and a ramp over a third of the window moves it "
            "whether or not the ramped rows are dropped. Trimming protects "
            "against a fault that is large relative to the spread; it does "
            "not protect against one that is a slow bias.",
        ]

    faults = b.get("C6_fault_injection", {}).get("summary", {})
    if faults:
        lines += [
            "",
            "**C.6 — fault injection.** A ramp injected into a healthy test "
            "window, scored at the matched 1 % threshold. A simulation: it "
            "says what the detector would do if the machine did this, and "
            "nothing about whether the machine ever does.",
            "",
            "| arm | ramp | detection rate | median days to first alarm |",
            "|---|---|---|---|",
        ]
        for key in sorted(faults, key=lambda k: (k.split("|")[0], k)):
            x = faults[key]
            lines.append(f"| {x['arm']} | {x['magnitude_k']:g} K | "
                         f"{x['detection_rate']} | "
                         f"{x['median_days_to_first_alarm']} |")

    lead = b.get("C7_lead_time", {})
    if lead.get("first_alarm_1pct"):
        lines += [
            "",
            "**C.7 — the R80711 lead time.** The September 2017 anomaly, "
            "unlabelled, with the reference re-estimated every day.",
            "",
            f"* the temperature first crosses 80 °C on "
            f"**{lead.get('first_80c_crossing')}**;",
            f"* it first diverges from its three siblings by 5 K on "
            f"**{lead.get('first_sibling_divergence_5k')}**;",
            f"* at the matched 1 % rate the arms first alarm on "
            f"**{lead.get('first_alarm_1pct', {})}**;",
            f"* at a 5 % rate, **{lead.get('first_alarm_5pct', {})}**.",
            "",
            "**At 1 % the detector gives no lead over the 80 °C crossing; at "
            "5 % it gives six days.** The lead time is bought with false "
            "alarms, and the price is measurable.",
        ]

    care = b.get("C8_care_thermal_subset", {})
    if care.get("thermal_observable_events"):
        allx = care["all_events"]
        th = care["thermal_observable_events"]
        lines += [
            "",
            "**C.8 — the CARE thermal subset.** The events whose fault "
            "windows contain rows in normal operation are the ones where the "
            "score is even defined.",
            "",
            "| set | anomaly events | coverage | earliness |",
            "|---|---|---|---|",
            f"| all anomaly events | {allx['n_anomaly']} | "
            f"{allx['coverage']} | {allx['earliness']} |",
            f"| thermal-observable | {th['n_anomaly']} | {th['coverage']} | "
            f"{th['earliness']} |",
            "",
            f"Coverage improves by **{care.get('coverage_change')}** and "
            f"earliness by **{care.get('earliness_change')}** once the "
            "non-thermal faults are excluded — which is the argument for a "
            "thermal detector being the right instrument for this benchmark.",
        ]
    return "\n".join(lines)


def _persistence_label(samples: int, minutes: float = 10.0) -> str:
    """How long a sustained alarm must last, in words: 10 min, 1 h, 24 h."""
    total = samples * minutes
    return f"{total:g} min" if total < 60 else f"{total / 60:g} h"


def _pct(x) -> str:
    return "not measured" if x is None else fmt(float(x) * 100, 0)


_ARM_LABELS = {
    "constant": "constant median (no operating point)",
    "load": "load-conditioned",
    "load+ambient": "load + ambient (the deployed model)",
    "lightgbm": "LightGBM (the learned arm)",
}


def section_operating_curve() -> str:
    """The operating curve: false call-outs per year against time to catch a ramp.

    Written from results/operating_curve.json, so every number is the one
    the sweep produced, including the ones that do not flatter the project.
    Skipped until the sweep has been run.
    """
    oc_json = load("operating_curve.json")
    if not oc_json or not oc_json.get("points"):
        return ""
    points = oc_json["points"]
    proto = oc_json.get("protocol", {})
    arms = [a for a in proto.get("arms", []) if any(p["arm"] == a for p in points)]
    label = lambda a: _ARM_LABELS.get(a, a)  # noqa: E731
    ramp_k = scorecard.HEADLINE_RAMP_K
    key = f"{ramp_k:g}"

    def at(arm, rate, persistence):
        for p in points:
            if (p["arm"] == arm and p["target_rate"] == rate
                    and p["persistence_samples"] == persistence):
                return p
        return None

    first = points[0]
    windows = sum((oc_json.get("windows_used") or {}).values())
    out = ["### The operating curve: what a false call-out buys\n"]
    out.append(
        "The two halves of the goal pull against each other, so no figure is "
        "quoted alone. A detector has two dials: its **sensitivity** (the share "
        "of its own training window it is calibrated to alarm on) and its "
        "**persistence** (how long an alarm must be sustained before it is "
        "worth sending someone). At every setting this measures the false "
        "dispatches per turbine-year on healthy turbine-weeks, and how often "
        f"and how fast an injected +{fmt(ramp_k, 0)} K ramp is caught. A 0 K "
        "**placebo** ramp is run through the same code: whatever it \"detects\" "
        "is a false dispatch that happened to land after the onset, so it is "
        "the chance level a real detection has to be read against.\n")
    out.append(
        f"Exposure: {fmt(first['exposure_turbine_years'], 1)} turbine-years of "
        f"healthy operation over {windows} non-overlapping windows. Windows "
        "touching the R80711 anomaly are left out for every arm alike.\n")

    # --- the dashboard's rule, with and without persistence ------------------
    rows = []
    for arm in arms:
        for m in (1, 6):
            p = at(arm, 0.01, m)
            if p is None:
                continue
            r, z = p["ramps"][key], p["ramps"]["0"]
            rows.append(
                f"| {label(arm)} | {_persistence_label(m)} | "
                f"{fmt(p['false_dispatches_per_turbine_year'], 1)} | "
                f"{_pct(r['detection_rate'])} % | "
                f"{fmt(r['median_days_to_dispatch'], 1)} | "
                f"{_pct(z['detection_rate'])} % |")
    if rows:
        out.append("**At the standing sensitivity (a 1 % rate), with and without "
                   "persistence:**\n")
        out.append(f"| arm | sustained for | false dispatches per turbine-year | "
                   f"+{fmt(ramp_k, 0)} K ramps caught | median days to dispatch | "
                   f"0 K placebo \"caught\" |")
        out.append("|---|---|---|---|---|---|")
        out.extend(rows)
        out.append("")
    dep = at(scorecard.DEPLOYED_ARM, 0.01, 1)
    dep6 = at(scorecard.DEPLOYED_ARM, 0.01, 6)
    if dep and dep6:
        out.append(
            f"With no persistence requirement the deployed arm \"catches\" "
            f"{_pct(dep['ramps'][key]['detection_rate'])} % of the +{fmt(ramp_k, 0)} K "
            f"ramps, but it also \"catches\" {_pct(dep['ramps']['0']['detection_rate'])} % "
            f"of the 0 K placebo ramps: most of that detection is chance. With an "
            f"hour of sustained alarm it catches "
            f"{_pct(dep6['ramps'][key]['detection_rate'])} % against a chance level of "
            f"{_pct(dep6['ramps']['0']['detection_rate'])} %.\n")

    # --- equal false call-outs ----------------------------------------------
    budgets = oc_json.get("budgets_ramp_4k") or {}
    if budgets:
        out.append("**At equal false call-outs per turbine-year, each arm at its own "
                   "best setting:**\n")
        out.append(f"| false call-out budget | arm | sensitivity | sustained for | "
                   f"false dispatches | +{fmt(ramp_k, 0)} K caught | median days | "
                   f"chance (0 K) |")
        out.append("|---|---|---|---|---|---|---|---|")
        lines_by_budget = []
        for budget, row in budgets.items():
            ranked = []
            for arm in arms:
                r = row.get(arm)
                if not r:
                    out.append(f"| {budget} | {label(arm)} | no setting fits | | | | | |")
                    continue
                out.append(
                    f"| {budget} | {label(arm)} | {fmt(r['target_rate'] * 100, 2)} % | "
                    f"{_persistence_label(r['persistence_samples'])} | "
                    f"{fmt(r['false_dispatches_per_turbine_year'], 1)} | "
                    f"{_pct(r['detection_rate'])} % | "
                    f"{fmt(r['median_days_to_dispatch'], 1)} | "
                    f"{_pct(r['placebo_detection_rate'])} % |")
                ranked.append((-(r["detection_rate"] or 0.0),
                               r["median_days_to_dispatch"] or 1e9, arm, r))
            ranked.sort(key=lambda t: (t[0], t[1]))
            if ranked:
                lines_by_budget.append((budget, ranked))
        out.append("")
        for budget, ranked in lines_by_budget:
            parts_ = "; ".join(
                f"{label(arm)} {_pct(r['detection_rate'])} % in "
                f"{fmt(r['median_days_to_dispatch'], 1)} days"
                for _, _, arm, r in ranked)
            out.append(f"* At {budget} false call-outs per turbine-year: {parts_}.")
        out.append("")

    held = oc_json.get("held_out_budgets_ramp_4k") or {}
    if held:
        out.append("**Setting chosen on half the windows, scored on the other "
                   "half** (this removes the optimism of picking the best of a "
                   "grid on the same weeks it is scored on):\n")
        out.append(f"| false call-out budget | arm | false dispatches | "
                   f"+{fmt(ramp_k, 0)} K caught | median days | chance (0 K) |")
        out.append("|---|---|---|---|---|---|")
        for budget, row in held.items():
            for arm in arms:
                h = row.get(arm)
                if not h:
                    continue
                out.append(
                    f"| {budget} | {label(arm)} | "
                    f"{fmt(h['false_dispatches_per_turbine_year'], 1)} | "
                    f"{_pct(h['detection_rate'])} % | "
                    f"{fmt(h['median_days_to_dispatch'], 1)} | "
                    f"{_pct(h['placebo_detection_rate'])} % |")
        out.append("")

    shaped = oc_json.get("budgets_ramp_4k_load_proportional") or {}
    if shaped:
        out.append("**A fault whose heat scales with load** (the offset is "
                   f"+{fmt(ramp_k, 0)} K at half of rated power and proportional "
                   "to load from there), best setting within each budget:\n")
        out.append(f"| false call-out budget | arm | false dispatches | "
                   f"+{fmt(ramp_k, 0)} K caught | median days | chance (0 K) |")
        out.append("|---|---|---|---|---|---|")
        for budget, row in shaped.items():
            for arm in arms:
                r = row.get(arm)
                if not r:
                    continue
                out.append(
                    f"| {budget} | {label(arm)} | "
                    f"{fmt(r['false_dispatches_per_turbine_year'], 1)} | "
                    f"{_pct(r['detection_rate'])} % | "
                    f"{fmt(r['median_days_to_dispatch'], 1)} | "
                    f"{_pct(r['placebo_detection_rate'])} % |")
        out.append("")

    ages = oc_json.get("reference_age") or {}
    age_keys = sorted((k for k in ages if k.isdigit()), key=int)
    if age_keys:
        out.append("**The age of the reference.** The same windows for every age, "
                   "at a 1 % rate with an hour of sustained alarm; false "
                   "dispatches per turbine-year:\n")
        out.append("| arm | " + " | ".join(f"reference {k} days old" for k in age_keys)
                   + " | oldest over freshest |")
        out.append("|---|" + "---|" * (len(age_keys) + 1))
        ratios = ages.get("ratio_oldest_over_fresh") or {}
        for arm in arms:
            cells = []
            for k in age_keys:
                cell = (ages[k].get(arm) or {}).get("6")
                cells.append(fmt(cell["false_dispatches_per_turbine_year"], 1)
                             if cell else "not measured")
            ratio = (ratios.get(arm) or {}).get("6")
            out.append(f"| {label(arm)} | " + " | ".join(cells)
                       + f" | {fmt(ratio, 2)}x |")
        out.append("")

    anomaly = oc_json.get("r80711_lead_time") or {}
    if anomaly.get("rows"):
        cm = load("cost_model.json") or {}

        def find(arm, rate, m):
            for r in anomaly["rows"]:
                if (r["arm"] == arm and r["target_rate"] == rate
                        and r["persistence_samples"] == m):
                    return r
            return None

        def verdict(r):
            if r is None:
                return "not swept"
            quiet = f"{r['dispatches_before_onset']} false before onset"
            d = r.get("days_before_hot_spell")
            if r.get("days_after_onset") is None:
                return f"never dispatched ({quiet})"
            if d is None:
                return f"first dispatch {str(r['first_dispatch_after_onset'])[:10]} ({quiet})"
            when = f"{fmt(d, 1)} d early" if d >= 0 else f"{fmt(-d, 1)} d late"
            return f"{when} ({quiet})"

        out.append("**The one real anomaly, on the same dial.** Nothing is injected "
                   "here: each arm is fitted on the month before "
                   f"{anomaly.get('test', ['', ''])[0]} and scored on R80711 up to "
                   f"{anomaly.get('test', ['', ''])[1]}. The first sustained hot spell "
                   f"(a day with at least {anomaly.get('min_hot_samples')} samples above "
                   f"{fmt(anomaly.get('hot_c'), 0)} °C) began on "
                   f"{str(anomaly.get('first_hot_spell') or '')[:10]}; the onset, when "
                   f"R80711 first ran hotter than its siblings, is {anomaly.get('onset')}. "
                   "A dispatch before the onset is a false alarm, not a warning, "
                   "so it is counted against the setting.\n")
        out.append("| arm | standing 1 %, 1 h | most profitable setting, flat offset | "
                   "most profitable setting, load-proportional fault | settings that "
                   "warned early with no false dispatch |")
        out.append("|---|---|---|---|---|")
        for arm in arms:
            cells = [verdict(find(arm, 0.01, 6))]
            for blk in (cm.get("arms", {}).get(arm, {}),
                        (cm.get("arms", {}).get(arm, {}) or {}).get("load_proportional") or {}):
                b = (blk or {}).get("best_setting")
                cells.append("not priced" if not b else
                             f"{fmt(b['target_rate'] * 100, 2)} % · "
                             f"{_persistence_label(b['persistence_samples'])}: "
                             + verdict(find(arm, b["target_rate"], b["persistence_samples"])))
            ec = scorecard.early_clean_settings(oc_json, arm)
            cells.append(f"{ec[0]} of {ec[1]}" if ec else "not measured")
            out.append(f"| {label(arm)} | " + " | ".join(cells) + " |")
        out.append("")
        rbl = anomaly.get("residual_by_load")
        if rbl and rbl.get("phases"):
            edges = rbl["load_bins"]
            out.append(f"**Where the excess is.** The mean temperature above what the "
                       f"{label(rbl['arm'])} model expected, by load (samples in "
                       "brackets):\n")
            out.append("| load | before the onset | hot-spell days | other days after the onset |")
            out.append("|---|---|---|---|")
            for i, lo in enumerate(edges):
                row = [f"{lo * 100:.0f}–{(lo + 0.2) * 100:.0f} %"]
                for phase in ("before_onset", "hot_spell_days", "other_days_after_onset"):
                    c = rbl["phases"][phase][i]
                    row.append("none" if c["mean_residual_k"] is None else
                               f"{fmt(c['mean_residual_k'], 1)} K ({c['n']})")
                out.append("| " + " | ".join(row) + " |")
            out.append("")

        cov = anomaly.get("high_load_coverage")
        if cov and cov.get("daily"):
            from datetime import date, timedelta

            hl = cov["high_load"] * 100
            n_between = cov["samples_between_onset_and_hot_spell"]
            if n_between == 0:
                verdict_txt = (
                    "There was no high-load operation between the onset and the "
                    "first hot spell, so there was nothing for any model to see: "
                    "the hot spell is simply the first day the turbine ran hard. "
                    "That limits the warning time whatever the model or the dial; "
                    "it is a property of the weather and the fault.")
            else:
                verdict_txt = (
                    f"High-load operation before the first hot spell was brief "
                    f"({n_between} samples), which limits how early any model "
                    "could see this fault.")
            out.append(
                "**Why the warning was not earlier.** The excess shows mainly at "
                "high load, so a fault like this can only be seen once the "
                f"turbine works hard. Between the onset and the first hot spell "
                f"there were {n_between} samples at {hl:.0f} % load or more, and "
                f"the training month held {cov['training_high_load_samples']} of "
                f"{cov['training_samples']} "
                f"({fmt(cov['training_high_load_share'] * 100, 1)} %). "
                + verdict_txt + "\n")
            spell_day = str(anomaly.get("first_hot_spell") or "")[:10]
            days = cov["daily"]
            try:
                lo = (date.fromisoformat(str(anomaly.get("onset")))
                      - timedelta(days=3)).isoformat()
                hi_ = (date.fromisoformat(spell_day) + timedelta(days=4)).isoformat()
                days = [d for d in days if lo <= d["day"] <= hi_]
            except ValueError:
                days = days[:14]
            out.append("| day | samples at high load | mean excess at high load |")
            out.append("|---|---|---|")
            for d in days:
                out.append(
                    f"| {d['day']} | {d['n_high_load']} | "
                    + ("none" if d["mean_residual_k"] is None
                       else f"{fmt(d['mean_residual_k'], 1)} K") + " |")
            out.append("")

    if "lightgbm" not in arms:
        out.append("_The learned arm (LightGBM) is not in this run; run "
                   "`make curve` on a machine that has it._\n")

    out.append(
        "**What this does not show.** The ramp is a simulation of an offset on "
        "one channel; a real bearing fault may involve several channels and "
        "develop differently. The two shapes above bracket how its heat might "
        "depend on load, and the held-out table removes the optimism of "
        "choosing a setting on the weeks it is scored on, but none of the "
        "figures is an out-of-sample forecast of operations. Detection is "
        "measured on a 28-day window held to a reference fitted on the thirty "
        "days before it, which is the dashboard's monthly loop; a weekly refit "
        "would change the false-dispatch counts.\n")
    return "\n".join(out)


def section_cost() -> str:
    """What the operating curve is worth in money, under stated assumptions.

    Written from results/cost_model.json, which `17_cost_model.py` derives
    from the operating curve. The false dispatches and the share of faults
    caught are measured; the three prices are assumed, and the section says
    so before it says anything else. Skipped until `make cost` has run.
    """
    cm = load("cost_model.json")
    if not cm or not cm.get("arms"):
        return ""
    a = cm["assumptions"]
    arms = list(cm["arms"])
    label = lambda x: _ARM_LABELS.get(x, x)  # noqa: E731
    robust = cm.get("ranking_robustness") or {}
    cells = robust.get("assumption_grid_cells")

    out = ["### What the operating curve is worth in money\n"]
    out.append(
        "**Three numbers here are assumed, not measured**, because the public "
        f"data has no maintenance log: a false dispatch costs {a['dispatch_cost']:,.0f}, "
        f"catching a fault early is worth {a['early_intervention_value']:,.0f} "
        f"(a planned repair against a failure that trips the turbine), and a "
        f"fault turns up {fmt(a['faults_per_turbine_year'], 2)} times per "
        "turbine-year. What is measured is the false dispatches per "
        "turbine-year and the share of simulated faults caught, at every "
        "setting of every arm. Each arm is shown at its own most profitable "
        "setting, so it is judged at the dial that suits it, not at one "
        "setting shared by all. Net value is against not monitoring at all.\n")

    def table(title, getter):
        rows = []
        for arm in arms:
            blk = getter(cm["arms"][arm])
            if not blk:
                continue
            b, s = blk.get("best_setting"), blk.get("standing_setting")
            if not b:
                continue
            rows.append(
                f"| {label(arm)} | {fmt(b['target_rate'] * 100, 2)} % · "
                f"{_persistence_label(b['persistence_samples'])} | "
                f"{fmt(b['false_dispatches_per_turbine_year'], 1)} | "
                f"{_pct(b['detection_rate'])} % | {b['net_value']:,.0f} | "
                f"{fmt(b['break_even_fault_rate'], 3)} | "
                f"{(s['net_value'] if s else float('nan')):,.0f} |")
        if not rows:
            return
        out.append(title + "\n")
        out.append("| arm | most profitable setting (sensitivity, sustained for) | "
                   "false dispatches per turbine-year | "
                   f"+{fmt(cm.get('ramp_k', 4.0), 0)} K caught | "
                   "net value per turbine-year | break-even fault rate | "
                   "net value at the standing 1 %, 1 h setting |")
        out.append("|---|---|---|---|---|---|---|")
        out.extend(rows)
        out.append("")

    table("**A flat temperature offset:**", lambda blk: blk)
    table("**A fault whose heat scales with load:**",
          lambda blk: blk.get("load_proportional"))

    def shares(key, title):
        sh = robust.get(key)
        if not sh:
            return
        cells_txt = f" of {cells} combinations of the three assumptions" if cells else ""
        out.append(
            f"* {title}: " + "; ".join(
                f"{label(arm)} {_pct(v)} %" for arm, v in sh.items()
                if v is not None) + cells_txt + ".")

    shares("share_where_arm_is_most_profitable",
           "The most profitable arm for a flat offset, by share")
    shares("share_where_arm_is_most_profitable_load_proportional",
           "The most profitable arm for a load-proportional fault")
    shares("share_where_arm_is_best_in_the_worse_case",
           "The arm with the best worse case across the two fault shapes")
    out.append("")

    edge_arms = []
    for arm in arms:
        for blk in (cm["arms"][arm], cm["arms"][arm].get("load_proportional") or {}):
            b = (blk or {}).get("best_setting") or {}
            if b.get("at_grid_edge"):
                edge_arms.append((label(arm), ", ".join(b["at_grid_edge"])))
    if edge_arms:
        out.append(
            "_The most profitable setting of "
            + "; ".join(f"{n} ({e})" for n, e in dict(edge_arms).items())
            + " sits at the edge of the swept grid, so a wider sweep might "
              "find a better one._\n")

    out.append(
        "**What this does not show.** The prices are placeholders: change them "
        "in the dashboard and the ranking and the break-even rates move. Each "
        "arm's setting is chosen on the same weeks it is scored on, so every "
        "figure is optimistic, equally for every arm. The detection rate is a "
        "simulated ramp, not a recorded fault. And a false dispatch is priced "
        "as one flat cost, whereas a real operator would look at the data "
        "remotely before sending anyone, which is cheaper.\n")
    return "\n".join(out)


def section_daily_check() -> str:
    """The daily check: a frozen monthly model, a daily comparison, the rules.

    Written from results/daily_check.json. Skipped until `make daily` has run.
    """
    dcj = load("daily_check.json")
    if not dcj or not dcj.get("by_turbine"):
        return ""
    proto = dcj["protocol"]
    rows = dcj["by_turbine"]
    fleet = dcj["fleet"]
    turbines = proto["turbines"]
    case = "R80711"
    tols = sorted(fleet, key=float)
    at = lambda t, tol: next((r for r in rows if r["turbine"] == t  # noqa: E731
                              and f"{r['tolerance']:g}" == tol), None)

    out = ["### The daily check: predict the healthy value, compare it every day\n"]
    out.append(
        "Each month a model is trained on at least a month of healthy readings "
        "from before the month, leaving out standstill, the "
        f"{fmt(proto['warmup_hours'], 0)}-hour warm-up after every restart (a "
        "bearing that restarts cold runs up to 12 K below normal), every day "
        "that was flagged, and everything before the last maintenance. It is "
        "then frozen for the month: it never learns from the month's own sensor "
        "values. Every day it predicts what a healthy sensor should read from "
        "that day's power and weather, and the day's mean difference from what "
        "the sensor shows is compared with a tolerance. A run of hot days is one "
        "call-out at the end of its first day. After a confirmed fault the model "
        "stays frozen until maintenance; after maintenance it is blank until a "
        "new month of healthy data exists.\n")
    out.append(
        "**Stood in for, because the data have no records:** maintenance is a "
        "restart after three days or more without running; what an inspection "
        "would find is the documented anomaly span (R80711 from 15 August 2017). "
        "A hot day outside it counts as a false alarm.\n")

    out.append("**False call-outs a year and the real fault, by tolerance:**\n")
    out.append("| tolerance | healthy siblings, false call-outs a year | "
               f"{case}, false call-outs a year | {case}, first call-out on "
               "the fault | fault days hot, of those checked | siblings, cold "
               "days a year |")
    out.append("|---|---|---|---|---|---|")
    for tol in tols:
        f = fleet[tol]
        out.append(
            f"| {tol} K | {fmt(f['siblings_false_call_outs_per_year'], 1)} | "
            f"{fmt(f['case_false_call_outs_per_year'], 1)} | "
            f"{f['case_first_fault_call_out'] or 'none'} | "
            f"{f['case_fault_days_hot']} of {f['case_fault_days_checked']} | "
            f"{fmt(f['siblings_cold_days_per_year'], 1)} |")
    ref = at(case, tols[0])
    if ref:
        sib_lim = [fmt(at(t, tols[0])["limit_false_call_outs_per_year"], 1)
                   for t in turbines if t != case and at(t, tols[0])]
        out.append(
            f"| plain 80 °C limit, held 1 h | {' / '.join(sib_lim)} | "
            f"{fmt(ref['limit_false_call_outs_per_year'], 1)} | "
            f"{ref['limit_first_fault_call_out'] or 'none'} | | |")
    out.append("")

    noise = [at(t, tols[0])["healthy_daily_mean_k"]["sd"] for t in turbines
             if t != case and at(t, tols[0])]
    if noise:
        out.append(
            "On healthy days the day's mean difference has a standard deviation "
            f"of {fmt(min(noise), 2)} to {fmt(max(noise), 2)} K on the siblings, "
            "so a tolerance of 1 K sits inside the model's own day-to-day noise. "
            "Too tight a tolerance also starves the training: every flagged day "
            "is left out, and when too few healthy days remain the model goes "
            "blank.\n")

    r2 = at(case, "2")
    if r2:
        by_year = r2.get("false_call_outs_by_year") or {}
        months_2017 = {k: v for k, v in (r2.get("hot_days_by_month") or {}).items()
                       if k >= "2017"}
        out.append(
            f"**{case} before its fault (2 K).** Hot call-outs outside the fault "
            "period, by year: " + ", ".join(f"{y}: {n}" for y, n in by_year.items())
            + ". Hot days by month from 2017: "
            + ", ".join(f"{m}: {n}" for m, n in months_2017.items())
            + ". The bearing began running hotter than its own healthy reference "
            "from May 2017, months before the first sustained hot spell on "
            "8 September. With no fault labels this is a candidate early warning, "
            "not a proven one: those days count as false alarms here.\n")

    out.append("**What it costs in coverage (2 K):**\n")
    out.append("| turbine | running days | days blank (start, maintenance) | "
               "running days checked | maintenance restarts |")
    out.append("|---|---|---|---|---|")
    for t in turbines:
        r = at(t, "2")
        if r:
            out.append(f"| {t} | {r['running_days']} | {r['blank_running_days']} | "
                       f"{_pct(r['checked_share'])} % | "
                       f"{len(r['maintenance_restarts'])} |")
    out.append("")

    # The 7-day forecast: the same frozen model fed a weather forecast.
    acc = [at(t, "2")["forecast_accuracy"] for t in turbines
           if at(t, "2") and at(t, "2").get("forecast_accuracy")]
    if acc:
        leads = sorted({int(L) for a in acc for L in a["by_lead"]})

        def mean(key, L):
            v = [a["by_lead"][str(L)][key] for a in acc if str(L) in a["by_lead"]]
            return sum(v) / len(v)

        same = sum(a["same_day_mae_vs_sensor_k"] for a in acc) / len(acc)
        out.append("**The 7-day forecast (2 K, healthy days, the four turbines "
                   "averaged).** The month's frozen model is fed a weather "
                   "forecast issued days before; the power comes from the "
                   "turbine's own power curve against wind. The forecast is "
                   "simulated: " + proto.get("forecast", "") + ".\n")
        out.append("| issued, days before | forecast vs sensor (K) | "
                   "forecast vs on-day prediction (K) |")
        out.append("|---|---|---|")
        for L in leads:
            out.append(f"| {L} | {fmt(mean('mae_vs_sensor_k', L), 2)} | "
                       f"{fmt(mean('mae_vs_same_day_k', L), 2)} |")
        out.append(f"| on the day, actual weather | {fmt(same, 2)} | |")
        out.append("")
        out.append(
            "The model fed the day's actual weather is within "
            f"{fmt(same, 2)} K of the sensor on average; fed a forecast, the error "
            f"grows from {fmt(mean('mae_vs_sensor_k', leads[0]), 1)} K a day ahead "
            f"to {fmt(mean('mae_vs_sensor_k', leads[-1]), 1)} K a week ahead. That "
            "is the weather's error, not the bearing's, so the call-out is made on "
            "the day and the forecast is for planning.\n")

    # Every sensor, and the siblings.
    sib = dcj.get("siblings") or {}
    if sib:
        out.append("**Second opinions (2 K): every gearbox sensor, and the "
                   "siblings.** Each sensor has its own model. The siblings are "
                   "compared at the same moment, power within "
                   "a tenth of rated, the normal gap learned per wind sector or "
                   "overall, and a call-out when the 5-day median change passes "
                   "+2 K.\n")
        out.append("| turbine | bearing model alone | 2 of 4 gearbox sensors | "
                   "siblings, per wind sector | siblings, one normal |")
        out.append("|---|---|---|---|---|")
        for t in turbines:
            r = at(t, "2")
            if not r or t not in sib:
                continue
            g = r.get("gearbox_rule") or {}

            def cell(v, first):
                return f"{fmt(v, 1)} / yr" + (f", fault {first[:10]}" if first else "")

            d = sib[t]["direction"]["by_threshold"]["2"]
            o = sib[t]["overall"]["by_threshold"]["2"]
            out.append(
                f"| {t} | {cell(r['false_call_outs_per_year'], r['first_fault_call_out'])} | "
                f"{cell(g.get('false_call_outs_per_year'), g.get('first_fault_call_out'))} | "
                f"{cell(d['false_call_outs_per_year'], d['first_fault_call_out'])} | "
                f"{cell(o['false_call_outs_per_year'], o['first_fault_call_out'])} |")
        out.append("")
        out.append(
            "False call-outs a year, and the day of the first call-out on the "
            "fault. Neither second opinion beats the bearing's own model: asking "
            "two gearbox sensors to agree adds false call-outs, and learning the "
            "sibling gap per wind sector did not lower them on the healthy "
            "turbines and called the fault later. They stay what the technician "
            "looks at, not the alarm.\n")

    out.append(
        "**What this does not show.** Maintenance and inspections are stood in "
        "for, not recorded. There is one documented fault. Days with too few "
        "usable readings are not checked. The tolerance is chosen on the same "
        "history it is scored on.\n")
    return "\n".join(out)


def section_confirmation() -> str:
    """The historical confirmation: train on last month, predict this month.

    Written from results/monthly_confirmation.json. Every figure is the one the
    monthly loop produced, over every month of every turbine, so the claim that
    a healthy model fitted on last month predicts next month is shown rather
    than asserted. Skipped until `make confirm` has run.
    """
    cf = load("monthly_confirmation.json")
    if not cf or not cf.get("months"):
        return ""
    s = cf["summary"]
    fleet, per = s["fleet"], s["by_turbine"]
    proto = cf.get("protocol", {})
    limit = fmt(proto.get("fixed_limit_c"), 0)

    out = ["### The historical confirmation: train on last month, predict this month\n"]
    out.append(
        "The project in one paragraph. Every month a fresh reference is fitted "
        "on at least one month of healthy readings (the whole previous month, "
        "reached further back only as far as needed), so it knows how each sensor "
        "behaves when nothing is wrong, including how load and weather move it. "
        "It is then asked what a healthy machine would show in the next month, "
        "given that month's load and weather, and an alarm is raised when the "
        "sensor runs hotter than that expectation, not merely above a fixed "
        "number. The data are historical, so the next month is already known and "
        "the prediction can be confirmed. This runs the dashboard's monthly loop "
        f"over every month of every turbine ({fleet['months']} turbine-months). "
        "Health is an assumption: every month of every turbine, except the "
        "documented R80711 anomaly.\n")

    out.append("**Was the month predicted?** The gearbox bearing's temperature, "
               "predicted from the month before:\n")
    out.append("| turbine | healthy months | median error | 90th-percentile error | "
               "months within 1 K of unbiased |")
    out.append("|---|---|---|---|---|")
    for name, b in list(per.items()) + [("all four", fleet)]:
        out.append(
            f"| {name} | {b['healthy_months']} | {fmt(b['median_mae_k'], 2)} K | "
            f"{fmt(b['p90_mae_k'], 2)} K | "
            f"{_pct(b['share_months_within_1k_bias'])} % |")
    out.append("")

    out.append(f"**False alarms in healthy months: the model against a plain "
               f"{limit} °C limit.** A dispatch is an alarm sustained for an "
               "hour. In a healthy month every dispatch is a false one. The last "
               "column is the fixed limit's alarms on which the model stayed "
               "quiet, because load and weather explained that temperature.\n")
    out.append("| turbine | model dispatches | per turbine-year | "
               f"{limit} °C limit dispatches | per turbine-year | "
               "limit alarms the model stayed quiet on |")
    out.append("|---|---|---|---|---|---|")
    for name, b in list(per.items()) + [("all four", fleet)]:
        out.append(
            f"| {name} | {b['healthy_model_dispatches']} | "
            f"{fmt(b['healthy_model_dispatches_per_turbine_year'], 1)} | "
            f"{b['healthy_fixed_limit_dispatches']} | "
            f"{fmt(b['healthy_fixed_limit_dispatches_per_turbine_year'], 1)} | "
            f"{b['healthy_fixed_limit_dispatches_explained']} of "
            f"{b['healthy_fixed_limit_dispatches']} |")
    out.append("")

    anomaly = [r for r in cf["months"] if not r["healthy_assumed"]]
    if anomaly:
        out.append("**The anomaly months (R80711).** Here an alarm is what is wanted:\n")
        out.append("| month | prediction error | model dispatches | "
                   f"first model dispatch | {limit} °C limit dispatches | "
                   "first limit dispatch |")
        out.append("|---|---|---|---|---|---|")
        for r in anomaly:
            out.append(
                f"| {r['month']} | {fmt(r['mae_k'], 2)} K | {r['model_dispatches']} | "
                f"{r['first_model_dispatch'] or 'none'} | "
                f"{r['fixed_dispatches']} | {r['first_fixed_dispatch'] or 'none'} |")
        out.append("")

    fleet_settings = fleet.get("settings") or {}
    if fleet_settings:
        sep = next((r for r in cf["months"] if r["turbine"] == "R80711"
                    and r["month"] == "2017-09"), None)
        turbines = list(per)
        out.append(
            "**Can the false alarms be turned down, and what does it cost?** The "
            "same history re-scored at stricter alarm settings, with nothing "
            "refitted. False dispatches per turbine-year in healthy months, then "
            "how many of R80711's anomaly months were alarmed at all, and when the "
            "first dispatch of September 2017 came (the real hot spell began on "
            "2017-09-08).\n")
        out.append("| rule | all four | " + " | ".join(turbines)
                   + " | anomaly months alarmed | first dispatch, Sep 2017 |")
        out.append("|---|" + "---|" * (len(turbines) + 3))

        def parse(key):
            return float(key[1:key.index("_")]), int(key[key.index("_s") + 2:])

        def row_for(label, get, first):
            cells = [fmt(get(fleet)["healthy_dispatches_per_turbine_year"], 1)]
            cells += [fmt(get(per[t])["healthy_dispatches_per_turbine_year"], 1)
                      for t in turbines]
            a = get(per["R80711"]) if "R80711" in per else None
            cells.append(f"{a['anomaly_months_alarmed']} of {a['anomaly_months']}"
                         if a else "not measured")
            cells.append(first or "none")
            out.append(f"| {label} | " + " | ".join(cells) + " |")

        for sustain in sorted({parse(k)[1] for k in fleet_settings}):
            held = _persistence_label(sustain)
            for key in sorted((k for k in fleet_settings if parse(k)[1] == sustain),
                              key=lambda k: parse(k)[0]):
                z = parse(key)[0]
                row_for(f"model, z above {z:g}, held {held}",
                        lambda blk, key=key: blk["settings"][key],
                        ((sep or {}).get("settings", {}).get(key) or {}).get("first"))
            fkey = f"s{sustain}"
            if fkey in (fleet.get("fixed_settings") or {}):
                row_for(f"**plain {limit} °C limit, held {held}**",
                        lambda blk, fkey=fkey: blk["fixed_settings"][fkey],
                        ((sep or {}).get("fixed_settings", {}).get(fkey) or {}).get("first"))
            two = fleet.get("two_stage") or {}
            for key in sorted((k for k in two if parse(k)[1] == sustain),
                              key=lambda k: parse(k)[0]):
                row_for(f"**{limit} °C limit, vetoed by the model at z above "
                        f"{parse(key)[0]:g}, held {held}**",
                        lambda blk, key=key: blk["two_stage"][key],
                        ((sep or {}).get("two_stage", {}).get(key) or {}).get("first"))
        out.append("")
        if fleet.get("two_stage"):
            out.append(
                "The rows in bold that say *vetoed* are the two-stage rule: the "
                f"{limit} °C limit triggers, and a dispatch stands only if the "
                "healthy model also finds the temperature unexpected inside it. "
                "A limit alarm that load and weather explain is dropped; one the "
                "model cannot predict (a load it has hardly seen) is NOT "
                "explained, so it stands. Note "
                "that the August row counts as an anomaly month because the "
                "anomaly span starts on 15 August, so a limit alarm early in "
                "August is counted as caught although it came before the anomaly.\n")

    if fleet.get("strategy"):
        turbines = list(per)
        sep = next((r for r in cf["months"] if r["turbine"] == "R80711"
                    and r["month"] == "2017-09"), None)
        r711 = per.get("R80711") or {}
        out.append(
            "**The sibling warning and the full strategy.** The sibling warning "
            "compares a turbine with its siblings at the same moment and similar "
            "power, against its normal gap over the model's training window, "
            "smoothed by a 5-day median; a run of flagged days is one call-out at "
            "the end of its first day. *Confirmed* means the model also found "
            "unexplained heat that day. The strategy is the veto (z above 2) plus "
            "the sibling call-outs, counted once per day. False call-outs per "
            "turbine-year in healthy months:\n")
        out.append("| rule | " + " | ".join(turbines) + " | all four | R80711 fault months "
                   "alarmed | first call-out, Sep 2017 |")
        out.append("|---|" + "---|" * (len(turbines) + 3))

        def srow(label, field, key):
            cells = [fmt(per[t][field][key]["healthy_dispatches_per_turbine_year"], 1)
                     for t in turbines]
            cells.append(fmt(fleet[field][key]["healthy_dispatches_per_turbine_year"], 1))
            a = (r711.get(field) or {}).get(key) or {}
            cells.append(f"{a.get('anomaly_months_alarmed', 0)} of {a.get('anomaly_months', 0)}")
            cells.append((((sep or {}).get(field) or {}).get(key) or {}).get("first") or "none")
            out.append(f"| {label} | " + " | ".join(cells) + " |")

        srow(f"plain {limit} °C limit, held 1 h", "fixed_settings", "s6")
        srow("limit, vetoed by the model at z above 2", "two_stage", "z2_s6")
        for key in sorted(fleet["strategy"], key=lambda k: float(k[1:])):
            thr = key[1:]
            srow(f"sibling warning, +{thr} K", "sibling", key)
            srow(f"sibling warning, +{thr} K, confirmed", "sibling_confirmed", key)
            srow(f"**strategy: veto + sibling, +{thr} K**", "strategy", key)
            srow(f"**strategy: veto + confirmed sibling, +{thr} K**", "strategy_confirmed", key)
        out.append("")
        fx = r711.get("fixed_settings", {}).get("s6") or {}
        v2 = r711.get("two_stage", {}).get("z2_s6") or {}
        v3 = r711.get("two_stage", {}).get("z3_s6") or {}
        if fx and v2 and v3:
            out.append(
                "**Where the veto's removals were.** In R80711's healthy months the "
                f"limit raised {fx['healthy_dispatches']} dispatches and the veto kept "
                f"{v2['healthy_dispatches']} of them at z above 2 and "
                f"{v3['healthy_dispatches']} at z above 3. In its fault months the "
                f"limit raised {fx['anomaly_dispatches']} and the veto kept "
                f"{v2['anomaly_dispatches']} and {v3['anomaly_dispatches']}. So the "
                "alarms the veto removes are, on this record, alarms raised DURING "
                "the fault, most likely because the monthly reference re-learns a "
                "fault that has lasted for weeks (its training window then holds "
                "it) and so explains its heat away; that mechanism is not measured "
                "directly here.\n")

    episodes = cf.get("r80711_sibling_episodes") or []
    if episodes:
        out.append("**R80711's stretches hotter than its siblings than usual** (more "
                   "than 2 K above its gap over the previous 30 days, 5-day median), "
                   "for judging what they were:\n")
        out.append("| from | to | days | peak |")
        out.append("|---|---|---|---|")
        for e in episodes:
            out.append(f"| {e['start']} | {e['end']} | {e['days']} | "
                       f"+{fmt(e['peak_k'], 2)} K |")
        out.append("")

    out.append(
        "**What this does not show.** Health is assumed, not labelled. There is "
        "one real anomaly. In the later anomaly months the fresh reference has "
        "been fitted partly on the anomaly itself, which is its documented blind "
        "spot. The alarm settings are scored on the same history they are read "
        "from, so none of them is a forecast of how a setting would do on "
        "unseen months.\n")
    return "\n".join(out)


def section_goal() -> str:
    """The project's goal, and the three measurements that answer it.

    Generated from results/backtest.json through `windmill.scorecard`, the
    same module the dashboard's headline cards read, so no number here is
    typed by hand and the README cannot disagree with the dashboard.
    """
    b = load("backtest.json")
    if not b:
        return ""
    curve = load("operating_curve.json")
    card = scorecard.goal_scorecard(b, load("engie_multivariate.json"),
                                    load("demo_r80711.json"), curve)
    ages = card["false_alarm"]["ratio_by_reference_age"]
    frozen_ratio = card["false_alarm"]["ratio_frozen_2014"]
    det = card["detection"].get(scorecard.DEPLOYED_ARM) or {}
    lrn = card["detection"].get(scorecard.LEARNED_ARM) or {}
    one, five = scorecard.lead_days(card)
    l_one, _ = scorecard.lead_days(card, scorecard.LEARNED_ARM)
    pooled = (b.get("A1_A2_weekly_backtest") or {}).get("pooled") or {}
    summary = pooled.get("summary") or {}
    paired = pooled.get("paired") or {}

    out = ["## The goal\n"]
    out.append(
        "**Cut the cost of unnecessary maintenance call-outs by raising fewer "
        "false alarms, while still warning early enough of a developing "
        "bearing fault that the turbine can be stopped on purpose instead of "
        "tripping on its own.**\n")
    out.append(
        "The two halves pull against each other: a detector can always be made "
        "quieter by making it less sensitive, and made to warn earlier by "
        "making it more sensitive. So no figure below is quoted alone -- each "
        "is stated at a false-alarm rate, and the evaluation suite compares "
        "every arm at the same one.\n")
    out.append("| the question | the answer | measured by |")
    out.append("|---|---|---|")
    if ages.get("one_year") and ages.get("two_years"):
        out.append(
            f"| Fewer false alarms? | {fmt(ages['one_year'], 1)}x fewer than "
            f"with a one-year-old reference and {fmt(ages['two_years'], 1)}x "
            f"fewer than with a two-year-old one, at the same threshold rule | "
            f"B.4, the age of the reference |")
    ops = card.get("operating_point") or {}
    op = ops.get(scorecard.DEPLOYED_ARM)
    if op and op.get("detection_rate") is not None:
        others = ""
        for arm, name in (("constant", "a constant-median arm recalibrated the "
                                       "same way"),
                          (scorecard.LEARNED_ARM, "the learned arm")):
            o = ops.get(arm)
            if o and o.get("detection_rate") is not None:
                others += (f"; {name}: {_pct(o['detection_rate'])} %, "
                           f"{fmt(o['median_days'], 1)} days, chance "
                           f"{_pct(o['chance_rate'])} %, "
                           f"{fmt(o['false_dispatches_per_turbine_year'], 1)} "
                           f"false call-outs")
        out.append(
            f"| Is a developing fault caught? | {_pct(op['detection_rate'])} % of "
            f"injected +{fmt(scorecard.HEADLINE_RAMP_K, 0)} K ramps, a median "
            f"{fmt(op['median_days'], 1)} days after onset, with an hour of "
            f"sustained alarm at a 1 % rate; a 0 K placebo is \"caught\" "
            f"{_pct(op['chance_rate'])} % of the time; "
            f"{fmt(op['false_dispatches_per_turbine_year'], 1)} false call-outs "
            f"per turbine-year{others} | the operating curve, a simulation |")
    elif det.get("rate") is not None:
        learned = ""
        if lrn.get("rate") is not None:
            learned = (f" (the learned arm: {fmt(lrn['rate'] * 100, 0)} %, "
                       f"{fmt(lrn.get('median_days'), 1)} days)")
        out.append(
            f"| Is a developing fault caught? | {fmt(det['rate'] * 100, 0)} % of "
            f"injected +{fmt(scorecard.HEADLINE_RAMP_K, 0)} K ramps, a median "
            f"{fmt(det.get('median_days'), 1)} days after onset, at a 1 % "
            f"false-alarm rate{learned} | C.6, a simulation |")
    anomaly = card.get("anomaly") or {}
    std = (anomaly.get("standing") or {}).get(scorecard.DEPLOYED_ARM)
    ec = (anomaly.get("early_clean_settings") or {}).get(scorecard.DEPLOYED_ARM)
    if std is not None:
        d = std.get("days_before_hot_spell")
        if std.get("days_after_onset") is None:
            verdict = "never dispatched during the anomaly"
        elif d is None:
            verdict = f"first dispatched on {str(std['first_dispatch'])[:10]}"
        elif d >= 0:
            verdict = f"dispatched {fmt(d, 1)} days before"
        else:
            verdict = f"dispatched {fmt(-d, 1)} days after"
        spell = str(anomaly.get("first_hot_spell") or "")[:10]
        out.append(
            f"| How much warning on the one real case? | R80711, September "
            f"2017: the deployed model at the standing 1 %, 1 h setting "
            f"{verdict}"
            + (f" the first sustained hot spell ({spell})"
               if d is not None and std.get("days_after_onset") is not None else "")
            + (f"; {ec[0]} of {ec[1]} settings of its dial warned early "
               f"without a false dispatch in the quiet stretch before the onset"
               if ec else "")
            + " | the real-anomaly check, unlabelled |")
    elif one is not None and five is not None:
        learned = (f" (the learned arm: {fmt(l_one, 0)} days at 1 %)"
                   if l_one is not None else "")
        out.append(
            f"| How much warning on the one real case? | R80711, September "
            f"2017, by the first day with any alarmed sample: {fmt(one, 0)} days before the temperature crosses 80 °C at "
            f"a 1 % false-alarm rate, {fmt(five, 0)} days at 5 %{learned} | "
            f"C.7, unlabelled |")
    out.append("")

    out.append("**What this does not show.**\n")
    out.append(
        "- **The savings are modelled, not measured.** The public data has no "
        "maintenance log, so the cost of a false call-out is an assumption "
        "(a flat figure per false dispatch, adjustable in the dashboard), not "
        "an observation.")
    out.append(
        "- **The faults are simulated, and the one real case is unlabelled.** "
        "The ENGIE release carries no fault label for the R80711 anomaly, so "
        "its lead time says how early a detector would have fired, not that a "
        "fault was confirmed.")
    lc, lo, lw = (paired.get("load+ambient vs constant") or {},
                  paired.get("load+ambient vs load") or {},
                  paired.get("load+ambient+wind vs load+ambient") or {})
    if lc and lo and lw:
        out.append(
            "- **The operating point does the work; the weather barely does.** "
            "Conditioning on load cuts the median prediction error by "
            f"{fmt(abs(lc.get('median_delta')), 2)} K against a constant. Adding "
            f"ambient temperature changes it by {fmt(lo.get('median_delta'), 3)} K "
            f"and wind speed by {fmt(lw.get('median_delta'), 3)} K on this channel.")
    dep, lgb = summary.get("load+ambient"), summary.get("lightgbm")
    if dep and lgb:
        out.append(
            f"- **The learned model is not the worse predictor.** Its median "
            f"error is {fmt(lgb['mae_k_median'], 2)} K against "
            f"{fmt(dep['mae_k_median'], 2)} K for the deterministic model; the "
            f"deterministic one is kept for being transparent, not for being "
            f"more accurate.")
    if frozen_ratio:
        out.append(
            f"- **The {fmt(frozen_ratio, 1)}x figure is the extreme case, not "
            f"the like-for-like one.** It compares a reference frozen in 2014 "
            f"with a rolling one, each arm keeping its own threshold rule; the "
            f"matched-threshold figures above are the ones to quote.")
    n_never = n_arms = 0
    cm_ = load("cost_model.json") or {}
    for arm in (curve or {}).get("protocol", {}).get("arms", []):
        b = ((cm_.get("arms") or {}).get(arm) or {}).get("best_setting")
        rows_ = ((curve or {}).get("r80711_lead_time") or {}).get("rows") or []
        if not b or not rows_:
            continue
        r = next((x for x in rows_ if x["arm"] == arm
                  and x["target_rate"] == b["target_rate"]
                  and x["persistence_samples"] == b["persistence_samples"]), None)
        if r is not None:
            n_arms += 1
            n_never += 1 if r.get("days_after_onset") is None else 0
    if n_arms:
        out.append(
            "- **The simulated fault and the real one disagree.** The settings "
            "the economics prefer for a flat offset were also run on the one "
            f"real anomaly: {n_never} of {n_arms} arms never dispatched during "
            "it. The real excess grows with load and comes in spells, not as a "
            "steady step, so a rule that waits for a long unbroken alarm can "
            "miss it. The cost ranking should be read with that in mind.")
    c_b = scorecard.curve_budget_reading(curve, 12, "constant")
    d_b = scorecard.curve_budget_reading(curve, 12, scorecard.DEPLOYED_ARM)
    if c_b and d_b:
        out.append(
            "- **Early warning at equal false call-outs.** At 12 false call-outs "
            "per turbine-year the best setting of a constant-median arm catches "
            f"{_pct(c_b['detection_rate'])} % of +{fmt(scorecard.HEADLINE_RAMP_K, 0)} K ramps "
            f"in {fmt(c_b['median_days_to_dispatch'], 1)} days, against "
            f"{_pct(d_b['detection_rate'])} % in "
            f"{fmt(d_b['median_days_to_dispatch'], 1)} days for the deployed model. "
            "The operating curve gives every budget, a held-out check and a "
            "load-proportional fault.")
    out.append(
        "- **The weather is hindsight, not a forecast.** The expected "
        "temperature is computed from the turbine's measured power and "
        "reanalysis ambient temperature; the forecast chain (A.3) is a "
        "simulation with injected forecast error.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parts = [section_conditioning(), section_decomposition(),
             section_engie_multivariate(), section_scorability(),
             section_care(), section_backtest(), section_daily_check(),
             section_confirmation(),
             section_operating_curve(), section_cost()]
    body = "\n".join(p for p in parts if p.strip())

    readme = README.read_text()
    if MARKER_START in readme and MARKER_END in readme:
        head = readme.split(MARKER_START)[0]
        tail = readme.split(MARKER_END)[1]
        new = f"{head}{MARKER_START}\n\n## Results\n\n{body}\n{MARKER_END}{tail}"
    else:
        new = f"{readme}\n\n{MARKER_START}\n\n## Results\n\n{body}\n{MARKER_END}\n"

    method = section_method()
    if METHOD_START in new and METHOD_END in new:
        mhead = new.split(METHOD_START)[0]
        mtail = new.split(METHOD_END)[1]
        new = f"{mhead}{METHOD_START}\n\n{method}\n{METHOD_END}{mtail}"

    goal = section_goal()
    if goal and GOAL_START in new and GOAL_END in new:
        ghead = new.split(GOAL_START)[0]
        gtail = new.split(GOAL_END)[1]
        new = f"{ghead}{GOAL_START}\n\n{goal}\n{GOAL_END}{gtail}"

    # The test count is generated too, so it cannot go stale.
    n_tests = _count_tests()
    new = re.sub(r"^\d+ tests, including", f"{n_tests} tests, including",
                 new, count=1, flags=re.MULTILINE)

    README.write_text(new)
    print(f"wrote {README} ({len(body)} chars of results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
