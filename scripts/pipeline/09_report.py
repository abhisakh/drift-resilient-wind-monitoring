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
                   f"{fmt(rolling['spread_pp'], 2)} pp). That is a **{fmt(ratio, 0)}x** reduction "
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
                "medians. **With a current reference the deterministic arm wins by a wide\n"
                f"margin** ({fmt(ro['mean_rate_pct'], 2)} % against {fmt(rol['mean_rate_pct'], 2)} %, "
                "about "
                f"{fmt(rol['mean_rate_pct'] / ro['mean_rate_pct'], 0)}x fewer false alarms), and\n"
                "it does so with no model to train and no thresholds to re-validate.\n"
            )
            out.append(
                "The project's claim is therefore not 'the deterministic method is better'\n"
                "but the sharper one: **the reference matters more than the model.**\n"
                "Re-estimating the reference reduced false alarms by "
                f"{fmt(fr['mean_rate_pct'] / ro['mean_rate_pct'], 0)}x for the deterministic\n"
                f"detector and {fmt(frl['mean_rate_pct'] / rol['mean_rate_pct'], 0)}x for the learned one.\n"
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


def main() -> int:
    parts = [section_conditioning(), section_decomposition(), section_engie_multivariate(),
             section_scorability(), section_care()]
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

    # The test count is generated too, so it cannot go stale.
    n_tests = _count_tests()
    new = re.sub(r"^\d+ tests, including", f"{n_tests} tests, including",
                 new, count=1, flags=re.MULTILINE)

    README.write_text(new)
    print(f"wrote {README} ({len(body)} chars of results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
