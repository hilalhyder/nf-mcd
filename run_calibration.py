"""
Experiment 2 (reviewer request): confidence calibration analysis for NF-MCD's
cross-modal agreement/confidence layer as a mismatch detector -- reliability
diagrams, Expected Calibration Error (ECE), Brier score, precision/recall for
identifying unreliable content (not just AUROC), and calibration before vs.
after missingness.

Reuse, not reimplementation
----------------------------------------------------------------------------
The manuscript's Section 6.4 mismatch-detection AUROC numbers come from
run_mechanism.py: `build_A` (synthetic graphs with KNOWN misaligned nodes,
via nf_mcd.datasets.generate_synthetic_multimodal_graph's misalignment_rate),
`build_B` (real cached embeddings -- crisismmd / fakeddit -- with an injected
image swap for a `q`-fraction of paired nodes, donor drawn from a different
community), `fit_kind` (fits NFMCD with a named config) and `_auc_ap`
(AUROC/AP against the injected mismatch label). This script imports and
reuses `build_B` and `fit_kind` from run_mechanism.py directly (real-data
part) and calls nf_mcd.datasets.generate_synthetic_multimodal_graph directly
for the synthetic part (build_A's synthetic construction is just that call
with misalignment_rate>0 -- reusing the generator itself, not reimplementing
it -- but build_A's own setting-string parser hardcodes
missing_modality_rate=0.1, so this script calls the generator directly to
get an independently controllable missingness axis, which is this
experiment's whole point). For missingness under real data, this script
reuses `apply_mask` from run_missing.py (identical MCAR mask machinery used
throughout this project) on top of build_B's swapped embeddings.

Convention (stated once, used throughout): NFMCD's fitted `confidence_` is
treated as the model's estimated probability that a node's content pair is
NOT mismatched. Equivalently, `p_mismatch = 1 - confidence` is the predicted
probability of the positive class y=1 ("this pair was injected as
mismatched"), which is what calibration (ECE, Brier, reliability diagram)
and precision/recall are computed against. AUROC/AP are computed on the same
p_mismatch score (equivalent to run_mechanism's `-agreement`/`-confidence`
ranking, just oriented as a probability here for the calibration metrics to
share one convention).

Every metric is computed only on nodes NFMCD actually placed in the shared
(CCA) cross-modal space (`modality_flags_ == "both"` and a defined
agreement), exactly as run_mechanism.det_row does -- a node with only one
modality, or none, has no cross-modal claim to calibrate.

Crash-safe: every (source, regime, param, missingness, seed, kind) row is
appended to experiments/calibration_results.csv with flush+fsync; reruns
skip finished rows. Other files (summary log, progress, reliability-diagram
plot) are written temp+rename.

Usage (from nfmcd_impl/):
    py run_calibration.py run [--workers N]   # resumable
    py run_calibration.py summary             # rebuild summary log + plots
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "calibration_results.csv")
SUMMARY_LOG = os.path.join(EXP, "calibration_summary.log")
PROGRESS_MD = os.path.join(EXP, "calibration_progress.md")

KINDS = ("default", "rank16")

SYN_N, SYN_K = 600, 6
SYN_CONTENT = {"hi": 3.0, "lo": 0.5}
SYN_MIS = (0.1, 0.2, 0.3, 0.4)
SYN_MISSING = (0.0, 0.4)
SYN_SEEDS = (0, 1, 2, 3, 4)

REAL_DATASETS = ("crisismmd", "fakeddit")
REAL_Q = (0.1, 0.2, 0.3, 0.4)
REAL_MISSING = (0.0, 0.4)
REAL_SEEDS = (0, 1, 2)

N_BINS = 10
FIELDS = ["source", "regime", "param", "missingness", "seed", "kind",
          "n_both", "n_bad", "prevalence", "auroc", "ap", "brier", "ece",
          "prec_t03", "rec_t03", "prec_t05", "rec_t05", "prec_t07", "rec_t07",
          "bins_json", "error", "secs"]


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Confidence-calibration analysis progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_calibration.py run --workers 4\n"
        "py run_calibration.py summary\n"
        "```\n\n"
        "Results: experiments/calibration_results.csv (append-only, fsynced per job). "
        "Summary: experiments/calibration_summary.log.\n"
    ))


def rkey(r):
    return (r["source"], r["regime"], f"{float(r['param']):.3f}", f"{float(r['missingness']):.2f}",
            int(r["seed"]), r["kind"])


def load_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header != FIELDS:
            return []
        return [dict(zip(FIELDS, r)) for r in rd if len(r) == len(FIELDS)]


def open_append():
    new = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not new:
        with open(RESULTS_CSV, "rb") as f:
            data = f.read()
        if not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            with open(RESULTS_CSV, "r+b") as f:
                f.truncate(cut)
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    return f, w


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------

def build_synth(content_level, mis_rate, missing_rate, seed):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph as gen

    scale = SYN_CONTENT[content_level]
    s = gen(n_nodes=SYN_N, n_communities=SYN_K, p_in=0.18, p_out=0.02,
            missing_modality_rate=missing_rate, misalignment_rate=mis_rate,
            overlap_rate=0.0, centroid_scale=scale, noise_scale=1.0, seed=seed)
    d = dict(G=s.G, e_t=list(s.text_embeddings), e_v=list(s.image_embeddings),
              true=s.true_communities, n_communities=SYN_K)
    bad = np.zeros(SYN_N, dtype=bool)
    if s.misaligned_nodes:
        bad[list(s.misaligned_nodes)] = True
    return d, bad


def build_real(ds, q, missing_rate, seed):
    from run_mechanism import build_B

    d, primary, bad = build_B(f"B|{ds}|q{q}", seed)
    if missing_rate > 0:
        from run_missing import apply_mask
        d = apply_mask(d, "both", missing_rate, seed)
    return d, bad


# ---------------------------------------------------------------------------
# calibration metrics
# ---------------------------------------------------------------------------

def calc_calibration(agreement, confidence, flags, bad, n_bins=N_BINS):
    """`confidence` is treated as P(pair is NOT mismatched); p_mismatch = 1 -
    confidence is the predicted probability of the positive class y=1
    (injected mismatch), which every metric below is computed against."""
    flags = np.asarray(flags)
    agreement = np.asarray(agreement, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bad = np.asarray(bad, dtype=bool)
    both = (flags == "both") & ~np.isnan(agreement)
    y = bad[both].astype(int)
    n = int(len(y))

    out = dict(n_both=n, n_bad=int(y.sum()) if n else 0,
               prevalence=float(y.mean()) if n else float("nan"),
               auroc=float("nan"), ap=float("nan"), brier=float("nan"), ece=float("nan"),
               pr={0.3: (float("nan"), float("nan")), 0.5: (float("nan"), float("nan")),
                   0.7: (float("nan"), float("nan"))},
               bins=[])
    if n == 0 or y.sum() == 0 or y.sum() == n:
        return out

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    p_mis = 1.0 - confidence[both]
    out["auroc"] = float(roc_auc_score(y, p_mis))
    out["ap"] = float(average_precision_score(y, p_mis))
    out["brier"] = float(brier_score_loss(y, p_mis))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx_bin = np.clip(np.digitize(p_mis, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        m = idx_bin == b
        cnt = int(m.sum())
        if cnt == 0:
            bins.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=0, conf=None, acc=None))
            continue
        conf_b = float(p_mis[m].mean())
        acc_b = float(y[m].mean())
        ece += (cnt / n) * abs(acc_b - conf_b)
        bins.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=cnt, conf=conf_b, acc=acc_b))
    out["ece"] = float(ece)
    out["bins"] = bins

    pr = {}
    for thr in (0.3, 0.5, 0.7):
        pred = p_mis >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        pr[thr] = (prec, rec)
    out["pr"] = pr
    return out


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def all_jobs():
    jobs = []
    for content_level in SYN_CONTENT:
        for mis in SYN_MIS:
            for missing in SYN_MISSING:
                for seed in SYN_SEEDS:
                    for kind in KINDS:
                        jobs.append(("synthetic", content_level, mis, missing, seed, kind))
    for ds in REAL_DATASETS:
        for q in REAL_Q:
            for missing in REAL_MISSING:
                for seed in REAL_SEEDS:
                    for kind in KINDS:
                        jobs.append(("real", ds, q, missing, seed, kind))
    return jobs


def run_one(job):
    source, regime, param, missing, seed, kind = job
    t0 = time.monotonic()
    row = dict(source=source, regime=regime, param=f"{param:.3f}", missingness=f"{missing:.2f}",
               seed=seed, kind=kind, n_both="", n_bad="", prevalence=float("nan"),
               auroc=float("nan"), ap=float("nan"), brier=float("nan"), ece=float("nan"),
               prec_t03=float("nan"), rec_t03=float("nan"), prec_t05=float("nan"), rec_t05=float("nan"),
               prec_t07=float("nan"), rec_t07=float("nan"), bins_json="", error="")
    try:
        from run_mechanism import fit_kind

        if source == "synthetic":
            d, bad = build_synth(regime, param, missing, seed)
        else:
            d, bad = build_real(regime, param, missing, seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_kind(d, seed, kind)

        met = calc_calibration(model.agreement_, model.confidence_, model.modality_flags_, bad)
        row.update(n_both=met["n_both"], n_bad=met["n_bad"], prevalence=met["prevalence"],
                   auroc=met["auroc"], ap=met["ap"], brier=met["brier"], ece=met["ece"],
                   prec_t03=met["pr"][0.3][0], rec_t03=met["pr"][0.3][1],
                   prec_t05=met["pr"][0.5][0], rec_t05=met["pr"][0.5][1],
                   prec_t07=met["pr"][0.7][0], rec_t07=met["pr"][0.7][1],
                   bins_json=json.dumps(met["bins"]))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def do_run(workers):
    os.makedirs(EXP, exist_ok=True)
    jobs = all_jobs()
    done = {rkey(r) for r in load_results() if not r.get("error")}

    def jkey(j):
        source, regime, param, missing, seed, kind = j
        return (source, regime, f"{param:.3f}", f"{missing:.2f}", seed, kind)

    pending = [j for j in jobs if jkey(j) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; running {len(pending)} with {workers} workers",
          flush=True)
    write_progress(len(jobs) - len(pending), len(jobs), "starting")
    if not pending:
        write_progress(len(jobs), len(jobs), "all done")
        return
    f, w = open_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    t0 = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, j): j for j in pending}
            for fut in as_completed(futs):
                row = fut.result()
                w.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if row["error"]:
                    n_err += 1
                    if n_err <= 8:
                        print(f"  FAILED {row['source']}/{row['regime']}/{row['param']}/miss={row['missingness']}/"
                              f"{row['kind']} seed={row['seed']}: {row['error']}", flush=True)
                if n_done % 25 == 0 or n_done == len(jobs):
                    write_progress(n_done, len(jobs), f"running ({time.monotonic() - t0:.0f}s elapsed)")
                    print(f"  {n_done}/{len(jobs)} done, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors, {time.monotonic() - t0:.0f}s)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _agg(rows, source, regime, missing, kind, metric, param=None):
    vals = []
    for r in rows:
        if r["source"] != source or r["kind"] != kind:
            continue
        if regime is not None and r["regime"] != regime:
            continue
        if abs(_fnum(r["missingness"]) - missing) > 1e-6:
            continue
        if param is not None and abs(_fnum(r["param"]) - param) > 1e-6:
            continue
        v = _fnum(r[metric])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)


def do_summary():
    rows = [r for r in load_results() if not r.get("error")]
    if not rows:
        print("No results yet.")
        return

    L = []
    L.append("Confidence calibration analysis: NF-MCD confidence as a mismatch detector.")
    L.append("Convention: p_mismatch = 1 - confidence_ (confidence_ = model's estimated P(pair NOT mismatched)).")
    L.append("ECE/Brier/precision/recall computed against the injected mismatch label on 'both'-modality nodes only.")
    L.append(f"Total finished rows: {len(rows)}")
    L.append("=" * 100)

    L.append("\n\n### Synthetic: ECE / Brier / AUROC vs. contamination (misalignment_rate), by content strength "
              "and missingness")
    L.append("(kind=default NF-MCD; mean+-sd over 5 seeds)")
    for content_level in SYN_CONTENT:
        for missing in SYN_MISSING:
            L.append(f"\ncontent={content_level} (centroid_scale={SYN_CONTENT[content_level]}), missingness={missing}")
            L.append(f"  {'mis_rate':<10}{'n_both':>8}{'prevalence':>12}{'ECE':>14}{'Brier':>14}{'AUROC':>14}{'AP':>14}")
            for mis in SYN_MIS:
                n_mu, _, _ = _agg(rows, "synthetic", content_level, missing, "default", "n_both", mis)
                p_mu, _, _ = _agg(rows, "synthetic", content_level, missing, "default", "prevalence", mis)
                ece_mu, ece_sd, n = _agg(rows, "synthetic", content_level, missing, "default", "ece", mis)
                br_mu, br_sd, _ = _agg(rows, "synthetic", content_level, missing, "default", "brier", mis)
                au_mu, au_sd, _ = _agg(rows, "synthetic", content_level, missing, "default", "auroc", mis)
                ap_mu, ap_sd, _ = _agg(rows, "synthetic", content_level, missing, "default", "ap", mis)
                if n == 0:
                    continue
                L.append(f"  {mis:<10.2f}{n_mu:>8.0f}{p_mu:>12.3f}{ece_mu:>7.3f}+-{ece_sd:<5.3f}"
                         f"{br_mu:>7.3f}+-{br_sd:<5.3f}{au_mu:>7.3f}+-{au_sd:<5.3f}{ap_mu:>7.3f}+-{ap_sd:<5.3f}")

    L.append("\n\n### Synthetic: precision/recall for identifying unreliable content, at 3 thresholds on p_mismatch")
    L.append("(kind=default; content=hi; mean over 5 seeds)")
    L.append(f"  {'mis_rate':<10}{'miss':<7}{'prec@0.3':>10}{'rec@0.3':>10}{'prec@0.5':>10}{'rec@0.5':>10}"
             f"{'prec@0.7':>10}{'rec@0.7':>10}")
    for missing in SYN_MISSING:
        for mis in SYN_MIS:
            vals = {}
            ok = True
            for col in ("prec_t03", "rec_t03", "prec_t05", "rec_t05", "prec_t07", "rec_t07"):
                mu, _, n = _agg(rows, "synthetic", "hi", missing, "default", col, mis)
                vals[col] = mu
                ok = ok and n > 0
            if not ok:
                continue
            L.append(f"  {mis:<10.2f}{missing:<7.1f}{vals['prec_t03']:>10.3f}{vals['rec_t03']:>10.3f}"
                     f"{vals['prec_t05']:>10.3f}{vals['rec_t05']:>10.3f}{vals['prec_t07']:>10.3f}{vals['rec_t07']:>10.3f}")

    L.append("\n\n### Synthetic: calibration BEFORE vs AFTER missingness (missingness=0.0 vs 0.4), content=hi, "
              "default kind")
    L.append(f"  {'mis_rate':<10}{'n_both@0.0':>12}{'ECE@0.0':>10}{'n_both@0.4':>12}{'ECE@0.4':>10}"
             f"{'AUROC@0.0':>11}{'AUROC@0.4':>11}")
    for mis in SYN_MIS:
        n0, _, _ = _agg(rows, "synthetic", "hi", 0.0, "default", "n_both", mis)
        e0, _, k0 = _agg(rows, "synthetic", "hi", 0.0, "default", "ece", mis)
        n4, _, _ = _agg(rows, "synthetic", "hi", 0.4, "default", "n_both", mis)
        e4, _, k4 = _agg(rows, "synthetic", "hi", 0.4, "default", "ece", mis)
        a0, _, _ = _agg(rows, "synthetic", "hi", 0.0, "default", "auroc", mis)
        a4, _, _ = _agg(rows, "synthetic", "hi", 0.4, "default", "auroc", mis)
        if k0 == 0 or k4 == 0:
            continue
        L.append(f"  {mis:<10.2f}{n0:>12.0f}{e0:>10.3f}{n4:>12.0f}{e4:>10.3f}{a0:>11.3f}{a4:>11.3f}")

    L.append("\n\n### Real data (crisismmd, fakeddit): ECE / Brier / AUROC vs injected swap fraction q, "
              "missingness 0.0 vs 0.4 (default kind; mean+-sd over 3 seeds)")
    for ds in REAL_DATASETS:
        for missing in REAL_MISSING:
            L.append(f"\n{ds}, missingness={missing}")
            L.append(f"  {'q':<8}{'n_both':>8}{'prevalence':>12}{'ECE':>14}{'Brier':>14}{'AUROC':>14}{'AP':>14}")
            for q in REAL_Q:
                n_mu, _, n = _agg(rows, "real", ds, missing, "default", "n_both", q)
                p_mu, _, _ = _agg(rows, "real", ds, missing, "default", "prevalence", q)
                ece_mu, ece_sd, _ = _agg(rows, "real", ds, missing, "default", "ece", q)
                br_mu, br_sd, _ = _agg(rows, "real", ds, missing, "default", "brier", q)
                au_mu, au_sd, _ = _agg(rows, "real", ds, missing, "default", "auroc", q)
                ap_mu, ap_sd, _ = _agg(rows, "real", ds, missing, "default", "ap", q)
                if n == 0:
                    continue
                L.append(f"  {q:<8.2f}{n_mu:>8.0f}{p_mu:>12.3f}{ece_mu:>7.3f}+-{ece_sd:<5.3f}"
                         f"{br_mu:>7.3f}+-{br_sd:<5.3f}{au_mu:>7.3f}+-{au_sd:<5.3f}{ap_mu:>7.3f}+-{ap_sd:<5.3f}")

    L.append("\n\n### Real data: precision/recall for identifying unreliable content at q=0.2 (default kind)")
    L.append(f"  {'dataset':<14}{'miss':<7}{'prec@0.3':>10}{'rec@0.3':>10}{'prec@0.5':>10}{'rec@0.5':>10}"
             f"{'prec@0.7':>10}{'rec@0.7':>10}")
    for ds in REAL_DATASETS:
        for missing in REAL_MISSING:
            vals = {}
            ok = True
            for col in ("prec_t03", "rec_t03", "prec_t05", "rec_t05", "prec_t07", "rec_t07"):
                mu, _, n = _agg(rows, "real", ds, missing, "default", col, 0.2)
                vals[col] = mu
                ok = ok and n > 0
            if not ok:
                continue
            L.append(f"  {ds:<14}{missing:<7.1f}{vals['prec_t03']:>10.3f}{vals['rec_t03']:>10.3f}"
                     f"{vals['prec_t05']:>10.3f}{vals['rec_t05']:>10.3f}{vals['prec_t07']:>10.3f}{vals['rec_t07']:>10.3f}")

    L.append("\n\n### default vs. rank16 CCA cap: ECE / AUROC at a representative cell "
             "(synthetic content=hi mis=0.2 miss=0.0; real q=0.2 miss=0.0)")
    L.append(f"  {'setting':<28}{'ECE(default)':>14}{'ECE(rank16)':>14}{'AUROC(default)':>16}{'AUROC(rank16)':>16}")
    for source, regime, param in (("synthetic", "hi", 0.2), ("real", "crisismmd", 0.2), ("real", "fakeddit", 0.2)):
        e_d, _, n_d = _agg(rows, source, regime, 0.0, "default", "ece", param)
        e_r, _, n_r = _agg(rows, source, regime, 0.0, "rank16", "ece", param)
        a_d, _, _ = _agg(rows, source, regime, 0.0, "default", "auroc", param)
        a_r, _, _ = _agg(rows, source, regime, 0.0, "rank16", "auroc", param)
        if n_d == 0 or n_r == 0:
            continue
        L.append(f"  {source + '/' + regime:<28}{e_d:>14.3f}{e_r:>14.3f}{a_d:>16.3f}{a_r:>16.3f}")

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)
    make_plots(rows)


def make_plots(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(plotting skipped: {exc})")
        return

    def pooled_bins(source, regime, missing, kind, param):
        pooled = {}
        for r in rows:
            if r["source"] != source or r["regime"] != regime or r["kind"] != kind:
                continue
            if abs(_fnum(r["missingness"]) - missing) > 1e-6 or abs(_fnum(r["param"]) - param) > 1e-6:
                continue
            if not r["bins_json"]:
                continue
            bins = json.loads(r["bins_json"])
            for i, b in enumerate(bins):
                if b["n"] == 0:
                    continue
                acc = pooled.setdefault(i, {"lo": b["lo"], "hi": b["hi"], "conf": [], "acc": [], "n": 0})
                acc["conf"].append(b["conf"])
                acc["acc"].append(b["acc"])
                acc["n"] += b["n"]
        xs, ys, ns = [], [], []
        for i in sorted(pooled):
            b = pooled[i]
            xs.append(float(np.mean(b["conf"])))
            ys.append(float(np.mean(b["acc"])))
            ns.append(b["n"])
        return xs, ys, ns

    panels = [
        ("synthetic", "hi", 0.2, 0.0, "content=hi, mis=0.2, miss=0.0"),
        ("synthetic", "hi", 0.2, 0.4, "content=hi, mis=0.2, miss=0.4"),
        ("real", "crisismmd", 0.2, 0.0, "crisismmd, q=0.2, miss=0.0"),
        ("real", "crisismmd", 0.2, 0.4, "crisismmd, q=0.2, miss=0.4"),
        ("real", "fakeddit", 0.2, 0.0, "fakeddit, q=0.2, miss=0.0"),
        ("real", "fakeddit", 0.2, 0.4, "fakeddit, q=0.2, miss=0.4"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (source, regime, param, missing, title) in zip(axes.ravel(), panels):
        xs, ys, ns = pooled_bins(source, regime, missing, "default", param)
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
        if xs:
            ax.plot(xs, ys, "o-", color="#0B6E6B")
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), fontsize=6, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted P(mismatch)")
        ax.set_ylabel("empirical mismatch rate")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Reliability diagrams: NF-MCD confidence as a mismatch-probability estimator (default kind, "
                 "pooled over seeds)")
    fig.tight_layout()
    out = os.path.join(EXP, "calibration_reliability.png")
    fig.savefig(out + ".tmp.png", dpi=140)
    plt.close(fig)
    os.replace(out + ".tmp.png", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if args.cmd == "run":
        do_run(args.workers)
        do_summary()
    else:
        do_summary()
