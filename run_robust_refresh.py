"""
Refresh every comparison that reports the NFMCD.robust() preset, now that it
also sets pca_rank_div=16 (added after clusterer_results.csv / rank_results.csv
/ nbr_explain caches were written -- those files' "robust"/"fcmad|raw" rows
used the OLD preset without the rank cap and are SUPERSEDED by this file for
any robust-preset number; they are not edited).

Recomputes nfmcd_default and nfmcd_robust (via the real NFMCD.robust()
classmethod, not a re-specified kwarg dict) across four groups, reusing
existing loaders/scoring:
  1. orig : the five original datasets + the two real-Fakeddit-graph variants
             (run_realgraph.get_dataset), k = ground truth, seeds 0,1,2.
  2. sweep: homophily sweep crisismmd/fakeddit @ p_out/p_in in RATIOS
             (run_realgraph.get_dataset("<base>@r<ratio>", seed)), seeds 0,1,2.
  3. miss : missing-modality 'both' mode @ p in {0,0.4,0.8} on
             crisismmd/fakeddit_real_lcc (run_missing.apply_mask), seeds 0,1,2
             (mask_seed = method seed, one sample per seed).
  4. mis  : injected cross-modal mismatch @ q in {0,0.1,0.2,0.4} on
             crisismmd/fakeddit/fakeddit_real_lcc (run_mechanism.build_B),
             seeds 0-4, plus detection AUROC of agreement_/confidence_ for
             flagging the corrupted nodes.

spectral_graph+content and kmeans_content are NOT recomputed (unchanged
protocol/settings) -- reference values are pulled straight from the existing
baselines_results.csv / realgraph_results.csv / missing_results.csv /
mech_results.csv.

Usage (from nfmcd_impl/):
    py run_robust_refresh.py run [--workers N]
    py run_robust_refresh.py summary

Crash-safe: each finished row is appended to experiments/robust_refresh_results.csv
and fsynced; a rerun skips rows already present. Other files: temp+rename.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "robust_refresh_results.csv")
SUMMARY_LOG = os.path.join(EXP, "robust_refresh_summary.log")
PROGRESS_MD = os.path.join(EXP, "robust_refresh_progress.md")

ORIG_DS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon", "fakeddit_real_full", "fakeddit_real_lcc"]
NOCONTENT_DS = {"dblp", "amazon"}  # kmeans_content reference is n/a here
SWEEP_BASE = ["crisismmd", "fakeddit"]
RATIOS = (0.02, 0.1, 0.25, 0.5, 1.0)
MISS_DS = ["crisismmd", "fakeddit_real_lcc"]
MISS_PS = (0.0, 0.4, 0.8)
MIS_DS = ["crisismmd", "fakeddit", "fakeddit_real_lcc"]
MIS_QS = (0.0, 0.1, 0.2, 0.4)
SEEDS3 = (0, 1, 2)
SEEDS5 = (0, 1, 2, 3, 4)
METHODS = ["nfmcd_default", "nfmcd_robust"]

FIELDS = ["group", "dataset", "seed", "method", "x", "k_used", "onmi", "modularity", "f1",
          "auroc_agree", "auroc_conf", "error", "secs"]


# ---------------------------------------------------------------------------
# crash-safe file helpers (same pattern as the other runners)
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
        "# Robust-preset refresh progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_robust_refresh.py run --workers 4\n"
        "py run_robust_refresh.py summary\n"
        "```\n\n"
        "Results: experiments/robust_refresh_results.csv (append-only, fsynced per row). "
        "Summary: experiments/robust_refresh_summary.log.\n"
        "This supersedes robust/fcmad|raw numbers in clusterer_results.csv and "
        "rank_results.csv, computed before pca_rank_div=16 was added to NFMCD.robust().\n"
    ))


def rkey(row):
    return (row["group"], row["dataset"], int(row["seed"]), row["method"], row["x"])


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "group":
                continue
            row = dict(zip(FIELDS, r))
            if row["error"]:
                continue
            try:
                for k in ("onmi", "modularity", "f1", "auroc_agree", "auroc_conf"):
                    row[k] = float(row[k]) if row[k] not in ("", "nan") else float("nan")
                row["seed"] = int(row["seed"])
            except ValueError:
                continue
            res[rkey(row)] = row
    return res


def open_for_append():
    new = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not new:
        with open(RESULTS_CSV, "rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_nl = f.read(1) == b"\n"
        if not ends_nl:
            with open(RESULTS_CSV, "ab") as f:
                f.write(b"\n")
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(FIELDS)
        f.flush()
        os.fsync(f.fileno())
    return f, w


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def all_jobs():
    jobs = []
    for ds in ORIG_DS:
        for m in METHODS:
            for s in SEEDS3:
                jobs.append(("orig", ds, s, m, ""))
    for base in SWEEP_BASE:
        for r in RATIOS:
            for m in METHODS:
                for s in SEEDS3:
                    jobs.append(("sweep", base, s, m, f"{r}"))
    for ds in MISS_DS:
        for p in MISS_PS:
            for m in METHODS:
                for s in SEEDS3:
                    jobs.append(("miss", ds, s, m, f"{p}"))
    for ds in MIS_DS:
        for q in MIS_QS:
            for m in METHODS:
                for s in SEEDS5:
                    jobs.append(("mis", ds, s, m, f"{q}"))
    return jobs


def fit_nfmcd(d, k, seed, method):
    from nf_mcd.pipeline import NFMCD
    model = NFMCD.robust(k, seed=seed) if method == "nfmcd_robust" else NFMCD(n_communities=k, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
    return model


def _auc(y, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, score))


def run_one(job):
    group, ds, seed, method, x = job
    t0 = time.monotonic()
    row = dict(group=group, dataset=ds, seed=seed, method=method, x=x, k_used="",
               onmi=float("nan"), modularity=float("nan"), f1=float("nan"),
               auroc_agree=float("nan"), auroc_conf=float("nan"), error="")
    try:
        from nf_mcd import baselines as b

        if group == "orig":
            from run_realgraph import get_dataset
            d = get_dataset(ds, seed)
            model = fit_nfmcd(d, d["n_communities"], seed, method)
            s = b.score(d, b._nfmcd_result(model))
            row.update(k_used=model.n_communities, onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])

        elif group == "sweep":
            from run_realgraph import get_dataset
            d = get_dataset(f"{ds}@r{x}", seed)
            model = fit_nfmcd(d, d["n_communities"], seed, method)
            s = b.score(d, b._nfmcd_result(model))
            row.update(k_used=model.n_communities, onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])

        elif group == "miss":
            from run_missing import load_dataset, apply_mask
            p = float(x)
            d0 = load_dataset(ds)
            d = apply_mask(d0, "both", p, mask_seed=seed) if p > 0 else d0
            model = fit_nfmcd(d, d["n_communities"], seed, method)
            s = b.score(d, b._nfmcd_result(model))
            row.update(k_used=model.n_communities, onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])

        elif group == "mis":
            from run_mechanism import build_B
            q = float(x)
            setting = f"B|{ds}|q{q}"
            d, primary, bad = build_B(setting, seed)
            model = fit_nfmcd(d, d["n_communities"], seed, method)
            s = b.score(d, b._nfmcd_result(model))
            row.update(k_used=model.n_communities, onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])
            n = d["G"].number_of_nodes()
            flags = np.array(model.modality_flags_)
            both = (flags == "both") & ~np.isnan(np.asarray(model.agreement_, float))
            ybad = bad[both]
            if both.any() and 0 < ybad.sum() < both.sum():
                ag = np.asarray(model.agreement_, float)[both]
                cf = np.asarray(model.confidence_, float)[both]
                row["auroc_agree"] = _auc(ybad, -ag)
                row["auroc_conf"] = _auc(ybad, -cf)
        else:
            raise ValueError(group)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} rows already in CSV; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    t0 = time.monotonic()
    write_progress(n_done, len(jobs), "running")
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(run_one, j) for j in pending]):
                row = fut.result()
                w.writerow([row[k] for k in FIELDS])
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if row["error"]:
                    n_err += 1
                    if n_err <= 8:
                        print(f"  FAILED {row['group']}/{row['dataset']}/x={row['x']}/{row['method']}: {row['error']}", flush=True)
                if n_done % 20 == 0:
                    write_progress(n_done, len(jobs), f"running ({time.monotonic() - t0:.0f}s elapsed)")
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors, {time.monotonic() - t0:.0f}s)")
    print(f"done: {n_done}/{len(jobs)} rows, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# reference baselines pulled from existing CSVs (not recomputed)
# ---------------------------------------------------------------------------

def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ref_orig():
    """spectral_graph+content / kmeans_content for the 5 original datasets
    (baselines_results.csv) and the 2 real-graph datasets (realgraph_results.csv)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in _read_csv(os.path.join(EXP, "baselines_results.csv")):
        if r["error"] or r["method"] not in ("spectral_graph+content", "kmeans_content"):
            continue
        out[(r["method"], r["dataset"])]["onmi"].append(float(r["onmi"]))
    for r in _read_csv(os.path.join(EXP, "realgraph_results.csv")):
        if r["error"] or r["dataset"] not in ("fakeddit_real_full", "fakeddit_real_lcc"):
            continue
        if r["method"] not in ("base:spectral_graph+content", "base:kmeans_content"):
            continue
        m = r["method"].split(":", 1)[1]
        out[(m, r["dataset"])]["onmi"].append(float(r["onmi"]))
    return out


def ref_sweep():
    out = defaultdict(lambda: defaultdict(list))
    for r in _read_csv(os.path.join(EXP, "realgraph_results.csv")):
        if r["error"] or "@r" not in r["dataset"]:
            continue
        if r["method"] not in ("base:spectral_graph+content", "base:kmeans_content"):
            continue
        m = r["method"].split(":", 1)[1]
        base, ratio = r["dataset"].split("@r")
        out[(m, base, ratio)]["onmi"].append(float(r["onmi"]))
    return out


def ref_miss():
    out = defaultdict(lambda: defaultdict(list))
    for r in _read_csv(os.path.join(EXP, "missing_results.csv")):
        if r["error"] or r["mode"] != "both" or r["method"] not in ("spectral_graph+content", "kmeans_content"):
            continue
        p = round(float(r["p"]), 2)
        if p not in (0.0, 0.4, 0.8):
            continue
        out[(r["method"], r["dataset"], f"{p}")]["onmi"].append(float(r["onmi"]))
    return out


def ref_mis():
    out = defaultdict(lambda: defaultdict(list))
    for r in _read_csv(os.path.join(EXP, "mech_results.csv")):
        if r["error"] or r["part"] != "B" or r["method"] not in ("b:spectral", "b:kmeans_content"):
            continue
        ps = r["setting"].split("|")
        ds, q = ps[1], ps[2][1:]
        m = "spectral_graph+content" if r["method"] == "b:spectral" else "kmeans_content"
        out[(m, ds, q)]["onmi"].append(float(r["onmi"]))
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _ms(v):
    a = np.array([x for x in v if not np.isnan(x)], dtype=float)
    if len(a) == 0:
        return "n/a"
    return f"{a.mean():.3f}±{a.std(ddof=0):.3f}" if len(a) > 1 else f"{a.mean():.3f}"


def do_summary():
    res = load_results()
    agg = defaultdict(lambda: defaultdict(list))
    for r in res.values():
        for k in ("onmi", "modularity", "f1", "auroc_agree", "auroc_conf"):
            if not np.isnan(r[k]):
                agg[(r["group"], r["dataset"], r["method"], r["x"])][k].append(r[k])

    L = []
    L.append("Robust-preset refresh: NFMCD.robust() now includes pca_rank_div=16.")
    L.append("This SUPERSEDES robust/fcmad|raw numbers in clusterer_results.csv and rank_results.csv")
    L.append("(computed before the rank cap was added to the preset) -- those files are unchanged, this")
    L.append("file is the current source of truth for the robust preset. spectral_graph+content and")
    L.append("kmeans_content are NOT recomputed here (unchanged protocol); pulled from baselines_results.csv /")
    L.append("realgraph_results.csv / missing_results.csv / mech_results.csv.")
    L.append(f"jobs finished: {len(res)}")
    L.append("")

    # (1) original 5 + real graph
    r1 = ref_orig()
    title = "(1) Original datasets + real Fakeddit graph: LFK ONMI (k = ground truth)"
    L.append(title); L.append("-" * len(title))
    L.append(f"{'dataset':<22}{'nfmcd_default':>16}{'nfmcd_robust':>16}{'spectral+content':>19}{'kmeans_content':>17}")
    for ds in ORIG_DS:
        d_ = _ms(agg[("orig", ds, "nfmcd_default", "")]["onmi"])
        rb = _ms(agg[("orig", ds, "nfmcd_robust", "")]["onmi"])
        sp = _ms(r1[("spectral_graph+content", ds)]["onmi"])
        km = "n/a" if ds in NOCONTENT_DS else _ms(r1[("kmeans_content", ds)]["onmi"])
        L.append(f"{ds:<22}{d_:>16}{rb:>16}{sp:>19}{km:>17}")
    L.append("")
    for metric, label in (("modularity", "modularity"), ("f1", "membership F1")):
        title = f"(1b) Original datasets + real Fakeddit graph: {label} (NF-MCD only)"
        L.append(title); L.append("-" * len(title))
        L.append(f"{'dataset':<22}{'nfmcd_default':>16}{'nfmcd_robust':>16}")
        for ds in ORIG_DS:
            d_ = _ms(agg[("orig", ds, "nfmcd_default", "")][metric])
            rb = _ms(agg[("orig", ds, "nfmcd_robust", "")][metric])
            L.append(f"{ds:<22}{d_:>16}{rb:>16}")
        L.append("")

    # (2) sweep
    r2 = ref_sweep()
    for base in SWEEP_BASE:
        title = f"(2) {base}: LFK ONMI vs p_out/p_in (homophily sweep)"
        L.append(title); L.append("-" * len(title))
        L.append(f"{'method':<20}" + "".join(f"{('r=%.2f' % r):>16}" for r in RATIOS))
        for m, src in (("nfmcd_default", None), ("nfmcd_robust", None)):
            row = f"{m:<20}"
            for r in RATIOS:
                row += f"{_ms(agg[('sweep', base, m, str(r))]['onmi']):>16}"
            L.append(row)
        for m, key in (("spectral_graph+content", "spectral_graph+content"), ("kmeans_content", "kmeans_content")):
            row = f"{m:<20}"
            for r in RATIOS:
                row += f"{_ms(r2[(key, base, str(r))]['onmi']):>16}"
            L.append(row)
        L.append("")

    # (3) missing
    r3 = ref_miss()
    for ds in MISS_DS:
        title = f"(3) {ds}: LFK ONMI vs missing rate p (mode=both)"
        L.append(title); L.append("-" * len(title))
        L.append(f"{'method':<24}" + "".join(f"{('p=%.1f' % p):>14}" for p in MISS_PS))
        for m in METHODS:
            row = f"{m:<24}"
            for p in MISS_PS:
                row += f"{_ms(agg[('miss', ds, m, str(p))]['onmi']):>14}"
            L.append(row)
        for m, key in (("spectral_graph+content", "spectral_graph+content"), ("kmeans_content", "kmeans_content")):
            row = f"{m:<24}"
            for p in MISS_PS:
                row += f"{_ms(r3[(key, ds, str(p))]['onmi']):>14}"
            L.append(row)
        L.append("")

    # (4) mismatch
    r4 = ref_mis()
    for ds in MIS_DS:
        title = f"(4) {ds}: LFK ONMI vs injected cross-modal mismatch fraction q"
        L.append(title); L.append("-" * len(title))
        L.append(f"{'method':<24}" + "".join(f"{('q=%.1f' % q):>14}" for q in MIS_QS))
        for m in METHODS:
            row = f"{m:<24}"
            for q in MIS_QS:
                row += f"{_ms(agg[('mis', ds, m, str(q))]['onmi']):>14}"
            L.append(row)
        for m, key in (("spectral_graph+content", "spectral_graph+content"), ("kmeans_content", "kmeans_content")):
            row = f"{m:<24}"
            for q in MIS_QS:
                row += f"{_ms(r4[(key, ds, str(q))]['onmi']):>14}"
            L.append(row)
        L.append("")
        title = f"(4b) {ds}: detection AUROC of injected mismatches (score = -agreement / -confidence)"
        L.append(title); L.append("-" * len(title))
        L.append(f"{'method':<24}" + "".join(f"{('q=%.1f' % q):>14}" for q in MIS_QS if q > 0))
        for m in METHODS:
            row = f"{m + ' (agreement)':<24}"
            for q in MIS_QS:
                if q == 0:
                    continue
                row += f"{_ms(agg[('mis', ds, m, str(q))]['auroc_agree']):>14}"
            L.append(row)
            row = f"{m + ' (confidence)':<24}"
            for q in MIS_QS:
                if q == 0:
                    continue
                row += f"{_ms(agg[('mis', ds, m, str(q))]['auroc_conf']):>14}"
            L.append(row)
        L.append("")

    # top-line
    L.append("=== Top-line: mean ONMI over all non-PHEME settings recomputed here ===")
    for m in METHODS:
        vals = []
        for (g, ds, mm, x), d_ in agg.items():
            if mm == m and ds != "pheme" and "onmi" in d_:
                vals.extend(d_["onmi"])
        L.append(f"  {m:<16} mean over {len(vals)} rows: {np.mean(vals):.3f} (sd {np.std(vals):.3f})")
    # spectral+content reference top-line, same settings (orig incl. pheme excluded, sweep, miss, mis)
    sp_vals = []
    for ds in ORIG_DS:
        if ds != "pheme":
            sp_vals.extend(r1[("spectral_graph+content", ds)]["onmi"])
    for base in SWEEP_BASE:
        for r in RATIOS:
            sp_vals.extend(r2[("spectral_graph+content", base, str(r))]["onmi"])
    for ds in MISS_DS:
        for p in MISS_PS:
            sp_vals.extend(r3[("spectral_graph+content", ds, str(p))]["onmi"])
    for ds in MIS_DS:
        for q in MIS_QS:
            sp_vals.extend(r4[("spectral_graph+content", ds, str(q))]["onmi"])
    L.append(f"  {'spectral+content':<16} mean over {len(sp_vals)} rows: {np.mean(sp_vals):.3f} (sd {np.std(sp_vals):.3f})")

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "run":
        do_run(args.workers)
        do_summary()
    else:
        do_summary()
