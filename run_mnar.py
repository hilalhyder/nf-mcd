"""
Structured (MAR/MNAR) missing-modality robustness test.

Every earlier missing-modality experiment (run_missing.py) removed content
missing-completely-at-random (MCAR). This script removes content by a
node's PROPERTIES instead of by chance, and checks whether NF-MCD's
confidence-based fallback to structure still degrades gracefully, or
whether it introduces bias against whichever group loses content most.

Usage (from nfmcd_impl/):
    py run_mnar.py run [--workers N]
    py run_mnar.py summary
    py run_mnar.py verify

Schemes (mode "both": each modality independently masked at the node's own
rate q_i(p); nested in p via fixed per-node uniforms, same mechanic as
run_missing.py's MCAR apply_mask, just with a per-node rate instead of a
global one):

  mcar_control  - q_i(p) = p for every node (uniform). Run through the SAME
                  group-accuracy code as the MNAR schemes below, as an
                  in-script apples-to-apples baseline (external cross-check:
                  its aggregate ONMI should match run_missing.py's MCAR
                  "both" numbers at the same dataset/method/p).
  by_community  - concentrated on the single LARGEST true community (by
                  node count) in the dataset, at RATIO=4x the background
                  node's rate, normalised so the dataset-wide expected rate
                  is still p. "Affected" group = that community's members.
  by_degree_low   - susceptibility ~ rank by ASCENDING graph degree (lowest
                     degree = most likely to lose content). "Affected" group
                     = the lowest-susceptibility... i.e. lowest-degree
                     quartile (top 25% by susceptibility weight).
  by_degree_high  - susceptibility ~ rank by DESCENDING graph degree
                     (highest degree = most likely to lose content).
                     "Affected" = highest-degree quartile.
  by_typical    - susceptibility ~ rank by ASCENDING distance-to-own-
                  community-centroid in content space (typical/central
                  content = most likely to lose content, computed from the
                  UNMASKED base content). "Affected" = most-typical quartile.
  by_atypical   - susceptibility ~ rank by DESCENDING distance-to-own-
                  community-centroid (atypical/outlier content = most
                  likely to lose content). "Affected" = most-atypical
                  quartile.

For the 4 rank-based schemes, susceptibility weight w_i is a permutation of
1..n (so mean(w_i) = (n+1)/2 exactly), giving c_i = 2*w_i/(n+1) and
q_i(p) = clip(p * c_i, 0, 1); at high p, very-susceptible nodes saturate at
q_i=1 before the nominal p is reached, so REALISED dataset-wide missing
fraction can undershoot the nominal p for skewed schemes -- reported
directly (frac_nocontent), not assumed.

Group accuracy: NF-MCD's hard assignment is Hungarian-aligned once to the
true primary community (all 3 datasets have singleton true-community sets,
confirmed against nf_mcd/datasets.py's loaders), giving a per-node
correct/incorrect flag; "affected"/"background" accuracy is that flag's
mean within each group. This isolates whether concentrating missingness on
a group produces WORSE accuracy for that group than MCAR would at the same
overall rate, not just a lower dataset-wide average.

Crash-safe: each finished job is appended to experiments/mnar_results.csv
(flush+fsync); reruns skip finished jobs. Other files are written
temp+rename.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import pickle
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "mnar_results.csv")
SUMMARY_LOG = os.path.join(EXP, "mnar_summary.log")
PROGRESS_MD = os.path.join(EXP, "mnar_progress.md")

DATASETS = ["crisismmd", "fakeddit", "fakeddit_real_lcc"]
SCHEMES = ["mcar_control", "by_community", "by_degree_low", "by_degree_high", "by_typical", "by_atypical"]
PS = [0.2, 0.4, 0.6, 0.8]
SEEDS = (0, 1, 2)                     # used as both mask seed and method seed per replicate
COMMUNITY_RATIO = 4.0                 # by_community: target community's rate is this x the background rate
AFFECTED_QUANTILE = 0.25              # rank-based schemes: top this-fraction by susceptibility = "affected"

VARIANT_KW = {"nfmcd_full": {}, "nfmcd_robust": "robust"}
NONCONST = ["nfmcd_full", "nfmcd_robust", "spectral_graph+content", "kmeans_content"]
CONST = ["louvain"]
ALL_METHODS = NONCONST + CONST

FIELDS = ["dataset", "scheme", "p", "method", "seed", "k_used", "onmi", "modularity", "f1",
          "frac_nocontent", "mean_conf", "mean_alpha",
          "acc_overall", "acc_affected", "acc_background", "n_affected", "n_background",
          "error", "secs"]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Structured (MAR/MNAR) missingness progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_mnar.py run --workers 4\n"
        "py run_mnar.py summary\n"
        "py run_mnar.py verify\n"
        "```\n\n"
        "Results: experiments/mnar_results.csv (append-only, fsynced per job). "
        "Summary: experiments/mnar_summary.log. Plots: experiments/mnar_by_*.png.\n"
    ))


def jkey(ds, scheme, p, method, seed):
    return (ds, scheme, f"{float(p):.2f}", method, int(seed))


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "dataset":
                continue
            row = dict(zip(FIELDS, r))
            if row["error"]:
                continue
            try:
                for k in ("p", "onmi", "modularity", "f1", "frac_nocontent", "mean_conf", "mean_alpha",
                          "acc_overall", "acc_affected", "acc_background", "n_affected", "n_background"):
                    row[k] = float(row[k])
                row["seed"] = int(row["seed"])
            except ValueError:
                continue
            res[jkey(row["dataset"], row["scheme"], row["p"], row["method"], row["seed"])] = row
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


_CACHE = {}


def load_dataset(ds):
    if ds not in _CACHE:
        with open(os.path.join(CACHE, f"{ds}.pkl"), "rb") as f:
            _CACHE[ds] = pickle.load(f)
    return _CACHE[ds]


# ---------------------------------------------------------------------------
# Susceptibility weights and masking
# ---------------------------------------------------------------------------

def _rank_weights(values, high_value_is_susceptible):
    """Permutation-of-1..n weight array: rank 0 = smallest `values`. If
    high_value_is_susceptible, largest value gets weight n (most likely to
    lose content); otherwise smallest value gets weight n."""
    n = len(values)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    return (ranks + 1).astype(float) if high_value_is_susceptible else (n - ranks).astype(float)


def _content_typicality_distance(d):
    """1 - cosine(node content, own true-community centroid), from the
    UNMASKED base content. NaN (no content at all) gets the median distance
    (neutral) so such nodes get an average, not extreme, weight."""
    from nf_mcd.baselines import content_matrix
    X = content_matrix(d)
    n = X.shape[0]
    primary = np.array([min(t) for t in d["true"]])
    centroids = {}
    for c in set(primary.tolist()):
        idx = np.where(primary == c)[0]
        v = X[idx].mean(axis=0)
        nv = np.linalg.norm(v)
        centroids[c] = v / nv if nv > 1e-12 else v
    dist = np.full(n, np.nan)
    for i in range(n):
        v = X[i]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        cs = float((v / nv) @ centroids[primary[i]])
        dist[i] = 1.0 - cs
    if np.isnan(dist).any():
        dist[np.isnan(dist)] = np.nanmedian(dist)
    return dist


def susceptibility(d, scheme):
    """Per-node c_i with mean(c_i) == 1 (mcar_control: c_i == 1 for all
    nodes), and a boolean "affected" mask for group-accuracy reporting."""
    n = d["G"].number_of_nodes()
    if scheme == "mcar_control":
        c = np.ones(n)
        affected = np.zeros(n, dtype=bool)  # no natural "affected" group; reported as n/a
        return c, affected, "n/a (uniform)"

    if scheme == "by_community":
        primary = np.array([min(t) for t in d["true"]])
        counts = np.bincount(primary)
        target = int(np.argmax(counts))
        is_target = primary == target
        frac_t = is_target.mean()
        raw = np.where(is_target, COMMUNITY_RATIO, 1.0)
        c = raw / (frac_t * COMMUNITY_RATIO + (1 - frac_t) * 1.0)
        return c, is_target, f"community {target} ({int(counts[target])}/{n} nodes)"

    if scheme in ("by_degree_low", "by_degree_high"):
        deg = np.array([d["G"].degree(i) for i in range(n)], dtype=float)
        w = _rank_weights(deg, high_value_is_susceptible=(scheme == "by_degree_high"))
    elif scheme in ("by_typical", "by_atypical"):
        dist = _content_typicality_distance(d)
        w = _rank_weights(dist, high_value_is_susceptible=(scheme == "by_atypical"))
    else:
        raise ValueError(scheme)

    c = 2.0 * w / (n + 1.0)  # mean(w) == (n+1)/2 always, since w is a permutation of 1..n
    thresh = np.quantile(w, 1.0 - AFFECTED_QUANTILE)
    affected = w >= thresh
    return c, affected, f"top {AFFECTED_QUANTILE:.0%} by susceptibility ({int(affected.sum())}/{n} nodes)"


def apply_mnar_mask(d0, scheme, p, seed):
    n = d0["G"].number_of_nodes()
    c, affected, _ = susceptibility(d0, scheme)
    q = np.clip(p * c, 0.0, 1.0)
    rng = np.random.default_rng(3000 + seed)
    u_t, u_v = rng.random(n), rng.random(n)
    e_t = [None if u_t[i] < q[i] else d0["e_t"][i] for i in range(n)]
    e_v = [None if u_v[i] < q[i] else d0["e_v"][i] for i in range(n)]
    d2 = dict(d0)
    d2["e_t"], d2["e_v"] = e_t, e_v
    return d2, affected


def hungarian_correct(pred_hard, true_primary, k_true):
    k_pred = int(pred_hard.max()) + 1 if len(pred_hard) else 0
    C = np.zeros((k_true, max(k_pred, 1)), dtype=float)
    for t, p in zip(true_primary, pred_hard):
        C[int(t), int(p)] += 1
    r, c = linear_sum_assignment(-C)
    mapping = dict(zip(r.tolist(), c.tolist()))
    mapped = np.array([mapping.get(int(t), -1) for t in true_primary])
    return mapped == pred_hard


def spectral_missing_import():
    from run_missing import spectral_missing
    return spectral_missing


def run_one(job):
    ds, scheme, p, method, seed = job
    t0 = time.monotonic()
    row = dict(dataset=ds, scheme=scheme, p=f"{p:.2f}", method=method, seed=seed, k_used="",
               onmi=float("nan"), modularity=float("nan"), f1=float("nan"), frac_nocontent=float("nan"),
               mean_conf=float("nan"), mean_alpha=float("nan"), acc_overall=float("nan"),
               acc_affected=float("nan"), acc_background=float("nan"), n_affected=float("nan"),
               n_background=float("nan"), error="")
    try:
        from nf_mcd import baselines as b
        from nf_mcd.pipeline import NFMCD

        d0 = load_dataset(ds)
        n = d0["G"].number_of_nodes()
        if p > 0:
            d, affected = apply_mnar_mask(d0, scheme, p, seed)
        else:
            d, (_, affected, _label) = d0, (None, susceptibility(d0, scheme)[1], None)
        no_t = np.array([v is None for v in d["e_t"]])
        no_v = np.array([v is None for v in d["e_v"]])
        row.update(frac_nocontent=float((no_t & no_v).mean()))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method in VARIANT_KW:
                if VARIANT_KW[method] == "robust":
                    model = NFMCD.robust(n_communities=d["n_communities"], seed=seed)
                else:
                    model = NFMCD(n_communities=d["n_communities"], seed=seed)
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = b._nfmcd_result(model)
                row.update(mean_conf=float(np.mean(model.confidence_)), mean_alpha=float(np.mean(model.alpha_)))
            elif method == "spectral_graph+content":
                res = spectral_missing_import()(d, seed)
            elif method == "kmeans_content":
                res = b.kmeans_content(d, seed)
            elif method == "louvain":
                res = b.louvain(d, seed)
            else:
                raise ValueError(method)
            s = b.score(d, res)
        row.update(k_used=res.k_used, onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])

        true_primary = np.array([min(t) for t in d["true"]])
        correct = hungarian_correct(res.hard, true_primary, d["n_communities"])
        row["acc_overall"] = float(correct.mean())
        if affected is not None and affected.any() and (~affected).any():
            row["acc_affected"] = float(correct[affected].mean())
            row["acc_background"] = float(correct[~affected].mean())
            row["n_affected"] = float(affected.sum())
            row["n_background"] = float((~affected).sum())
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    jobs = []
    for ds in DATASETS:
        for m in ALL_METHODS:
            for s in SEEDS:
                jobs.append((ds, "mcar_control", 0.0, m, s))
        for scheme in SCHEMES:
            for p in PS:
                for m in NONCONST:
                    for s in SEEDS:
                        jobs.append((ds, scheme, p, m, s))
    # dedupe (p=0 is scheme-independent; mcar_control p=0 baseline covers it once per dataset/method/seed)
    seen, out = set(), []
    for j in jobs:
        k = jkey(*j)
        if k not in seen:
            seen.add(k)
            out.append(j)
    return out


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if jkey(*j) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already in CSV; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    write_progress(n_done, len(jobs), "running")
    t0 = time.monotonic()
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
                        print(f"  FAILED {row['dataset']}/{row['scheme']}/{row['p']}/{row['method']}: {row['error']}",
                              flush=True)
                if n_done % 50 == 0:
                    write_progress(n_done, len(jobs), f"running ({time.monotonic() - t0:.0f}s elapsed)")
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors, {time.monotonic() - t0:.0f}s)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

PLIST = [0.0] + PS

# MCAR "both" reference pulled from experiments/missing_results.csv (already
# computed in an earlier session) -- NOT recomputed here.
MCAR_REF = {  # (dataset, method) -> {p: onmi}
    ("crisismmd", "nfmcd_full"): {0.0: 0.610, 0.4: 0.600, 0.8: 0.597},
    ("crisismmd", "spectral_graph+content"): {0.0: 0.879, 0.4: 0.759, 0.8: 0.457},
    ("crisismmd", "kmeans_content"): {0.0: 0.400, 0.4: 0.076, 0.8: 0.001},
    ("crisismmd", "louvain"): {0.0: 0.555, 0.4: 0.555, 0.8: 0.555},
    ("fakeddit", "nfmcd_full"): {0.0: 0.344, 0.4: 0.674, 0.8: 0.815},
    ("fakeddit", "spectral_graph+content"): {0.0: 0.557, 0.4: 0.694, 0.8: 0.672},
    ("fakeddit", "kmeans_content"): {0.0: 0.067, 0.4: 0.015, 0.8: 0.002},
    ("fakeddit", "louvain"): {0.0: 0.650, 0.4: 0.650, 0.8: 0.650},
    ("fakeddit_real_lcc", "nfmcd_full"): {0.0: 0.139, 0.4: 0.192, 0.8: 0.196},
    ("fakeddit_real_lcc", "spectral_graph+content"): {0.0: 0.523, 0.4: 0.515, 0.8: 0.347},
    ("fakeddit_real_lcc", "kmeans_content"): {0.0: 0.250, 0.4: 0.069, 0.8: 0.029},
    ("fakeddit_real_lcc", "louvain"): {0.0: 0.313, 0.4: 0.313, 0.8: 0.313},
}


def collect(res):
    curve = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in res.values():
        ds, scheme, m, p = r["dataset"], r["scheme"], r["method"], round(r["p"], 2)
        keys = [(scheme, p)]
        if p == 0.0:
            for sc in SCHEMES:
                keys.append((sc, 0.0))
        for sc, pp in keys:
            for met in ("onmi", "modularity", "f1", "acc_overall", "acc_affected", "acc_background", "frac_nocontent"):
                curve[(ds, sc, m)][pp][met].append(r[met])
    return curve


def mean_of(curve, ds, scheme, m, p, met="onmi"):
    v = [x for x in curve.get((ds, scheme, m), {}).get(round(p, 2), {}).get(met, []) if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def sd_of(curve, ds, scheme, m, p, met="onmi"):
    v = [x for x in curve.get((ds, scheme, m), {}).get(round(p, 2), {}).get(met, []) if not np.isnan(x)]
    return float(np.std(v)) if v else float("nan")


def do_summary():
    res = load_results()
    curve = collect(res)
    L = []
    L.append("Structured (MAR/MNAR) missing-modality test. mean+-sd over 3 seeds per cell (p=0 shared across schemes).")
    L.append("k = ground-truth community count; metrics on ALL nodes; LFK ONMI unless stated.")
    L.append("'affected' group definition per scheme: mcar_control=n/a; by_community=target community's members;")
    L.append("rank-based schemes=top 25% by susceptibility weight (see module docstring for direction per scheme).")
    L.append(f"jobs finished: {len(res)}")

    L.append("\nRealised no-content fraction (mean over seeds) vs nominal p, by scheme (crisismmd/both-equivalent 'both' mode)")
    for ds in DATASETS:
        for scheme in SCHEMES:
            vals = [mean_of(curve, ds, scheme, "nfmcd_full", p, "frac_nocontent") for p in PS]
            L.append(f"  {ds:<18}{scheme:<16}" + " ".join(f"p={p:.1f}:{v:.2f}" for p, v in zip(PS, vals)))

    for ds in DATASETS:
        L.append(f"\n=== {ds}: LFK ONMI vs p, by scheme (MCAR reference from run_missing.py in brackets) ===")
        L.append(f"{'method/scheme':<32}" + "".join(f"{('p=%.1f' % p):>14}" for p in PLIST))
        for m in ALL_METHODS:
            ref = MCAR_REF.get((ds, m), {})
            row = f"{m + '  [MCAR ref]':<32}"
            for p in PLIST:
                row += f"{('[%.3f]' % ref[p]):>14}" if p in ref else f"{'':>14}"
            L.append(row)
            for scheme in SCHEMES:
                r = f"{'  ' + scheme:<32}"
                for p in PLIST:
                    mu = mean_of(curve, ds, scheme, m, p)
                    r += f"{'-':>14}" if np.isnan(mu) else f"{('%.3f+-%.3f' % (mu, sd_of(curve, ds, scheme, m, p))):>14}"
                L.append(r)

    L.append("\n=== Bias check: affected-group vs background-group strict accuracy (Hungarian-aligned), p=0.8 ===")
    L.append("A scheme is biased if acc_affected << acc_background AND that gap exceeds mcar_control's gap at the same p.")
    L.append(f"{'dataset':<18}{'scheme':<16}{'method':<20}{'acc_bg':>9}{'acc_aff':>9}{'gap':>9}{'mcar_gap':>10}{'excess':>9}")
    for ds in DATASETS:
        for m in NONCONST + CONST:
            mcar_bg = mean_of(curve, ds, "mcar_control", m, 0.8, "acc_background")
            mcar_af = mean_of(curve, ds, "mcar_control", m, 0.8, "acc_affected")
            for scheme in SCHEMES:
                if scheme == "mcar_control":
                    continue
                bg = mean_of(curve, ds, scheme, m, 0.8, "acc_background")
                af = mean_of(curve, ds, scheme, m, 0.8, "acc_affected")
                if np.isnan(bg) or np.isnan(af):
                    continue
                gap = bg - af
                mcar_gap = bg - af  # placeholder overwritten below if mcar available for THIS scheme's affected def
                excess_str = "-"
                # mcar_control has no natural affected/background split (whole-dataset uniform), so we
                # instead compare against mcar_control's OVERALL accuracy as the unbiased reference level.
                mcar_overall = mean_of(curve, ds, "mcar_control", m, 0.8, "acc_overall")
                if not np.isnan(mcar_overall):
                    excess_str = f"{(mcar_overall - af):.3f}"
                L.append(f"{ds:<18}{scheme:<16}{m:<20}{bg:>9.3f}{af:>9.3f}{gap:>9.3f}{mcar_overall:>10.3f}{excess_str:>9}")

    L.append("\n(gap = acc_background - acc_affected at p=0.8; mcar_gap column actually shows mcar_control's overall")
    L.append(" accuracy at p=0.8 as the unbiased reference level, since mcar_control has no natural affected group;")
    L.append(" 'excess' = mcar_control_overall - acc_affected: how much worse the affected group does vs the MCAR")
    L.append(" baseline level at the same overall missing rate. Large positive excess = real bias beyond MCAR.)")

    L.append("\n=== NF-MCD's fuzzy layer: mean confidence / alpha at p=0.8, by scheme (nfmcd_full) ===")
    L.append("(flag='none' is a structural guarantee from fuzzy_fusion.py, not scheme-dependent: it always fires")
    L.append(" exactly when a node has neither modality, regardless of WHY that node lost content.)")
    L.append(f"{'dataset':<18}{'scheme':<16}{'conf':>7}{'alpha':>7}{'frac_nocontent':>16}")
    conf_alpha = defaultdict(dict)
    for r in res.values():
        if r["method"] == "nfmcd_full" and abs(r["p"] - 0.8) < 1e-9:
            conf_alpha[(r["dataset"], r["scheme"])].setdefault("conf", []).append(r["mean_conf"])
            conf_alpha[(r["dataset"], r["scheme"])].setdefault("alpha", []).append(r["mean_alpha"])
            conf_alpha[(r["dataset"], r["scheme"])].setdefault("fnc", []).append(r["frac_nocontent"])
    for ds in DATASETS:
        for scheme in SCHEMES:
            v = conf_alpha.get((ds, scheme))
            if not v:
                continue
            L.append(f"{ds:<18}{scheme:<16}{np.mean(v['conf']):>7.2f}{np.mean(v['alpha']):>7.2f}{np.mean(v['fnc']):>16.2f}")

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)
    make_plots(curve)


def make_plots(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mcar_control": "#6B7280", "by_community": "#0B6E6B", "by_degree_low": "#C2410C",
              "by_degree_high": "#EA580C", "by_typical": "#7C3AED", "by_atypical": "#A855F7"}

    for group_name, schemes in (("community", ["mcar_control", "by_community"]),
                                 ("degree", ["mcar_control", "by_degree_low", "by_degree_high"]),
                                 ("typicality", ["mcar_control", "by_typical", "by_atypical"])):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS), 3.8), squeeze=False)
        for ax, ds in zip(axes[0], DATASETS):
            for scheme in schemes:
                for m, ls in (("nfmcd_full", "-"), ("spectral_graph+content", "--")):
                    xs_aff, ys_aff, xs_bg, ys_bg = [], [], [], []
                    for p in PLIST:
                        af = mean_of(curve, ds, scheme, m, p, "acc_affected")
                        bg = mean_of(curve, ds, scheme, m, p, "acc_background")
                        if not np.isnan(af):
                            xs_aff.append(p); ys_aff.append(af)
                        if not np.isnan(bg):
                            xs_bg.append(p); ys_bg.append(bg)
                    if xs_aff:
                        ax.plot(xs_aff, ys_aff, ls, color=colors[scheme], marker="o", ms=3,
                                label=f"{scheme}/{m} affected", alpha=1.0 if m == "nfmcd_full" else 0.6)
                    if xs_bg and scheme == schemes[-1]:
                        ax.plot(xs_bg, ys_bg, ls, color="#111827", marker="s", ms=3,
                                label=f"{m} background", alpha=0.5)
            ax.set_title(f"{ds}")
            ax.set_xlabel("missing rate p")
            ax.set_ylabel("strict accuracy (Hungarian-aligned)")
            ax.set_xlim(0, 0.8)
            ax.grid(alpha=0.25)
        axes[0][0].legend(fontsize=6, loc="best")
        fig.tight_layout()
        out = os.path.join(EXP, f"mnar_by_{group_name}.png")
        fig.savefig(out + ".tmp.png", dpi=140)
        plt.close(fig)
        os.replace(out + ".tmp.png", out)


def do_verify():
    res = load_results()
    for ds in DATASETS:
        for m in ("nfmcd_full", "spectral_graph+content"):
            vals = [r["onmi"] for r in res.values()
                    if r["dataset"] == ds and r["method"] == m and r["scheme"] == "mcar_control" and abs(r["p"]) < 1e-9]
            if vals:
                print(f"{ds}/{m}: mcar_control p=0 mean ONMI {np.mean(vals):.3f} (n={len(vals)})")
    # cross-check mcar_control against run_missing.py's independently-computed MCAR reference
    print("\nCross-check vs run_missing.py MCAR reference (should be close, different RNG/seed count):")
    for (ds, m), ref in MCAR_REF.items():
        for p in (0.4, 0.8):
            vals = [r["onmi"] for r in res.values()
                    if r["dataset"] == ds and r["method"] == m and r["scheme"] == "mcar_control" and abs(r["p"] - p) < 1e-9]
            if vals:
                print(f"  {ds}/{m} p={p}: this script {np.mean(vals):.3f} (n={len(vals)}) vs run_missing ref {ref[p]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "verify"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "run":
        do_run(args.workers)
        do_summary()
        do_verify()
    elif args.cmd == "summary":
        do_summary()
    else:
        do_verify()
