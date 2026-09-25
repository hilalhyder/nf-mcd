"""
Does NF-MCD's cross-modal agreement / fuzzy confidence detect mismatched
image-text pairs, and does down-weighting them help community detection?

Part A: synthetic graphs with KNOWN misaligned nodes (generate_synthetic_multimodal_graph).
Part B: real cached embeddings (crisismmd, fakeddit_real_lcc, fakeddit) with label-aware
        injected misalignment (image swapped for one from a different community).

One job = (part, setting, seed); it computes every benefit method and every detection
score and appends the rows (flush + fsync). Resumable: finished (part, setting, seed,
method) rows are skipped.

    py run_mechanism.py smoke                # one tiny job, prints rows
    py run_mechanism.py run [--workers N]    # resumable
    py run_mechanism.py summary              # tables -> experiments/mech_summary.log
    py run_mechanism.py plot                 # experiments/mech_detection.png, mech_benefit.png
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "mech_results.csv")
SUMMARY_LOG = os.path.join(EXP, "mech_summary.log")
PROGRESS_MD = os.path.join(EXP, "mech_progress.md")

SEEDS = (0, 1, 2, 3, 4)
A_SIZES = ((300, 4), (600, 6))
A_MIS = (0.0, 0.1, 0.2, 0.4)
A_CONTENT = {"hi": 3.0, "lo": 0.08}      # centroid_scale (noise_scale fixed at 1.0)
A_STRUCT = {"hi": 0.02, "lo": 0.10}      # p_out (p_in fixed at 0.18)
B_DATASETS = ("crisismmd", "fakeddit_real_lcc", "fakeddit")
B_Q = (0.0, 0.1, 0.2, 0.4)

BENCH = ["default", "rank16", "robust", "alpha05", "concat05", "text_only",
         "image_only", "structure_only", "spectral", "kmeans_content"]
DET = ["default", "rank16", "cons", "kmmatch", "random"]
FIELDS = ["part", "setting", "seed", "method", "onmi", "modularity", "acc_all", "acc_clean", "acc_bad",
          "auroc_agree", "ap_agree", "auroc_conf", "ap_conf", "prevalence", "n_both", "n_bad",
          "agree_mean_clean", "agree_sd_clean", "agree_mean_bad", "agree_sd_bad", "dprime", "ovl",
          "conf_mean_clean", "conf_mean_bad", "error", "secs"]
NAN = float("nan")


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Mechanism (misalignment detection) progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total} jobs)\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n```\n"
        "cd C:\\Users\\HP\\Projects\\Paper_SSC\\nfmcd_impl\n"
        "py run_mechanism.py run --workers 4\npy run_mechanism.py summary\npy run_mechanism.py plot\n```\n\n"
        "Results: experiments/mech_results.csv (append-only, fsynced per job). "
        "Summary: experiments/mech_summary.log.\n"))


def load_rows():
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
    if not new:  # drop a torn final line, if any
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


# --------------------------------------------------------------------------
# settings / data
# --------------------------------------------------------------------------

def all_settings():
    out = []
    for n, k in A_SIZES:
        for mis in A_MIS:
            for c in A_CONTENT:
                for s in A_STRUCT:
                    out.append(f"A|n{n}|mis{mis}|c{c}|s{s}")
    for ds in B_DATASETS:
        for q in B_Q:
            out.append(f"B|{ds}|q{q}")
    return out


def parse_setting(setting):
    p = setting.split("|")
    if p[0] == "A":
        return dict(part="A", n=int(p[1][1:]), mis=float(p[2][3:]), c=p[3][1:], s=p[4][1:])
    return dict(part="B", ds=p[1], q=float(p[2][1:]))


def expected_methods():
    return [f"b:{m}" for m in BENCH] + [f"d:{m}" for m in DET]


def build_A(setting, seed):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph as gen
    ps = parse_setting(setting)
    n = ps["n"]
    k = dict(A_SIZES)[n]
    s = gen(n_nodes=n, n_communities=k, p_in=0.18, p_out=A_STRUCT[ps["s"]],
            missing_modality_rate=0.1, misalignment_rate=ps["mis"], overlap_rate=0.10,
            centroid_scale=A_CONTENT[ps["c"]], noise_scale=1.0, seed=seed)
    d = dict(G=s.G, e_t=list(s.text_embeddings), e_v=list(s.image_embeddings),
             true=s.true_communities, n_communities=k)
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    primary = np.repeat(np.arange(k), sizes)
    bad = np.zeros(n, bool)
    if s.misaligned_nodes:
        bad[list(s.misaligned_nodes)] = True
    return d, primary, bad


def build_B(setting, seed):
    from run_realgraph import get_dataset
    ps = parse_setting(setting)
    d0 = get_dataset(ps["ds"], seed)
    n = d0["G"].number_of_nodes()
    primary = np.array([min(t) for t in d0["true"]])
    e_t, e_v = d0["e_t"], d0["e_v"]
    paired = np.array([e_t[i] is not None and e_v[i] is not None for i in range(n)])
    has_img = np.array([e_v[i] is not None for i in range(n)])
    rng = np.random.default_rng(2000 + seed)
    u = rng.random(n)
    img_idx = np.where(has_img)[0]
    donors = np.full(n, -1)
    for i in range(n):
        cand = img_idx[primary[img_idx] != primary[i]]
        if len(cand):
            donors[i] = rng.choice(cand)
    bad = paired & (u < ps["q"]) & (donors >= 0)
    e_v2 = list(e_v)
    for i in np.where(bad)[0]:
        e_v2[i] = e_v[donors[i]]
    d = dict(d0)
    d["e_v"] = e_v2
    return d, primary, bad


# --------------------------------------------------------------------------
# models and metrics
# --------------------------------------------------------------------------

def fit_kind(d, seed, kind):
    from nf_mcd.pipeline import NFMCD
    k = d["n_communities"]
    n = d["G"].number_of_nodes()
    e_t, e_v = d["e_t"], d["e_v"]
    kw = {}
    if kind == "rank16":
        kw = dict(pca_rank_div=16)
    elif kind == "cons":
        kw = dict(confidence_mode="consistency")
    elif kind == "alpha05":
        kw = dict(alpha_min=0.5, alpha_max=0.5)
    elif kind == "concat05":
        kw = dict(content_features="raw_pca", alpha_min=0.5, alpha_max=0.5)
    elif kind == "text_only":
        e_v = [None] * n
    elif kind == "image_only":
        e_t = [None] * n
    elif kind == "structure_only":
        e_t, e_v = [None] * n, [None] * n
    model = NFMCD.robust(k, seed=seed) if kind == "robust" else NFMCD(n_communities=k, seed=seed, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=e_t, image_embeddings=e_v)
    return model


def correct_vector(hard, primary, k_true):
    from scipy.optimize import linear_sum_assignment
    hard = np.asarray(hard, int)
    k_pred = int(hard.max()) + 1
    M = np.zeros((k_pred, k_true))
    np.add.at(M, (hard, primary), 1)
    r, c = linear_sum_assignment(-M)
    mapping = -np.ones(k_pred, int)
    mapping[r] = c
    return mapping[hard] == primary


def _auc_ap(y, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, bool)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return NAN, NAN
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _ovl(a, b, bins=40):
    if len(a) < 2 or len(b) < 2:
        return NAN
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    return float(np.minimum(ha / len(a), hb / len(b)).sum())


def bench_row(base, kind, d, seed, ctx, get):
    from nf_mcd import baselines as b
    if kind == "spectral":
        res = b.spectral_graph_content(d, seed)
    elif kind == "kmeans_content":
        res = b.kmeans_content(d, seed)
    else:
        res = b._nfmcd_result(get(kind))
    s = b.score(d, res)
    ok = correct_vector(res.hard, ctx["primary"], d["n_communities"])
    both, bad = ctx["both"], ctx["bad"]
    clean_m = both & ~bad
    bad_m = both & bad
    row = dict(base)
    row.update(onmi=s["onmi"], modularity=s["modularity"], acc_all=float(ok.mean()),
               acc_clean=float(ok[clean_m].mean()) if clean_m.any() else NAN,
               acc_bad=float(ok[bad_m].mean()) if bad_m.any() else NAN,
               n_both=int(both.sum()), n_bad=int(bad_m.sum()))
    return row


def det_row(base, kind, d, seed, ctx, get):
    both_in, bad = ctx["both"], ctx["bad"]
    row = dict(base)
    if kind in ("default", "rank16", "cons"):
        model = get(kind)
        ag = np.asarray(model.agreement_, float)
        cf = np.asarray(model.confidence_, float)
        flags = np.array(model.modality_flags_)
        both = (flags == "both") & ~np.isnan(ag)
        y = bad[both]
        row["n_both"], row["n_bad"] = int(both.sum()), int(y.sum())
        row["prevalence"] = float(y.mean()) if both.any() else NAN
        row["auroc_agree"], row["ap_agree"] = _auc_ap(y, -ag[both])
        row["auroc_conf"], row["ap_conf"] = _auc_ap(y, -cf[both])
        ac, ab = ag[both & ~bad], ag[both & bad]
        cc, cb = cf[both & ~bad], cf[both & bad]
        if len(ac):
            row["agree_mean_clean"], row["agree_sd_clean"] = float(ac.mean()), float(ac.std())
            row["conf_mean_clean"] = float(cc.mean())
        if len(ab):
            row["agree_mean_bad"], row["agree_sd_bad"] = float(ab.mean()), float(ab.std())
            row["conf_mean_bad"] = float(cb.mean())
        if len(ac) > 1 and len(ab) > 1:
            pooled = math.sqrt((ac.var() + ab.var()) / 2)
            row["dprime"] = float((ac.mean() - ab.mean()) / pooled) if pooled > 0 else NAN
            row["ovl"] = _ovl(ac, ab)
    else:
        idx = np.where(both_in)[0]
        y = bad[idx]
        row["n_both"], row["n_bad"] = int(len(idx)), int(y.sum())
        row["prevalence"] = float(y.mean()) if len(idx) else NAN
        if kind == "random":
            score = np.random.default_rng(999 + seed).random(len(idx))
        else:
            from scipy.optimize import linear_sum_assignment
            from sklearn.cluster import KMeans
            k = d["n_communities"]
            Xt = np.stack([d["e_t"][i] / (np.linalg.norm(d["e_t"][i]) + 1e-12) for i in idx])
            Xv = np.stack([d["e_v"][i] / (np.linalg.norm(d["e_v"][i]) + 1e-12) for i in idx])
            lt = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Xt)
            lv = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Xv)
            M = np.zeros((k, k))
            np.add.at(M, (lt, lv), 1)
            r, c = linear_sum_assignment(-M)
            mp = np.zeros(k, int)
            mp[r] = c
            score = (mp[lt] != lv).astype(float)
        row["auroc_agree"], row["ap_agree"] = _auc_ap(y, score)
    return row


def run_job(args):
    part, setting, seed, todo = args
    t0 = time.monotonic()
    rows = []
    try:
        d, primary, bad = build_A(setting, seed) if part == "A" else build_B(setting, seed)
        n = d["G"].number_of_nodes()
        both = np.array([d["e_t"][i] is not None and d["e_v"][i] is not None for i in range(n)])
        ctx = dict(primary=primary, bad=bad & both, both=both)
    except Exception as exc:  # noqa: BLE001
        msg = f"build: {type(exc).__name__}: {exc}".replace("\n", " ")
        return [dict(part=part, setting=setting, seed=seed, method=m, error=msg, secs=0.0) for m in todo]

    cache = {}

    def get(kind):
        if kind not in cache:
            cache[kind] = fit_kind(d, seed, kind)
        return cache[kind]

    for m in todo:
        base = dict(part=part, setting=setting, seed=seed, method=m, error="")
        t1 = time.monotonic()
        try:
            kind_prefix, kind = m.split(":", 1)
            row = bench_row(base, kind, d, seed, ctx, get) if kind_prefix == "b" else det_row(base, kind, d, seed, ctx, get)
        except Exception as exc:  # noqa: BLE001
            row = dict(base)
            row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        row["secs"] = round(time.monotonic() - t1, 2)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def pending_jobs():
    done = {}
    for r in load_rows():
        if not r["error"]:
            done.setdefault((r["part"], r["setting"], int(r["seed"])), set()).add(r["method"])
    exp = expected_methods()
    jobs = []
    for seed in SEEDS:
        for s in all_settings():
            part = s[0]
            todo = [m for m in exp if m not in done.get((part, s, seed), set())]
            if todo:
                jobs.append((part, s, seed, todo))
    return jobs


def do_run(workers):
    os.makedirs(EXP, exist_ok=True)
    jobs = pending_jobs()
    total_all = len(SEEDS) * len(all_settings())
    total = len(jobs)
    print(f"{total} jobs pending (of {total_all}); workers={workers}", flush=True)
    write_progress("run", total_all - total, total_all, f"{total} pending")
    if not jobs:
        return
    f, w = open_append()
    done = 0
    errors = 0
    t0 = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_job, j): j for j in jobs}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    errors += bool(r.get("error"))
                    w.writerow({k: r.get(k, "") for k in FIELDS})
                f.flush()
                os.fsync(f.fileno())
                done += 1
                write_progress("run", total_all - total + done, total_all,
                               f"{errors} errors; {time.monotonic() - t0:.0f}s elapsed")
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total} jobs  errors={errors}  {time.monotonic() - t0:.0f}s", flush=True)
    finally:
        f.close()
    write_progress("run complete", total_all, total_all, f"finished, {errors} errors")
    print(f"done: {errors} error rows", flush=True)


def do_smoke():
    s = "A|n300|mis0.2|chi|shi"
    rows = run_job(("A", s, 0, expected_methods()))
    for r in rows:
        keys = ["method", "onmi", "acc_clean", "acc_bad", "auroc_agree", "auroc_conf", "dprime", "ovl", "error", "secs"]
        print({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    s = "B|fakeddit_real_lcc|q0.2"
    rows = run_job(("B", s, 0, ["b:default", "d:default", "d:kmmatch"]))
    for r in rows:
        print({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "")) for k in
               ["method", "onmi", "acc_bad", "auroc_agree", "auroc_conf", "dprime", "error"]})


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def load_df():
    import pandas as pd
    rows = load_rows()
    df = pd.DataFrame(rows, columns=FIELDS)
    for c in FIELDS:
        if c not in ("part", "setting", "method", "error"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["error"].fillna("") == ""].copy()
    ps = df["setting"].map(parse_setting)
    for key in ("n", "mis", "c", "s", "ds", "q"):
        df[key] = ps.map(lambda p, key=key: p.get(key))
    return df


def _f(x, nd=3):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ms(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.3f}±{v.std():.3f}"


def _table(title, colnames, rowlabels, cells, note=""):
    w = max(len(r) for r in rowlabels) + 2
    lines = [title, "-" * len(title), " " * w + "".join(f"{c:>15}" for c in colnames)]
    for r, cs in zip(rowlabels, cells):
        lines.append(f"{r:<{w}}" + "".join(f"{c:>15}" for c in cs))
    if note:
        lines.append(note)
    return "\n".join(lines) + "\n"


def do_summary():
    df = load_df()
    if df.empty:
        print("no rows yet")
        return
    out = ["Cross-modal misalignment: detection and benefit",
           "Part A = synthetic (known misaligned nodes); Part B = real embeddings with injected image swaps.",
           f"rows: {len(df)}; seeds per cell up to {len(SEEDS)}; ± is sd over runs (seeds and, in Part A, n / structure level).",
           "AUROC uses score = -agreement (or -confidence); positive class = misaligned. 0.5 = chance.", ""]

    A = df[df["part"] == "A"]
    B = df[df["part"] == "B"]

    # ---- T1 detection Part A
    det_rows = [("agreement (default, CCA rank n/4)", "d:default", "auroc_agree"),
                ("confidence (default)", "d:default", "auroc_conf"),
                ("agreement (rank n/16)", "d:rank16", "auroc_agree"),
                ("confidence (rank n/16)", "d:rank16", "auroc_conf"),
                ("consistency confidence", "d:cons", "auroc_conf"),
                ("k-means cluster mismatch (baseline)", "d:kmmatch", "auroc_agree"),
                ("random score", "d:random", "auroc_agree")]
    for c in ("hi", "lo"):
        sub = A[(A["c"] == c) & (A["mis"] > 0)]
        mis_list = sorted(sub["mis"].unique())
        cells = []
        for label, m, col in det_rows:
            cells.append([_ms(sub[(sub["method"] == m) & (sub["mis"] == mis)][col]) for mis in mis_list])
        out.append(_table(f"T1{'ab'[c=='lo']}. Part A detection AUROC, content {'informative (centroid 3.0)' if c=='hi' else 'weak (centroid 0.08)'}"
                          " (mean over structure levels and n)",
                          [f"mis={m}" for m in mis_list], [r[0] for r in det_rows], cells))

    # by n and structure, agreement default vs rank16 at mis 0.2
    lab, cells = [], []
    for c in ("hi", "lo"):
        for n in (300, 600):
            sub = A[(A["c"] == c) & (A["n"] == n) & (A["mis"] == 0.2)]
            lab.append(f"content={c}, n={n}")
            cells.append([_ms(sub[sub["method"] == "d:default"]["auroc_agree"]),
                          _ms(sub[sub["method"] == "d:rank16"]["auroc_agree"]),
                          _ms(sub[sub["method"] == "d:cons"]["auroc_conf"]),
                          _ms(sub[sub["method"] == "d:default"]["n_both"]).split("±")[0]])
    out.append(_table("T1c. Part A detection at mis=0.2 by size (few pairs vs many): AUROC agreement default / rank n/16 / consistency conf; mean paired nodes",
                      ["default", "rank n/16", "consistency", "paired nodes"], lab, cells))
    lab, cells = [], []
    for c in ("hi", "lo"):
        for s in ("hi", "lo"):
            sub = A[(A["c"] == c) & (A["s"] == s) & (A["mis"] == 0.2)]
            lab.append(f"content={c}, structure={s}")
            cells.append([_ms(sub[sub["method"] == "d:default"]["auroc_agree"]),
                          _ms(sub[sub["method"] == "d:cons"]["auroc_conf"])])
    out.append(_table("T1d. Part A detection at mis=0.2 by structure informativeness (consistency confidence uses the graph)",
                      ["agreement default", "consistency conf"], lab, cells))

    # ---- T2 agreement distribution
    lab, cells = [], []
    for c in ("hi", "lo"):
        for m in ("d:default", "d:rank16"):
            sub = A[(A["c"] == c) & (A["mis"] > 0) & (A["method"] == m)]
            lab.append(f"A content={c}, {m[2:]}")
            cells.append([_ms(sub["agree_mean_clean"]), _ms(sub["agree_mean_bad"]),
                          _ms(sub["dprime"]), _ms(sub["ovl"]), _ms(sub["conf_mean_clean"]), _ms(sub["conf_mean_bad"])])
    for ds in B_DATASETS:
        for m in ("d:default", "d:rank16"):
            sub = B[(B["ds"] == ds) & (B["q"] > 0) & (B["method"] == m)]
            lab.append(f"B {ds}, {m[2:]}")
            cells.append([_ms(sub["agree_mean_clean"]), _ms(sub["agree_mean_bad"]),
                          _ms(sub["dprime"]), _ms(sub["ovl"]), _ms(sub["conf_mean_clean"]), _ms(sub["conf_mean_bad"])])
    out.append(_table("T2. Agreement of clean vs misaligned pairs in the shared (CCA) space; d' = separation in pooled sd; overlap = histogram overlap (1 = identical)",
                      ["agree clean", "agree bad", "d'", "overlap", "conf clean", "conf bad"], lab, cells))

    # ---- T3 benefit Part A
    bm = [("NF-MCD default", "b:default"), ("NF-MCD rank n/16", "b:rank16"), ("NF-MCD robust preset", "b:robust"),
          ("alpha fixed 0.5", "b:alpha05"), ("no-fuzzy concat, alpha 0.5", "b:concat05"),
          ("NF-MCD text-only", "b:text_only"), ("NF-MCD image-only", "b:image_only"),
          ("NF-MCD structure-only", "b:structure_only"), ("spectral (graph + content)", "b:spectral"),
          ("k-means on content", "b:kmeans_content")]
    for c in ("hi", "lo"):
        sub = A[A["c"] == c]
        mis_list = sorted(sub["mis"].unique())
        cells = [[_ms(sub[(sub["method"] == m) & (sub["mis"] == mis)]["onmi"]) for mis in mis_list] +
                 [f"{sub[(sub['method'] == m) & (sub['mis'] == mis_list[-1])]['onmi'].mean() - sub[(sub['method'] == m) & (sub['mis'] == 0)]['onmi'].mean():+.3f}"]
                 for _, m in bm]
        out.append(_table(f"T3{'ab'[c=='lo']}. Part A LFK ONMI vs misalignment rate, content {'informative' if c=='hi' else 'weak'} (mean over structure levels and n)",
                          [f"mis={m}" for m in mis_list] + ["drop 0->0.4"], [r[0] for r in bm], cells))

    # ---- T4 accuracy on clean vs corrupted
    for c in ("hi", "lo"):
        lab, cells = [], []
        for label, m in bm:
            row = []
            for mis in (0.2, 0.4):
                sub = A[(A["c"] == c) & (A["mis"] == mis) & (A["method"] == m)]
                row += [_ms(sub["acc_clean"]), _ms(sub["acc_bad"])]
            lab.append(label)
            cells.append(row)
        out.append(_table(f"T4{'ab'[c=='lo']}. Part A node accuracy on aligned vs misaligned paired nodes, content {'informative' if c=='hi' else 'weak'}",
                          ["clean@0.2", "bad@0.2", "clean@0.4", "bad@0.4"], lab, cells))

    # ---- T5 Part B
    for ds in B_DATASETS:
        sub = B[B["ds"] == ds]
        qs = sorted(sub["q"].unique())
        cells = [[_ms(sub[(sub["method"] == m) & (sub["q"] == q)][col]) for q in qs if q > 0]
                 for _, m, col in det_rows]
        out.append(_table(f"T5a. Part B {ds}: detection AUROC of injected swaps",
                          [f"q={q}" for q in qs if q > 0], [r[0] for r in det_rows], cells))
        cells = [[_ms(sub[(sub["method"] == m) & (sub["q"] == q)]["onmi"]) for q in qs] +
                 [f"{sub[(sub['method'] == m) & (sub['q'] == qs[-1])]['onmi'].mean() - sub[(sub['method'] == m) & (sub['q'] == 0)]['onmi'].mean():+.3f}"]
                 for _, m in bm]
        out.append(_table(f"T5b. Part B {ds}: LFK ONMI vs injected-swap fraction q",
                          [f"q={q}" for q in qs] + ["drop 0->0.4"], [r[0] for r in bm], cells))
        cells = [[_ms(sub[(sub["method"] == m) & (sub["q"] == q)]["acc_bad"]) for q in qs if q > 0] +
                 [_ms(sub[(sub["method"] == m) & (sub["q"] == 0.2)]["acc_clean"])] for _, m in bm]
        out.append(_table(f"T5c. Part B {ds}: node accuracy on the CORRUPTED nodes (last column: accuracy on clean paired nodes at q=0.2)",
                          [f"q={q}" for q in qs if q > 0] + ["clean@0.2"], [r[0] for r in bm], cells))

    # ---- T6 fuzzy weighting vs alternatives
    def wtl(df_, a, b, col):
        w = t = l = 0
        for _, g in df_.groupby("setting"):
            ga = g[g["method"] == a][col].dropna()
            gb = g[g["method"] == b][col].dropna()
            if len(ga) < 2 or len(gb) < 2:
                continue
            diff = ga.mean() - gb.mean()
            tol = max(0.02, ga.std() + gb.std())
            if diff > tol:
                w += 1
            elif diff < -tol:
                l += 1
            else:
                t += 1
        return f"{w}/{t}/{l}"

    lab, cells = [], []
    groups = [("A content=hi, mis>0", A[(A["c"] == "hi") & (A["mis"] > 0)]),
              ("A content=lo, mis>0", A[(A["c"] == "lo") & (A["mis"] > 0)])] + \
             [(f"B {ds}, q>0", B[(B["ds"] == ds) & (B["q"] > 0)]) for ds in B_DATASETS]
    for name, g in groups:
        lab.append(name)
        cells.append([wtl(g, "b:default", "b:alpha05", "onmi"), wtl(g, "b:default", "b:concat05", "onmi"),
                      wtl(g, "b:default", "b:alpha05", "acc_bad"), wtl(g, "b:default", "b:concat05", "acc_bad"),
                      wtl(g, "b:default", "b:spectral", "onmi")])
    out.append(_table("T6. Default NF-MCD (fuzzy alpha) wins/ties/losses over settings vs alpha fixed 0.5, no-fuzzy concat (alpha 0.5), and spectral+content; tie = |diff| < max(0.02, sd sum)",
                      ["ONMI vs a=0.5", "ONMI vs concat", "acc_bad vs a=0.5", "acc_bad vs concat", "ONMI vs spectral"], lab, cells))

    # ---- sanity: q=0 reproduces known values
    known = {"crisismmd": 0.609, "fakeddit": 0.308, "fakeddit_real_lcc": 0.126}
    out.append("Sanity check: Part B q=0, b:default ONMI vs known earlier default values (k = truth)")
    for ds, kv in known.items():
        sub = B[(B["ds"] == ds) & (B["q"] == 0) & (B["method"] == "b:default")]
        out.append(f"  {ds:<18} this run {_ms(sub['onmi'])}   earlier {kv:.3f}")
    out.append("")
    atomic_write_text(SUMMARY_LOG, "\n".join(out))
    print("\n".join(out))


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_df()
    if df.empty:
        return
    A, B = df[df["part"] == "A"], df[df["part"] == "B"]
    det = [("agreement (default)", "d:default", "auroc_agree"), ("agreement (rank n/16)", "d:rank16", "auroc_agree"),
           ("consistency conf", "d:cons", "auroc_conf"), ("k-means mismatch", "d:kmmatch", "auroc_agree"),
           ("random", "d:random", "auroc_agree")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    for ax, (title, sub, xcol) in zip(axes, [("A content informative", A[A["c"] == "hi"], "mis"), ("A content weak", A[A["c"] == "lo"], "mis")] +
                                      [(f"B {ds}", B[B["ds"] == ds], "q") for ds in B_DATASETS]):
        for label, m, col in det:
            g = sub[(sub["method"] == m) & (sub[xcol] > 0)].groupby(xcol)[col].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=2, label=label)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("misaligned fraction")
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("detection AUROC")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "mech_detection.png"), dpi=130)
    plt.close(fig)

    bm = [("default", "b:default"), ("rank n/16", "b:rank16"), ("robust", "b:robust"), ("alpha 0.5", "b:alpha05"),
          ("concat, alpha 0.5", "b:concat05"), ("text-only", "b:text_only"), ("image-only", "b:image_only"),
          ("structure-only", "b:structure_only"), ("spectral+content", "b:spectral"), ("k-means content", "b:kmeans_content")]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    panels = [("A content informative", A[A["c"] == "hi"], "mis"), ("A content weak", A[A["c"] == "lo"], "mis")] + \
             [(f"B {ds}", B[B["ds"] == ds], "q") for ds in B_DATASETS]
    for ax, (title, sub, xcol) in zip(axes[0], panels):
        for label, m in bm:
            g = sub[sub["method"] == m].groupby(xcol)["onmi"].mean()
            ax.plot(g.index, g.values, marker="o", label=label)
        ax.set_title(title + ": ONMI")
        ax.set_xlabel("misaligned fraction")
    for ax, (title, sub, xcol) in zip(axes[1], panels):
        for label, m in bm:
            g = sub[(sub["method"] == m) & (sub[xcol] > 0)].groupby(xcol)["acc_bad"].mean()
            ax.plot(g.index, g.values, marker="o", label=label)
        ax.set_title(title + ": accuracy on misaligned nodes")
        ax.set_xlabel("misaligned fraction")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "mech_benefit.png"), dpi=130)
    plt.close(fig)
    print("plots written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "summary", "plot"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "smoke":
        do_smoke()
    elif a.cmd == "run":
        do_run(a.workers)
    elif a.cmd == "summary":
        do_summary()
    else:
        do_plot()
