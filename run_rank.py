"""
Does a smaller PCA rank cap (pca_rank_div) before CCA help NF-MCD, and is it a real
CCA-quality effect or just lower alpha?

Sweep pca_rank_div in {2,4,8,16,32,64} x common_dim in {4,8,16} across: original datasets,
real-relations Fakeddit graphs, homophily sweep, missing modalities, injected mismatches;
plus held-out (K-fold) canonical correlations per rank (label-free rank rule candidate),
alpha-controlled runs (isolating CCA quality from the alpha effect), and leave-one-dataset-out
selection. Nothing in nf_mcd/* is modified and nothing is applied.

    py run_rank.py smoke
    py run_rank.py run [--workers N]      # resumable
    py run_rank.py summary                # experiments/rank_summary.log
    py run_rank.py plot                   # experiments/rank_onmi.png, rank_heldout_cca.png
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
RESULTS_CSV = os.path.join(EXP, "rank_results.csv")
SUMMARY_LOG = os.path.join(EXP, "rank_summary.log")
PROGRESS_MD = os.path.join(EXP, "rank_progress.md")

RANKS = (2, 4, 8, 16, 32, 64)
CDIMS = (4, 8, 16)
ROB_RANKS = (4, 16, 64)
AC_RANKS = (2, 4, 16, 64)
RATIOS = (0.02, 0.1, 0.25, 0.5, 1.0)
DS_MIS = ("crisismmd", "fakeddit", "fakeddit_real_lcc")
Q_MIS = (0.0, 0.1, 0.2, 0.4)
P_MISS = (0.0, 0.4, 0.8)
SEEDS3 = (0, 1, 2)
SEEDS5 = (0, 1, 2, 3, 4)
KFOLD = 5
NAN = float("nan")

FIELDS = ["setting", "seed", "method", "onmi", "modularity", "f1", "acc_all", "acc_clean", "acc_bad",
          "auroc_agree", "dprime", "ovl", "agree_mean", "agree_sd", "agree_gt08", "alpha_mean", "conf_mean",
          "n_both", "n_bad", "eff_rank", "n_comp", "cv_corr", "train_corr", "cv_cos", "train_cos",
          "cv_cos_sd", "cv_cos_gt08", "note", "error", "secs"]


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
        "# PCA rank cap sweep progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total} jobs)\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n```\n"
        "cd C:\\Users\\HP\\Projects\\Paper_SSC\\nfmcd_impl\n"
        "py run_rank.py run --workers 4\npy run_rank.py summary\npy run_rank.py plot\n```\n\n"
        "Results: experiments/rank_results.csv (append-only, fsynced per job). "
        "Summary: experiments/rank_summary.log.\n"))


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


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

def all_settings():
    """(setting, seeds); setting = '<group>~<key>'."""
    out = []
    for ds in ("crisismmd", "fakeddit", "pheme"):
        out.append((f"orig~{ds}", SEEDS3))
    for ds in ("fakeddit_real_full", "fakeddit_real_lcc"):
        out.append((f"real~{ds}", SEEDS3))
    for b in ("crisismmd", "fakeddit"):
        for r in RATIOS:
            out.append((f"sweep~{b}@r{r}", SEEDS3))
    for ds in ("crisismmd", "fakeddit_real_lcc"):
        for p in P_MISS:
            out.append((f"miss~{ds}|both@p{p}", SEEDS3))
    for ds in DS_MIS:
        for q in Q_MIS:
            out.append((f"mis~B|{ds}|q{q}", SEEDS5))
    for ds in DS_MIS:
        for q in (0.0, 0.2):
            out.append((f"H~B|{ds}|q{q}", SEEDS3))
    return out


def expected_methods(setting):
    group = setting.split("~", 1)[0]
    if group == "H":
        return [f"cv|r{r}|c{c}" for r in RANKS for c in CDIMS]
    refs = ["ref:alpha05", "ref:concat05", "ref:spectral", "ref:kmeans_content", "ref:louvain", "ref:structure_only"]
    if setting == "orig~pheme":  # text only: the rank cap is irrelevant, control
        return ["nf|r4|c8"] + refs
    m = [f"nf|r{r}|c{c}" for r in RANKS for c in CDIMS]
    m += [f"rob|r{r}|c8" for r in ROB_RANKS]
    m += [f"ac_self|r{r}" for r in AC_RANKS] + [f"ac_fixed|r{r}" for r in AC_RANKS]
    return m + refs


def build(setting, seed):
    group, key = setting.split("~", 1)
    from run_realgraph import get_dataset
    bad = None
    if group in ("orig", "real", "sweep"):
        d = get_dataset(key, seed)
        primary = np.array([min(t) for t in d["true"]])
    elif group == "miss":
        from run_missing import apply_mask
        ds, p = key.split("|both@p")
        d = apply_mask(get_dataset(ds, seed), "both", float(p), seed)
        primary = np.array([min(t) for t in d["true"]])
    else:
        from run_mechanism import build_B
        d, primary, bad = build_B(key, seed)
    return d, primary, bad


def eff_rank(n_paired, r, dim_t, dim_v):
    if dim_t is None or dim_v is None or n_paired < 10:
        return None
    safe = max(2, n_paired // r)
    return max(1, min(safe, dim_t, dim_v, n_paired - 1))


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

def fit_model(d, seed, kind, r=4, c=8, alpha=None):
    from nf_mcd.pipeline import NFMCD
    k = d["n_communities"]
    n = d["G"].number_of_nodes()
    e_t, e_v = d["e_t"], d["e_v"]
    kw = dict(pca_rank_div=r, common_dim=c)
    if alpha is not None:
        kw.update(alpha_min=alpha, alpha_max=alpha)
    if kind == "rob":
        model = NFMCD.robust(k, seed=seed, **kw)
    elif kind == "concat05":
        model = NFMCD(n_communities=k, seed=seed, content_features="raw_pca", alpha_min=0.5, alpha_max=0.5)
    elif kind == "structure_only":
        model = NFMCD(n_communities=k, seed=seed)
        e_t, e_v = [None] * n, [None] * n
    elif kind == "alpha05":
        model = NFMCD(n_communities=k, seed=seed, alpha_min=0.5, alpha_max=0.5)
    else:
        model = NFMCD(n_communities=k, seed=seed, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=e_t, image_embeddings=e_v)
    return model


def _score_row(base, d, res, ctx):
    from nf_mcd import baselines as b
    from run_mechanism import correct_vector
    s = b.score(d, res)
    row = dict(base)
    row.update(onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])
    ok = correct_vector(res.hard, ctx["primary"], d["n_communities"])
    row["acc_all"] = float(ok.mean())
    bad, both = ctx["bad"], ctx["both"]
    if bad is not None:
        cl, bd = both & ~bad, both & bad
        row["acc_clean"] = float(ok[cl].mean()) if cl.any() else NAN
        row["acc_bad"] = float(ok[bd].mean()) if bd.any() else NAN
    return row


def nf_row(base, model, d, ctx):
    from nf_mcd import baselines as b
    from run_mechanism import _auc_ap, _ovl
    row = _score_row(base, d, b._nfmcd_result(model), ctx)
    ag = np.asarray(model.agreement_, float)
    cf = np.asarray(model.confidence_, float)
    flags = np.array(model.modality_flags_)
    both = (flags == "both") & ~np.isnan(ag)
    row["n_both"] = int(both.sum())
    row["alpha_mean"] = float(np.mean(model.alpha_))
    row["conf_mean"] = float(cf.mean())
    fus = model.fusion
    if getattr(fus, "_fitted", False):
        row["eff_rank"] = int(fus._pca_t.n_components_)
        row["n_comp"] = int(fus._cca.x_rotations_.shape[1])
    if both.any():
        row["agree_mean"], row["agree_sd"] = float(ag[both].mean()), float(ag[both].std())
        row["agree_gt08"] = float((ag[both] > 0.8).mean())
        bad = ctx["bad"]
        if bad is not None:
            y = bad[both]
            row["n_bad"] = int(y.sum())
            if 0 < y.sum() < len(y):
                row["auroc_agree"] = _auc_ap(y, -ag[both])[0]
                ac, ab = ag[both & ~bad], ag[both & bad]
                pooled = math.sqrt((ac.var() + ab.var()) / 2)
                row["dprime"] = float((ac.mean() - ab.mean()) / pooled) if pooled > 0 else NAN
                row["ovl"] = _ovl(ac, ab)
    return row


def ref_row(base, kind, d, seed, ctx, group):
    from nf_mcd import baselines as b
    if kind == "spectral":
        if group == "miss":
            from run_missing import spectral_missing
            res = spectral_missing(d, seed)
        else:
            res = b.spectral_graph_content(d, seed)
    elif kind == "kmeans_content":
        res = b.kmeans_content(d, seed)
    elif kind == "louvain":
        res = b.louvain(d, seed)
    else:
        res = b._nfmcd_result(fit_model(d, seed, kind))
    return _score_row(base, d, res, ctx)


# --------------------------------------------------------------------------
# held-out CCA
# --------------------------------------------------------------------------

def cv_rows(setting, seed, todo, d, bad):
    """K-fold held-out canonical correlations / pair cosines per (rank cap, common_dim)."""
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from run_mechanism import _auc_ap
    n = d["G"].number_of_nodes()
    idx = np.array([i for i in range(n) if d["e_t"][i] is not None and d["e_v"][i] is not None])
    Xt = np.stack([d["e_t"][i] for i in idx])
    Xv = np.stack([d["e_v"][i] for i in idx])
    ybad = bad[idx] if bad is not None else None
    npair = len(idx)
    rows = []
    for m in todo:
        t1 = time.monotonic()
        base = dict(setting=setting, seed=seed, method=m, error="")
        _, rs, cs = m.split("|")
        r, c = int(rs[1:]), int(cs[1:])
        er = eff_rank(npair, r, Xt.shape[1], Xv.shape[1])
        row = dict(base)
        if er is None or er < c:
            row.update(note="skip:eff_rank<common_dim", eff_rank=er if er is not None else "")
            row["secs"] = 0.0
            rows.append(row)
            continue
        try:
            tr_c, cv_c, tr_cos, cv_cos = [], [], [], []
            cv_all = np.full(npair, np.nan)
            for tr, te in KFold(KFOLD, shuffle=True, random_state=seed).split(idx):
                rk = max(1, min(er, len(tr) - 1))

                def prep(X, mean=None, std=None):
                    return (X - mean) / std

                mt, st = Xt[tr].mean(0), Xt[tr].std(0) + 1e-8
                mv, sv = Xv[tr].mean(0), Xv[tr].std(0) + 1e-8
                pt = PCA(n_components=rk, random_state=seed).fit(prep(Xt[tr], mt, st))
                pv = PCA(n_components=rk, random_state=seed).fit(prep(Xv[tr], mv, sv))
                Zt_tr, Zv_tr = pt.transform(prep(Xt[tr], mt, st)), pv.transform(prep(Xv[tr], mv, sv))
                zmt, zst = Zt_tr.mean(0), Zt_tr.std(0) + 1e-8
                zmv, zsv = Zv_tr.mean(0), Zv_tr.std(0) + 1e-8
                nc = max(1, min(c, rk, len(tr) - 1))
                cca = CCA(n_components=nc, scale=False).fit((Zt_tr - zmt) / zst, (Zv_tr - zmv) / zsv)

                def proj(Xa, Xb):
                    a = (pt.transform(prep(Xa, mt, st)) - zmt) / zst @ cca.x_rotations_
                    bb = (pv.transform(prep(Xb, mv, sv)) - zmv) / zsv @ cca.y_rotations_
                    return a, bb

                def comp_corr(a, bb):
                    return float(np.mean([np.corrcoef(a[:, j], bb[:, j])[0, 1] for j in range(a.shape[1])]))

                def cosines(a, bb):
                    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
                    bb = bb / (np.linalg.norm(bb, axis=1, keepdims=True) + 1e-12)
                    return (a * bb).sum(1)

                a_tr, b_tr = proj(Xt[tr], Xv[tr])
                a_te, b_te = proj(Xt[te], Xv[te])
                tr_c.append(comp_corr(a_tr, b_tr))
                cv_c.append(comp_corr(a_te, b_te))
                ct = cosines(a_tr, b_tr)
                ce = cosines(a_te, b_te)
                tr_cos.append(float(ct.mean()))
                cv_cos.append(float(ce.mean()))
                cv_all[te] = ce
            row.update(cv_corr=float(np.mean(cv_c)), train_corr=float(np.mean(tr_c)),
                       cv_cos=float(np.mean(cv_cos)), train_cos=float(np.mean(tr_cos)),
                       cv_cos_sd=float(np.nanstd(cv_all)), cv_cos_gt08=float(np.nanmean(cv_all > 0.8)),
                       eff_rank=int(er), n_comp=int(min(c, er)))
            if ybad is not None and 0 < ybad.sum() < len(ybad):
                row["auroc_agree"] = _auc_ap(ybad, -cv_all)[0]
                row["n_bad"] = int(ybad.sum())
                row["n_both"] = int(npair)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        row["secs"] = round(time.monotonic() - t1, 2)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

def run_job(args):
    setting, seed, todo = args
    group = setting.split("~", 1)[0]
    try:
        d, primary, bad = build(setting, seed)
        n = d["G"].number_of_nodes()
        both = np.array([d["e_t"][i] is not None and d["e_v"][i] is not None for i in range(n)])
        ctx = dict(primary=primary, both=both, bad=(bad & both) if bad is not None else None)
    except Exception as exc:  # noqa: BLE001
        msg = f"build: {type(exc).__name__}: {exc}".replace("\n", " ")
        return [dict(setting=setting, seed=seed, method=m, error=msg, secs=0.0) for m in todo]
    if group == "H":
        return cv_rows(setting, seed, todo, d, bad)

    n_paired = int(both.sum())
    dim_t = next((v.shape[0] for v in d["e_t"] if v is not None), None)
    dim_v = next((v.shape[0] for v in d["e_v"] if v is not None), None)
    cache = {}

    def get(kind, r=4, c=8, alpha=None):
        key = (kind, r, c, alpha)
        if key not in cache:
            cache[key] = fit_model(d, seed, kind, r, c, alpha)
        return cache[key]

    rows = []
    for m in todo:
        t1 = time.monotonic()
        base = dict(setting=setting, seed=seed, method=m, error="")
        try:
            head = m.split("|")[0].split(":")[0]
            if head == "nf" or head == "rob":
                _, rs, cs = m.split("|")
                r, c = int(rs[1:]), int(cs[1:])
                er = eff_rank(n_paired, r, dim_t, dim_v)
                if er is not None and er < c:
                    row = dict(base)
                    row.update(note="skip:eff_rank<common_dim", eff_rank=er)
                else:
                    row = nf_row(base, get(head, r, c), d, ctx)
            elif head in ("ac_self", "ac_fixed"):
                r = int(m.split("|")[1][1:])
                a_src = get("nf", r, 8) if head == "ac_self" else get("nf", 4, 8)
                a = float(np.clip(np.mean(a_src.alpha_), 0.01, 0.99))
                row = nf_row(base, get("nf", r, 8, a), d, ctx)
                row["note"] = f"alpha={a:.3f}"
            else:
                row = ref_row(base, m.split(":", 1)[1], d, seed, ctx, group)
        except Exception as exc:  # noqa: BLE001
            row = dict(base)
            row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        row["secs"] = round(time.monotonic() - t1, 2)
        rows.append(row)
    return rows


def pending_jobs():
    done = {}
    for r in load_rows():
        if not r["error"]:
            done.setdefault((r["setting"], int(r["seed"])), set()).add(r["method"])
    jobs = []
    for setting, seeds in all_settings():
        exp = expected_methods(setting)
        for seed in seeds:
            todo = [m for m in exp if m not in done.get((setting, seed), set())]
            if todo:
                jobs.append((setting, seed, todo))
    return jobs


def do_run(workers):
    os.makedirs(EXP, exist_ok=True)
    jobs = pending_jobs()
    total_all = sum(len(s) for _, s in all_settings())
    total = len(jobs)
    print(f"{total} jobs pending (of {total_all}); workers={workers}", flush=True)
    write_progress("run", total_all - total, total_all, f"{total} pending")
    if not jobs:
        return
    f, w = open_append()
    done = errors = 0
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
    for s, seed, ms in [("mis~B|crisismmd|q0.2", 0, ["nf|r4|c8", "nf|r16|c8", "ac_self|r16", "ref:louvain"]),
                        ("H~B|crisismmd|q0.2", 0, ["cv|r4|c8", "cv|r16|c8", "cv|r64|c16"])]:
        for r in run_job((s, seed, ms)):
            keys = ["method", "onmi", "acc_bad", "auroc_agree", "agree_mean", "alpha_mean", "eff_rank",
                    "cv_corr", "train_corr", "cv_cos", "note", "error", "secs"]
            print({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def load_df():
    import pandas as pd
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["error"].isna()]
    df = df[~df["note"].fillna("").str.startswith("skip")].copy()
    df["group"] = df["setting"].str.split("~").str[0]
    df["key"] = df["setting"].str.split("~").str[1]
    return df


def family(setting):
    group, key = setting.split("~", 1)
    if group == "H":
        return None
    if group in ("orig", "sweep"):
        base = key.split("@r")[0]
        if base == "pheme":
            return None
        return base
    if group == "real":
        return "fakeddit_real"
    if group == "miss":
        return "crisismmd" if key.startswith("crisismmd") else "fakeddit_real"
    if group == "mis":
        ds = key.split("|")[1]
        q = float(key.split("|q")[1])
        if q == 0.0:
            return None
        return {"crisismmd": "crisismmd", "fakeddit": "fakeddit"}.get(ds, "fakeddit_real")
    return None


def do_summary():
    import pandas as pd
    df = load_df()
    df["fam"] = df["setting"].map(family)
    out = []

    def P(s=""):
        out.append(s)

    def group_of(row_setting):
        g, key = row_setting.split("~", 1)
        if g == "orig":
            return "pheme" if key == "pheme" else "orig"
        if g == "mis":
            return "mis" if float(key.split("|q")[1]) > 0 else "mis0"
        return g

    df["grp"] = df["setting"].map(group_of)
    GR = ["orig", "pheme", "real", "sweep", "miss", "mis"]

    sm = df.groupby(["method", "setting"], as_index=False).mean(numeric_only=True)
    sm["grp"] = sm["setting"].map(group_of)

    nonp = sm[~sm.setting.str.startswith("H~") & (sm.setting != "orig~pheme")]
    tot_settings = nonp.setting.nunique()

    def feasible_ranks(c):
        return [r for r in RANKS if nonp[nonp.method == f"nf|r{r}|c{c}"].setting.nunique() >= 0.8 * tot_settings]

    def common(c):
        sets = [set(nonp[nonp.method == f"nf|r{r}|c{c}"].setting) for r in feasible_ranks(c)]
        return set.intersection(*sets)

    S = {c: common(c) for c in CDIMS}
    cur = {"set": None}

    def gmean(method, col="onmi", grp=None):
        x = sm[(sm.method == method)]
        if cur["set"] is not None:
            x = x[x.setting.isin(cur["set"])]
        if grp is not None:
            x = x[x.grp == grp]
        return float(x[col].mean()) if len(x) else float("nan")

    def fmt(v, nd=3):
        return "  n/a " if v != v else f"{v:.{nd}f}"

    nf_methods = [f"nf|r{r}|c{c}" for r in RANKS for c in CDIMS]
    P("PCA rank cap sweep: LFK ONMI unless stated; k = ground-truth count; mean of setting means "
      "(3 seeds; 5 for injected mismatches)")
    P("Groups: orig = crisismmd, fakeddit | pheme = text-only control | real = real-relations Fakeddit (2) | "
      "sweep = homophily sweep (10) | miss = missing-modality both p{0,.4,.8} (6) | mis = injected mismatches q>0 (9)")
    P("rank = n_paired // pca_rank_div (div 4 = default, div 2 = looser, div 64 = tight)")

    P(f"\nNOTE on coverage: a config is skipped where rank = max(2, n_paired//div) < common_dim (few pairs left).")
    for cval in CDIMS:
        rs = feasible_ranks(cval)
        P(f"  common_dim={cval}: ranks compared = div {rs}; {len(S[cval])} of {tot_settings} non-PHEME settings "
          f"feasible for all of them (tables below use only those, so every row covers the same settings)"
          + ("" if len(rs) == len(RANKS) else f"; div {[r for r in RANKS if r not in rs]} dropped (too many skips)"))

    # T1
    for cval in CDIMS:
        cur["set"] = S[cval]
        P("\nTable 1 (common_dim=%d): ONMI by pca_rank_div and setting group" % cval)
        P(f"{'div':>4} " + " ".join(f"{g:>7}" for g in GR) + f" {'all(non-PHEME)':>15} {'d vs div4':>10}")
        base = np.nanmean([gmean(f"nf|r4|c{cval}", grp=g) for g in GR if g != "pheme"])
        for r in feasible_ranks(cval):
            vals = [gmean(f"nf|r{r}|c{cval}", grp=g) for g in GR]
            allv = np.nanmean([v for g, v in zip(GR, vals) if g != "pheme"])
            P(f"{r:>4} " + " ".join(f"{fmt(v):>7}" for v in vals) + f" {fmt(allv):>15} {allv - base:>+10.3f}")
    P("\nTable 1b: mean ONMI over the settings feasible for each common_dim (columns cover different setting sets; "
      "compare down a column, not across)")
    P(f"{'div':>4} " + " ".join(f"{'c'+str(c):>8}" for c in CDIMS))
    for r in RANKS:
        vals = []
        for c in CDIMS:
            cur["set"] = S[c]
            vals.append(np.nanmean([gmean(f"nf|r{r}|c{c}", grp=g) for g in GR if g != "pheme"])
                        if r in feasible_ranks(c) else float("nan"))
        P(f"{r:>4} " + " ".join(f"{fmt(v):>8}" for v in vals))

    cur["set"] = None
    P("\nTable 1d: missing-modality group by p (c=8, ONMI, mean over crisismmd and fakeddit_real_lcc; n/a = skipped "
      "because rank < common_dim)")
    P(f"{'div':>4} " + " ".join(f"{'p='+str(p):>8}" for p in P_MISS))
    for r in RANKS:
        vals = []
        for p in P_MISS:
            x = sm[(sm.method == f"nf|r{r}|c8") & sm.setting.str.startswith("miss~") & sm.setting.str.endswith(f"p{p}")]
            vals.append(float(x.onmi.mean()) if len(x) == 2 else float("nan"))
        P(f"{r:>4} " + " ".join(f"{fmt(v):>8}" for v in vals))
    for ref in ("ref:alpha05", "ref:spectral", "ref:kmeans_content", "ref:structure_only"):
        vals = []
        for p in P_MISS:
            x = sm[(sm.method == ref) & sm.setting.str.startswith("miss~") & sm.setting.str.endswith(f"p{p}")]
            vals.append(float(x.onmi.mean()) if len(x) == 2 else float("nan"))
        P(f"{ref[4:][:4]:>4} " + " ".join(f"{fmt(v):>8}" for v in vals) + f"   ({ref})")

    cur["set"] = S[8]
    P("\nReferences (same groups, same c=8 setting set): ")
    for ref in ["ref:alpha05", "ref:concat05", "ref:spectral", "ref:kmeans_content", "ref:louvain",
                "ref:structure_only"]:
        vals = [gmean(ref, grp=g) for g in GR]
        allv = np.nanmean([v for g, v in zip(GR, vals) if g != "pheme"])
        P(f"{ref:>20} " + " ".join(f"{fmt(v):>7}" for v in vals) + f" {fmt(allv):>15}")
    P("Robust preset (adaptive-m FCM, raw-PCA content) with different rank caps (c=8):")
    for r in ROB_RANKS:
        vals = [gmean(f"rob|r{r}|c8", grp=g) for g in GR]
        allv = np.nanmean([v for g, v in zip(GR, vals) if g != "pheme"])
        P(f"{'rob r'+str(r):>20} " + " ".join(f"{fmt(v):>7}" for v in vals) + f" {fmt(allv):>15}")

    P("\nTable 1c: costs at c=8: modularity and F1, group mean (non-PHEME settings)")
    P(f"{'div':>4} {'modularity':>11} {'F1':>7} {'ONMI':>7}")
    for r in RANKS:
        P(f"{r:>4} " + " ".join(
            f"{np.nanmean([gmean(f'nf|r{r}|c8', col=c, grp=g) for g in GR if g != 'pheme']):>{w}.3f}"
            for c, w in (("modularity", 11), ("f1", 7), ("onmi", 7))))
    P("    per-group modularity, div 4 vs 16 vs 64 (c=8): ")
    for r in (4, 16, 64):
        P(f"    div {r:>2}: " + " ".join(f"{g}={gmean(f'nf|r{r}|c8', 'modularity', g):.3f}" for g in GR))

    # T2 agreement stats
    P("\nTable 2: agreement/alpha statistics per rank cap (c=8), settings with real pairs and no injected corruption")
    P("  (orig crisismmd/fakeddit, real, sweep, miss p=0; 'both' nodes)")
    P(f"{'div':>4} {'eff_rank(crisis)':>17} {'agree mean':>11} {'agree sd':>9} {'share>0.8':>10} "
      f"{'alpha':>7} {'conf':>7}")
    clean = sm[sm.setting.map(lambda s: group_of(s) in ("orig", "real", "sweep") and not s.endswith("pheme") or
                              s.startswith("miss~") and s.endswith("p0.0")) & sm.setting.isin(S[8])]
    for r in RANKS:
        x = clean[clean.method == f"nf|r{r}|c8"]
        er = sm[(sm.method == f"nf|r{r}|c8") & (sm.setting == "orig~crisismmd")]["eff_rank"]
        P(f"{r:>4} {fmt(float(er.mean()) if len(er) else float('nan'), 0):>17} {fmt(x.agree_mean.mean()):>11} "
          f"{fmt(x.agree_sd.mean()):>9} {fmt(x.agree_gt08.mean()):>10} {fmt(x.alpha_mean.mean()):>7} "
          f"{fmt(x.conf_mean.mean()):>7}")

    # T3 detection
    P("\nTable 3: AUROC of agreement for detecting injected mismatches (c=8) by rank cap; d' and overlap")
    cols = [(ds, q) for ds in DS_MIS for q in (0.1, 0.2, 0.4)]
    P(f"{'div':>4} " + " ".join(f"{ds[:9]+'@'+str(q):>15}" for ds, q in cols))
    for r in RANKS:
        vals = []
        for ds, q in cols:
            x = sm[(sm.method == f"nf|r{r}|c8") & (sm.setting == f"mis~B|{ds}|q{q}")]
            vals.append(float(x.auroc_agree.mean()) if len(x) else float("nan"))
        P(f"{r:>4} " + " ".join(f"{fmt(v):>15}" for v in vals))
    P("   mean d' (clean vs corrupted pairs, higher = better separated) / histogram overlap, averaged over q>0:")
    for r in RANKS:
        x = sm[(sm.method == f"nf|r{r}|c8") & (sm.grp == "mis")]
        P(f"   div {r:>2}: d'={fmt(x.dprime.mean())}  overlap={fmt(x.ovl.mean())}  "
          f"AUROC(all)={fmt(x.auroc_agree.mean())}")
    P("   ONMI degradation from q=0 to q=0.4 (c=8), by dataset:")
    P(f"{'div':>4} " + " ".join(f"{ds:>19}" for ds in DS_MIS))
    for r in RANKS:
        vals = []
        for ds in DS_MIS:
            a = sm[(sm.method == f"nf|r{r}|c8") & (sm.setting == f"mis~B|{ds}|q0.0")].onmi
            b = sm[(sm.method == f"nf|r{r}|c8") & (sm.setting == f"mis~B|{ds}|q0.4")].onmi
            vals.append(float(b.mean() - a.mean()) if len(a) and len(b) else float("nan"))
        P(f"{r:>4} " + " ".join(f"{v:>+19.3f}" for v in vals))
    for ref in ["ref:alpha05", "ref:concat05", "ref:spectral"]:
        vals = []
        for ds in DS_MIS:
            a = sm[(sm.method == ref) & (sm.setting == f"mis~B|{ds}|q0.0")].onmi
            b = sm[(sm.method == ref) & (sm.setting == f"mis~B|{ds}|q0.4")].onmi
            vals.append(float(b.mean() - a.mean()) if len(a) and len(b) else float("nan"))
        P(f"{ref[4:]:>4} " + " ".join(f"{v:>+19.3f}" for v in vals))

    # T4 held-out CCA
    P("\nTable 4: held-out (5-fold) canonical correlations vs training, and pair cosine (label-free rank rule?)")
    P("  cv_corr = mean held-out correlation of the c canonical pairs; gap = train - held-out; "
      "cv_cos = mean held-out cosine of projected pairs (what 'agreement' measures)")
    hdf = df[df.grp.isin(["H"]) | df.group.eq("H")].copy()
    hs = hdf.groupby(["method", "setting"], as_index=False).mean(numeric_only=True)
    for st in [f"H~B|{ds}|q{q}" for ds in DS_MIS for q in (0.0, 0.2)]:
        x = hs[hs.setting == st]
        if not len(x):
            continue
        P(f"\n  {st}  (n_paired ~ see eff_rank; c=8 rows shown, plus best c per rank in the CSV)")
        P(f"{'div':>4} {'eff_rank':>8} {'train_corr':>11} {'cv_corr':>8} {'gap':>7} {'train_cos':>10} "
          f"{'cv_cos':>7} {'cv_cos_sd':>9} {'cv>0.8':>7} {'AUROC(cv)':>10}")
        for r in RANKS:
            y = x[x.method == f"cv|r{r}|c8"]
            if not len(y):
                continue
            y = y.iloc[0]
            P(f"{r:>4} {y.eff_rank:>8.0f} {y.train_corr:>11.3f} {y.cv_corr:>8.3f} {y.train_corr - y.cv_corr:>7.3f} "
              f"{y.train_cos:>10.3f} {y.cv_cos:>7.3f} {y.cv_cos_sd:>9.3f} {y.cv_cos_gt08:>7.3f} "
              f"{fmt(y.auroc_agree):>10}")
    P("\n  Rank chosen by held-out correlation (argmax cv_corr, c=8 and any c) vs rank with best downstream ONMI:")
    P(f"{'dataset':>20} {'argmax cv_corr (c=8)':>21} {'argmax cv_corr (any c)':>23} {'best ONMI div (c=8)':>20} "
      f"{'min gap div (c=8)':>18}")
    for ds in DS_MIS:
        st = f"H~B|{ds}|q0.0"
        x = hs[hs.setting == st]
        if not len(x):
            continue
        c8 = x[x.method.str.endswith("|c8")]
        best_cv8 = c8.loc[c8.cv_corr.idxmax(), "method"] if len(c8) else "n/a"
        best_any = x.loc[x.cv_corr.idxmax(), "method"]
        gaps = c8.assign(gap=c8.train_corr - c8.cv_corr)
        min_gap = gaps.loc[gaps.gap.idxmin(), "method"] if len(gaps) else "n/a"
        # downstream ONMI on the matching original dataset
        ostr = {"crisismmd": "orig~crisismmd", "fakeddit": "orig~fakeddit",
                "fakeddit_real_lcc": "real~fakeddit_real_lcc"}[ds]
        o = sm[(sm.setting == ostr) & (sm.method.isin([f"nf|r{r}|c8" for r in RANKS]))]
        best_o = o.loc[o.onmi.idxmax(), "method"] if len(o) else "n/a"
        P(f"{ds:>20} {best_cv8:>21} {best_any:>23} {best_o:>20} {min_gap:>18}")

    # T5 alpha-controlled
    P("\nTable 5: alpha-controlled ONMI (c=8). self = alpha clamped to this rank's own mean alpha; "
      "fixed = alpha clamped to the div-4 default's mean alpha (isolates CCA quality)")
    P(f"{'div':>4} {'group':>6} {'ONMI default':>13} {'alpha mean':>11} {'ONMI self':>10} {'ONMI fixed':>11}")
    for g in ("orig", "real", "sweep", "miss", "mis"):
        for r in AC_RANKS:
            P(f"{r:>4} {g:>6} {fmt(gmean(f'nf|r{r}|c8', grp=g)):>13} {fmt(gmean(f'nf|r{r}|c8', 'alpha_mean', g)):>11} "
              f"{fmt(gmean(f'ac_self|r{r}', grp=g)):>10} {fmt(gmean(f'ac_fixed|r{r}', grp=g)):>11}")
    P("  Reading: 'fixed' varies ONLY the CCA/fused content across ranks at the same alpha; 'default' also "
      "changes alpha. Gain(default) - gain(fixed) ~ share due to the alpha effect.")
    P("\n  Summary over all non-PHEME groups:")
    P(f"{'div':>4} {'default':>8} {'alpha':>7} {'self':>7} {'fixed':>7}")
    base_all = np.nanmean([gmean('nf|r4|c8', grp=g) for g in GR if g != 'pheme'])
    for r in AC_RANKS:
        de = np.nanmean([gmean(f'nf|r{r}|c8', grp=g) for g in GR if g != 'pheme'])
        al = np.nanmean([gmean(f'nf|r{r}|c8', 'alpha_mean', g) for g in GR if g != 'pheme'])
        se = np.nanmean([gmean(f'ac_self|r{r}', grp=g) for g in GR if g != 'pheme'])
        fi = np.nanmean([gmean(f'ac_fixed|r{r}', grp=g) for g in GR if g != 'pheme'])
        P(f"{r:>4} {de:>8.3f} {al:>7.3f} {se:>7.3f} {fi:>7.3f}   (div-4 default = {base_all:.3f})")

    # T6 LODO
    P("\nTable 6: leave-one-dataset-out selection of (pca_rank_div, common_dim) over the 18-config grid")
    P("  families: crisismmd (orig, sweep, miss, mis q>0), fakeddit (orig, sweep, mis q>0), "
      "fakeddit_real (real x2, miss, mis q>0); PHEME excluded from selection")
    fam_df = df[df.fam.notna() & df.method.isin(nf_methods) & df.setting.isin(S[8])].copy()
    # candidates = configs with a result on every setting of the c=8 common set (identical coverage for all)
    have = fam_df.groupby("method").setting.nunique()
    n_used = int(have.max())
    cand_ok = set(have[have == n_used].index)
    fam_df = fam_df[fam_df.method.isin(cand_ok)]
    P(f"  candidate configs (feasible on all {n_used} family settings of the c=8 common set): {sorted(cand_ok)}")
    fams = ["crisismmd", "fakeddit", "fakeddit_real"]
    fam_setting_mean = fam_df.groupby(["method", "fam", "setting"], as_index=False).mean(numeric_only=True)
    fam_mean = fam_setting_mean.groupby(["method", "fam"], as_index=False).mean(numeric_only=True)
    seed_fam = fam_df.groupby(["method", "fam", "seed"], as_index=False).mean(numeric_only=True)
    default = "nf|r4|c8"
    P(f"{'held-out':>14} {'chosen on other 2':>18} {'tuned':>7} {'default':>8} {'diff':>7} {'2*noise':>8} "
      f"{'verdict':>8} {'per-setting W/T/L':>18}")
    for hold in fams:
        others = [f for f in fams if f != hold]
        cand = fam_mean[fam_mean.fam.isin(others)].groupby("method").onmi.mean()
        chosen = cand.idxmax()
        tuned = fam_mean[(fam_mean.method == chosen) & (fam_mean.fam == hold)].onmi.iloc[0]
        dflt = fam_mean[(fam_mean.method == default) & (fam_mean.fam == hold)].onmi.iloc[0]
        sd_t = seed_fam[(seed_fam.method == chosen) & (seed_fam.fam == hold)].onmi.std()
        sd_d = seed_fam[(seed_fam.method == default) & (seed_fam.fam == hold)].onmi.std()
        noise = np.nansum([sd_t, sd_d])
        diff = tuned - dflt
        verdict = "tie" if abs(diff) < max(0.02, noise) else ("tuned" if diff > 0 else "default")
        a = fam_setting_mean[(fam_setting_mean.method == chosen) & (fam_setting_mean.fam == hold)].set_index("setting").onmi
        b = fam_setting_mean[(fam_setting_mean.method == default) & (fam_setting_mean.fam == hold)].set_index("setting").onmi
        dd = (a - b).dropna()
        w = int((dd > 0.02).sum())
        l = int((dd < -0.02).sum())
        t = len(dd) - w - l
        P(f"{hold:>14} {chosen:>18} {tuned:>7.3f} {dflt:>8.3f} {diff:>+7.3f} {2 * noise:>8.3f} {verdict:>8} "
          f"{f'{w}/{t}/{l}':>18}")
    P("  (per-setting W/T/L uses a fixed 0.02 tie band; family-level verdict uses max(0.02, sd_tuned + sd_default))")
    P("\n  Also the fixed configs, held-out family means (ONMI): ")
    P(f"{'config':>12} " + " ".join(f"{f:>14}" for f in fams) + f" {'mean':>7}")
    for cfg in ["nf|r4|c8", "nf|r8|c8", "nf|r16|c8", "nf|r32|c8", "nf|r64|c8", "nf|r16|c16", "nf|r16|c4"]:
        v = [float(fam_mean[(fam_mean.method == cfg) & (fam_mean.fam == f)].onmi.mean()) for f in fams]
        P(f"{cfg:>12} " + " ".join(f"{x:>14.3f}" for x in v) + f" {np.nanmean(v):>7.3f}")

    # PHEME control
    cur["set"] = None
    P("\nControl: PHEME (text-only, rank cap irrelevant) nf|r4|c8 ONMI = "
      f"{fmt(gmean('nf|r4|c8', grp='pheme'))}; spectral = {fmt(gmean('ref:spectral', grp='pheme'))}")
    n_err = 0
    if os.path.exists(RESULTS_CSV):
        raw = pd.read_csv(RESULTS_CSV)
        n_err = int(raw["error"].notna().sum())
        n_skip = int(raw["note"].fillna("").str.startswith("skip").sum())
        P(f"\nRows: {len(raw)}  errors: {n_err}  skipped (eff_rank < common_dim): {n_skip}")
    atomic_write_text(SUMMARY_LOG, "\n".join(out) + "\n")
    print("\n".join(out))


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_df()

    def group_of(s):
        g, key = s.split("~", 1)
        if g == "orig":
            return "pheme" if key == "pheme" else "orig"
        if g == "mis":
            return "mis" if float(key.split("|q")[1]) > 0 else "mis0"
        return g

    df["grp"] = df["setting"].map(group_of)
    sm = df.groupby(["method", "setting"], as_index=False).mean(numeric_only=True)
    sm["grp"] = sm["setting"].map(group_of)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, c in zip(axes, CDIMS):
        for g in ("orig", "real", "sweep", "miss", "mis"):
            ys = [sm[(sm.method == f"nf|r{r}|c{c}") & (sm.grp == g)].onmi.mean() for r in RANKS]
            ax.plot(RANKS, ys, marker="o", label=g)
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("pca_rank_div (higher = smaller rank cap)")
        ax.set_ylabel("LFK ONMI (group mean)")
        ax.set_title(f"common_dim = {c}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "rank_onmi.png"), dpi=140)
    plt.close(fig)

    hd = df[df.group == "H"]
    hs = hd.groupby(["method", "setting"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, ds in zip(axes, DS_MIS):
        st = f"H~B|{ds}|q0.0"
        tr = [hs[(hs.method == f"cv|r{r}|c8") & (hs.setting == st)].train_corr.mean() for r in RANKS]
        cv = [hs[(hs.method == f"cv|r{r}|c8") & (hs.setting == st)].cv_corr.mean() for r in RANKS]
        cs = [hs[(hs.method == f"cv|r{r}|c8") & (hs.setting == st)].cv_cos.mean() for r in RANKS]
        ax.plot(RANKS, tr, marker="o", label="train canonical corr")
        ax.plot(RANKS, cv, marker="s", label="held-out canonical corr")
        ax.plot(RANKS, cs, marker="^", ls="--", label="held-out pair cosine")
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("pca_rank_div")
        ax.set_title(f"{ds} (c=8, no injected corruption)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "rank_heldout_cca.png"), dpi=140)
    plt.close(fig)
    print("wrote rank_onmi.png, rank_heldout_cca.png")


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
