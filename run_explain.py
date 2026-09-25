"""
Explainability evaluation of NF-MCD (new file; nf_mcd/* is untouched).

  E1  what the global IF-THEN rules explain (fidelity vs majority baseline and
      vs post-hoc decision-tree surrogates on interpretable / structural inputs)
  E2  faithfulness of alpha (content-vs-structure trust) by block interventions
  E3  stability across seeds (partition, explanation type, rule set)
  E4  explanations under injected missing modalities

Usage (from nfmcd_impl/, with `py`):
    py run_explain.py run [--workers N]   # E1/E2 fit units + E4 units (resumable)
    py run_explain.py e3                  # stability from saved per-fit artifacts
    py run_explain.py summary             # tables + plots
    py run_explain.py all [--workers N]

Every finished unit is appended to experiments/explain_results.csv (flush+fsync)
and marked with a "_done" row, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
CSV = os.path.join(EXP, "explain_results.csv")
PROG = os.path.join(EXP, "explain_progress.md")
LOG = os.path.join(EXP, "explain_summary.log")

DATASETS = ["crisismmd", "fakeddit", "fakeddit_real_lcc", "pheme", "dblp", "amazon"]
VARIANTS = ["default", "robust"]
SEEDS = [0, 1, 2, 3, 4]
E4_DS = ["crisismmd", "fakeddit_real_lcc"]
E4_P = [0.2, 0.6]
LEAVES = [2, 3, 4, 6, 8, 12, 16, 24, 32]
HEADER = ["exp", "dataset", "variant", "seed", "cond", "metric", "value"]


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------

def _atomic_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_pickle(path, obj):
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_rows(rows):
    new = (not os.path.exists(CSV)) or os.path.getsize(CSV) == 0
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())


def read_rows():
    rows = []
    if not os.path.exists(CSV):
        return rows
    with open(CSV, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if len(r) == len(HEADER):
                rows.append(r)
    return rows


def done_units():
    return {r[4] for r in read_rows() if r[5] == "_done"}


def write_progress(note=""):
    units = all_units()
    done = done_units()
    n_done = sum(1 for u in units if u[0] in done)
    e3_done = sum(1 for r in read_rows() if r[0] == "E3" and r[5] == "_done")
    _atomic_text(PROG, (
        "# Explainability evaluation progress\n\n"
        f"- Units done: **{n_done}/{len(units)}** (E1/E2 fit units + E4 units)\n"
        f"- E3 groups done: **{e3_done}/{len(DATASETS) * len(VARIANTS)}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished units are skipped automatically)\n\n"
        "```\ncd C:\\Users\\HP\\Projects\\Paper_SSC\\nfmcd_impl\n"
        "py run_explain.py run --workers 4\npy run_explain.py e3\npy run_explain.py summary\n```\n\n"
        "Results: experiments/explain_results.csv (append-only, fsynced). Summary: experiments/explain_summary.log.\n"
    ))


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

def fit_model(d, seed, variant):
    from nf_mcd.pipeline import NFMCD
    n = d["n_communities"]
    model = NFMCD(n_communities=n, seed=seed) if variant == "default" else NFMCD.robust(n_communities=n, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
    return model


def content_block(model, d):
    from nf_mcd import topology as topo
    if model.content_features == "raw_pca":
        return topo.raw_pca_content(d["e_t"], d["e_v"], dim=model.raw_pca_dim, seed=model.seed)
    return model.fused_content_


def fused_features(model, Zc):
    from nf_mcd import topology as topo
    return topo.fuse_features(Zc, model.Z_s_, model.alpha_, standardize_blocks=model.standardize_blocks)


def sqdist(X, C):
    d2 = (X ** 2).sum(1)[:, None] - 2.0 * X @ C.T + (C ** 2).sum(1)[None, :]
    return np.maximum(d2, 0.0)


def memberships(F, C, m):
    d2 = np.maximum(sqdist(F, C), 1e-12)
    inv = d2 ** (-1.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.ptp(a[ok]) < 1e-12 or np.ptp(b[ok]) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[ok], b[ok])[0])


def terciles(x, seed):
    """0/1/2 by rank with random tie-breaking (identical alphas are split arbitrarily)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    r = rankdata(x + rng.random(len(x)) * 1e-9, method="ordinal")
    return ((3 * (r - 1)) // len(x)).astype(int)


# ---------------------------------------------------------------------------
# E2: block-intervention faithfulness of alpha
# ---------------------------------------------------------------------------

def e2_stats(model, Zc, seed):
    """Returns dict of scalar metrics (already keyed by metric name; cond is added by caller)."""
    F = fused_features(model, Zc)
    nc = Zc.shape[1]
    C = model.fcm_result_.centers
    m = getattr(model.fcm_result_, "m_used", model.fcm_m)
    U = model.U_
    out = {"recon_maxabs": float(np.abs(memberships(F, C, m) - U).max())}

    has = np.linalg.norm(Zc, axis=1) > 1e-9
    out["n_with_content"] = int(has.sum())
    out["n_nodes"] = int(len(has))
    if has.sum() < 10:
        return out

    Fc0 = F.copy()
    Fc0[:, :nc] = 0.0
    Fs0 = F.copy()
    Fs0[:, nc:] = 0.0
    Uc = memberships(Fc0, C, m)
    Us = memberships(Fs0, C, m)
    top = U.argmax(1)
    tv_c = 0.5 * np.abs(U - Uc).sum(1)
    tv_s = 0.5 * np.abs(U - Us).sum(1)
    flip_c = (Uc.argmax(1) != top).astype(float)
    flip_s = (Us.argmax(1) != top).astype(float)
    delta = tv_c - tv_s

    # same intervention but replacing a block by its column mean (marginalisation) instead of zero,
    # so the perturbed features stay closer to the data distribution
    Fcm = F.copy()
    Fcm[:, :nc] = F[:, :nc].mean(axis=0, keepdims=True)
    Fsm = F.copy()
    Fsm[:, nc:] = F[:, nc:].mean(axis=0, keepdims=True)
    Ucm = memberships(Fcm, C, m)
    Usm = memberships(Fsm, C, m)
    tv_cm = 0.5 * np.abs(U - Ucm).sum(1)
    tv_sm = 0.5 * np.abs(U - Usm).sum(1)
    out["flip_content_mean"] = float((Ucm.argmax(1) != top)[has].mean())
    out["flip_structure_mean"] = float((Usm.argmax(1) != top)[has].mean())
    out["spearman_alpha_delta_mean"] = spearman(model.alpha_[has], (tv_cm - tv_sm)[has])

    # generic post-hoc block attribution (no alpha): which block supports the assignment more
    d2c = sqdist(F[:, :nc], C[:, :nc])
    d2s = sqdist(F[:, nc:], C[:, nc:])
    second = np.argsort(-U, axis=1)[:, 1]
    idx = np.arange(len(top))
    ph = (d2c[idx, second] - d2c[idx, top]) - (d2s[idx, second] - d2s[idx, top])
    # content share of squared feature norm (what the alpha weighting does mechanically)
    nsh = (F[:, :nc] ** 2).sum(1) / np.maximum((F ** 2).sum(1), 1e-12)

    alpha = model.alpha_
    out["spearman_alpha_delta"] = spearman(alpha[has], delta[has])
    out["spearman_posthoc_delta"] = spearman(ph[has], delta[has])
    out["spearman_normshare_delta"] = spearman(nsh[has], delta[has])
    out["spearman_alpha_tvc"] = spearman(alpha[has], tv_c[has])
    out["spearman_alpha_tvs"] = spearman(alpha[has], tv_s[has])
    out["mean_tv_content"] = float(tv_c[has].mean())
    out["mean_tv_structure"] = float(tv_s[has].mean())
    out["flip_content"] = float(flip_c[has].mean())
    out["flip_structure"] = float(flip_s[has].mean())
    out["alpha_std"] = float(alpha[has].std())

    t = terciles(alpha[has], seed)
    for q in range(3):
        sel = t == q
        out[f"flip_content_T{q + 1}"] = float(flip_c[has][sel].mean()) if sel.any() else float("nan")
        out[f"flip_structure_T{q + 1}"] = float(flip_s[has][sel].mean()) if sel.any() else float("nan")
        out[f"alpha_mean_T{q + 1}"] = float(alpha[has][sel].mean()) if sel.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# E1: what do the global rules / surrogates explain
# ---------------------------------------------------------------------------

def neighbor_features(G, hard, k):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    deg = np.zeros(n)
    frac = np.zeros((n, k))
    for v in nodes:
        i = pos[v]
        nb = [pos[w] for w in G.neighbors(v) if w != v]
        deg[i] = len(nb)
        if nb:
            cnt = np.bincount(hard[nb], minlength=k)[:k]
            frac[i] = cnt / cnt.sum()
    return deg, frac


def e1_rows(model, d, seed):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from nf_mcd import explain as expl

    hard = model.predict_hard()
    k = model.n_communities
    n = len(hard)
    out = {}

    rules = model.global_rules(min_support=3)
    fid = model.rule_fidelity(min_support=3)
    conf_bins = np.array([expl._fuzzy_bin(c) for c in model.confidence_])
    alpha_bins = np.array([expl._fuzzy_bin(a) for a in model.alpha_])
    keys = {(r.confidence_bin, r.alpha_bin) for r in rules}
    covered = np.array([(conf_bins[i], alpha_bins[i]) in keys for i in range(n)])
    maj_all = float(np.bincount(hard, minlength=k).max() / n)
    maj_cov = float(np.bincount(hard[covered], minlength=k).max() / covered.sum()) if covered.any() else float("nan")
    out["rules|n_rules"] = float(len(rules))
    out["rules|fidelity"] = float(fid)
    out["rules|coverage"] = float(covered.mean())
    out["rules|majority_covered"] = maj_cov
    out["rules|majority_all"] = maj_all
    out["rules|lift_pts"] = float(fid - maj_cov) if np.isfinite(fid) and np.isfinite(maj_cov) else float("nan")

    agr = np.where(np.isnan(model.agreement_), -1.0, model.agreement_)
    flags = np.array(model.modality_flags_)
    flag_oh = np.stack([(flags == f).astype(float) for f in ("both", "both_unaligned", "text_only", "image_only", "none")], axis=1)
    A = np.column_stack([model.alpha_, model.confidence_, agr, flag_oh])
    deg, frac = neighbor_features(model.G_, hard, k)
    B = np.column_stack([A, deg, frac])
    Cm = np.column_stack([deg, frac])

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for name, X in (("A", A), ("B", B), ("C", Cm)):
        for L in LEAVES:
            if np.ptp(X, axis=0).max() < 1e-12:
                fit_acc = maj_all
                cv_acc = maj_all
            else:
                clf = DecisionTreeClassifier(max_leaf_nodes=L, random_state=seed)
                clf.fit(X, hard)
                fit_acc = float((clf.predict(X) == hard).mean())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_acc = float(cross_val_score(DecisionTreeClassifier(max_leaf_nodes=L, random_state=seed),
                                                   X, hard, cv=kf).mean())
            out[f"tree_{name}_L{L}|fit_acc"] = fit_acc
            out[f"tree_{name}_L{L}|cv_acc"] = cv_acc
    out["tree|majority_all"] = maj_all
    return out


# ---------------------------------------------------------------------------
# artifacts for E3
# ---------------------------------------------------------------------------

def artifact(model):
    from nf_mcd import explain as expl
    hard = model.predict_hard()
    ab = np.array([{"Low": 0, "Medium": 1, "High": 2}[expl._fuzzy_bin(a)] for a in model.alpha_])
    agr = model.agreement_
    term = np.full(len(hard), -1)
    ok = ~np.isnan(agr)
    if ok.any():
        term[ok] = np.argmax(model.anfis.membership(agr[ok]), axis=1)
    fl = {"both": 0, "both_unaligned": 0, "text_only": 1, "image_only": 2, "none": 3}
    flag = np.array([fl[f] for f in model.modality_flags_])
    rules = {(r.confidence_bin, r.alpha_bin): (int(r.dominant_community), int(r.support))
             for r in model.global_rules(min_support=3)}
    return dict(hard=hard, alpha_bin=ab, term=term, flag=flag, rules=rules)


def art_path(dataset, variant, seed):
    return os.path.join(CACHE, f"explain_fit_{dataset}_{variant}_{seed}.pkl")


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def all_units():
    units = [(f"core|{ds}|{v}|{s}", "core", ds, v, s, None) for ds in DATASETS for v in VARIANTS for s in SEEDS]
    units += [(f"e4|{ds}|{v}|p{p}|{s}", "e4", ds, v, s, p) for ds in E4_DS for v in VARIANTS for p in E4_P for s in SEEDS]
    return units


def run_unit(unit):
    uid, kind, ds, variant, seed, p = unit
    from run_realgraph import get_dataset
    t0 = time.monotonic()
    rows = []

    def add(exp, cond, metric, value):
        rows.append([exp, ds, variant, seed, cond, metric, value])

    d = get_dataset(ds, seed)
    if kind == "core":
        model = fit_model(d, seed, variant)
        Zc = content_block(model, d)
        for metric, val in e2_stats(model, Zc, seed).items():
            add("E2", "all", metric, val)
        for key, val in e1_rows(model, d, seed).items():
            cond, metric = key.split("|")
            add("E1", cond, metric, val)
        _atomic_pickle(art_path(ds, variant, seed), artifact(model))
    else:
        from run_missing import apply_mask
        dm = apply_mask(d, "both", p, seed)
        model = fit_model(dm, seed, variant)
        Zc = content_block(model, dm)
        for metric, val in e2_stats(model, Zc, seed).items():
            add("E4", f"p{p}", metric, val)
        n = len(dm["e_t"])
        has_t = np.array([dm["e_t"][i] is not None for i in range(n)])
        has_v = np.array([dm["e_v"][i] is not None for i in range(n)])
        status = np.where(has_t & has_v, "both", np.where(has_t, "text_only", np.where(has_v, "image_only", "none")))
        obs = np.array(["both" if f == "both_unaligned" else f for f in model.modality_flags_])
        add("E4", f"p{p}", "flag_consistency", float((obs == status).mean()))
        for st in ("both", "text_only", "image_only", "none"):
            sel = status == st
            add("E4", f"p{p}|{st}", "n_nodes", int(sel.sum()))
            if sel.any():
                add("E4", f"p{p}|{st}", "mean_alpha", float(model.alpha_[sel].mean()))
                add("E4", f"p{p}|{st}", "mean_confidence", float(model.confidence_[sel].mean()))
                add("E4", f"p{p}|{st}", "frac_alpha_at_min", float((model.alpha_[sel] <= model.alpha_min + 1e-9).mean()))
                add("E4", f"p{p}|{st}", "flag_match", float((obs[sel] == st).mean()))
    add("_", uid, "_done", round(time.monotonic() - t0, 2))
    return uid, rows


def do_run(workers):
    os.makedirs(CACHE, exist_ok=True)
    done = done_units()
    todo = [u for u in all_units() if u[0] not in done]
    print(f"{len(todo)} units to run ({len(all_units()) - len(todo)} already done)", flush=True)
    write_progress("running")
    if not todo:
        return
    t0 = time.monotonic()
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_unit, u): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                uid, rows = fut.result()
                append_rows(rows)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"  FAILED {u[0]}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} units ({time.monotonic() - t0:.0f}s)", flush=True)
                write_progress("running")
    write_progress(f"run finished ({errors} errors)")
    print(f"done in {time.monotonic() - t0:.0f}s, {errors} errors", flush=True)


# ---------------------------------------------------------------------------
# E3 stability from saved artifacts
# ---------------------------------------------------------------------------

def rule_agreement(a, b):
    from scipy.optimize import linear_sum_assignment
    k = int(max(a["hard"].max(), b["hard"].max())) + 1
    M = np.zeros((k, k))
    for x, y in zip(a["hard"], b["hard"]):
        M[x, y] += 1
    r, c = linear_sum_assignment(-M)
    mapping = {int(cc): int(rr) for rr, cc in zip(r, c)}
    common = set(a["rules"]) & set(b["rules"])
    if not common:
        return float("nan"), float("nan"), 0
    same = wsame = wtot = 0.0
    for key in common:
        da, sa = a["rules"][key]
        db, sb = b["rules"][key]
        w = min(sa, sb)
        ok = da == mapping.get(db, -1)
        same += ok
        wsame += w * ok
        wtot += w
    return same / len(common), wsame / max(wtot, 1), len(common)


def do_e3():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    done = done_units()
    for ds in DATASETS:
        for v in VARIANTS:
            gid = f"e3|{ds}|{v}"
            if gid in done:
                continue
            arts = {}
            for s in SEEDS:
                pth = art_path(ds, v, s)
                if os.path.exists(pth):
                    import pickle
                    with open(pth, "rb") as f:
                        arts[s] = pickle.load(f)
            if len(arts) < 2:
                print(f"  E3 {ds}/{v}: fewer than 2 artifacts, skipped", flush=True)
                continue
            rows = []
            ss = sorted(arts)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    a, b = arts[ss[i]], arts[ss[j]]
                    cond = f"pair{ss[i]}-{ss[j]}"

                    def add(metric, val):
                        rows.append(["E3", ds, v, -1, cond, metric, val])

                    add("ari", float(adjusted_rand_score(a["hard"], b["hard"])))
                    add("nmi", float(normalized_mutual_info_score(a["hard"], b["hard"])))
                    add("same_alpha_bin", float((a["alpha_bin"] == b["alpha_bin"]).mean()))
                    add("same_term", float((a["term"] == b["term"]).mean()))
                    add("same_flag", float((a["flag"] == b["flag"]).mean()))
                    add("same_explanation", float(((a["alpha_bin"] == b["alpha_bin"]) & (a["term"] == b["term"])
                                                   & (a["flag"] == b["flag"])).mean()))
                    fr, wfr, nb = rule_agreement(a, b)
                    add("rule_same_dominant", fr)
                    add("rule_same_dominant_weighted", wfr)
                    add("rule_bins_common", float(nb))
            rows.append(["E3", ds, v, -1, gid, "_done", len(ss)])
            append_rows(rows)
            print(f"  E3 {ds}/{v}: {len(ss)} seeds", flush=True)
    write_progress("e3 done")


# ---------------------------------------------------------------------------
# summary + plots
# ---------------------------------------------------------------------------

def load_df():
    import pandas as pd
    rows = [r for r in read_rows() if r[5] != "_done"]
    df = pd.DataFrame(rows, columns=HEADER)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["seed"] = df["seed"].astype(int)
    return df


def fmt(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:6.3f}"


def do_summary():
    import pandas as pd
    df = load_df()
    if df.empty:
        print("no results yet")
        return
    L = []

    def P(s=""):
        L.append(s)

    def table(sub, index, columns, value="value", aggfunc="mean", **kw):
        t = sub.pivot_table(index=index, columns=columns, values=value, aggfunc=aggfunc)
        return t

    P("Explainability evaluation of NF-MCD (seeds 0-4, k = ground-truth count, means over seeds)")
    P("variant 'default' = NFMCD(); 'robust' = NFMCD.robust() (adaptive-m FCM on raw-PCA content)")
    P("dblp/amazon have no content: explanations degenerate to structure-only (alpha = alpha_min for all nodes).")
    P()

    # --- sanity: reconstruction of U from features/centers
    e2 = df[df.exp == "E2"]
    rc = e2[e2.metric == "recon_maxabs"].groupby(["dataset", "variant"])["value"].max()
    P("Sanity: max |U_recomputed - U| over seeds (intervention machinery reproduces the fitted memberships)")
    P(rc.to_string(float_format=lambda x: f"{x:.2e}"))
    P()

    # --- E1
    e1 = df[df.exp == "E1"]
    P("=" * 100)
    P("E1  What do the global rules explain?")
    P("=" * 100)
    P("(a) Global IF-THEN rules: fidelity vs majority baseline (over the nodes the rules cover)")
    hdr = f"{'dataset':<18}{'variant':<9}{'#rules':>7}{'coverage':>9}{'fidelity':>9}{'majority':>9}{'lift_pts':>9}"
    P(hdr)
    for ds in DATASETS:
        for v in VARIANTS:
            s = e1[(e1.dataset == ds) & (e1.variant == v) & (e1.cond == "rules")]
            g = s.groupby("metric")["value"].mean()
            if g.empty:
                continue
            P(f"{ds:<18}{v:<9}{g.get('n_rules', np.nan):7.1f}{fmt(g.get('coverage')):>9}{fmt(g.get('fidelity')):>9}"
              f"{fmt(g.get('majority_covered')):>9}{fmt(g.get('lift_pts')):>9}")
    P()
    P("(b) Post-hoc decision-tree surrogates predicting the model's hard community.")
    P("    A = [alpha, confidence, agreement, modality flag]   B = A + [degree, neighbor-community fractions]")
    P("    C = [degree, neighbor-community fractions] only.  fit = training fidelity, cv = 5-fold accuracy.")
    P("    majority = always predict the most common community.")
    for L_ in (4, 16):
        P()
        P(f"    ---- {L_} leaves ----")
        P(f"    {'dataset':<18}{'variant':<9}{'major':>7}{'A fit':>7}{'A cv':>7}{'B fit':>7}{'B cv':>7}{'C fit':>7}{'C cv':>7}")
        for ds in DATASETS:
            for v in VARIANTS:
                s = e1[(e1.dataset == ds) & (e1.variant == v)]
                if s.empty:
                    continue
                def g(name, metric):
                    x = s[(s.cond == f"tree_{name}_L{L_}") & (s.metric == metric)]["value"]
                    return x.mean() if len(x) else np.nan
                maj = s[(s.cond == "tree") & (s.metric == "majority_all")]["value"].mean()
                P(f"    {ds:<18}{v:<9}{fmt(maj):>7}{fmt(g('A', 'fit_acc')):>7}{fmt(g('A', 'cv_acc')):>7}"
                  f"{fmt(g('B', 'fit_acc')):>7}{fmt(g('B', 'cv_acc')):>7}{fmt(g('C', 'fit_acc')):>7}{fmt(g('C', 'cv_acc')):>7}")
    P()

    # --- E2
    P("=" * 100)
    P("E2  Is alpha (content-vs-structure trust) a faithful explanation? (block interventions, nodes with content)")
    P("=" * 100)
    P("Intervention: zero the content block / zero the structure block of the fused feature, recompute memberships with the")
    P("fitted centers. delta = TV shift(content removed) - TV shift(structure removed). A faithful alpha gives")
    P("Spearman(alpha, delta) > 0 and higher content-flip rate in the high-alpha tercile. 'posthoc' = label-free margin")
    P("attribution (block contribution to d2(2nd)-d2(top)); 'normshare' = content share of the squared feature norm.")
    P()
    P(f"{'dataset':<18}{'variant':<9}{'n_cont':>7}{'rho(alpha)':>11}{'rho(post)':>10}{'rho(norm)':>10}"
      f"{'flip_c':>8}{'flip_s':>8}{'tv_c':>7}{'tv_s':>7}")
    for ds in DATASETS:
        for v in VARIANTS:
            s = e2[(e2.dataset == ds) & (e2.variant == v)]
            g = s.groupby("metric")["value"].mean()
            if g.empty:
                continue
            P(f"{ds:<18}{v:<9}{g.get('n_with_content', np.nan):7.0f}{fmt(g.get('spearman_alpha_delta')):>11}"
              f"{fmt(g.get('spearman_posthoc_delta')):>10}{fmt(g.get('spearman_normshare_delta')):>10}"
              f"{fmt(g.get('flip_content')):>8}{fmt(g.get('flip_structure')):>8}"
              f"{fmt(g.get('mean_tv_content')):>7}{fmt(g.get('mean_tv_structure')):>7}")
    P()
    P("Same intervention with the block replaced by its column MEAN instead of zero (less off-distribution):")
    P(f"{'dataset':<18}{'variant':<9}{'rho(alpha)':>11}{'flip_c':>8}{'flip_s':>8}")
    for ds in DATASETS:
        for v in VARIANTS:
            s = e2[(e2.dataset == ds) & (e2.variant == v)]
            g = s.groupby("metric")["value"].mean()
            if g.empty or "flip_content_mean" not in g:
                continue
            P(f"{ds:<18}{v:<9}{fmt(g.get('spearman_alpha_delta_mean')):>11}{fmt(g.get('flip_content_mean')):>8}{fmt(g.get('flip_structure_mean')):>8}")
    P()
    P("Top-1 flip rate by alpha tercile (T1 = lowest alpha):  content removed | structure removed   (mean alpha of tercile)")
    P(f"{'dataset':<18}{'variant':<9}{'T1 c':>7}{'T2 c':>7}{'T3 c':>7}   {'T1 s':>7}{'T2 s':>7}{'T3 s':>7}   {'a1':>6}{'a2':>6}{'a3':>6}")
    for ds in DATASETS:
        for v in VARIANTS:
            s = e2[(e2.dataset == ds) & (e2.variant == v)]
            g = s.groupby("metric")["value"].mean()
            if g.empty or "flip_content_T1" not in g:
                continue
            P(f"{ds:<18}{v:<9}{fmt(g['flip_content_T1']):>7}{fmt(g['flip_content_T2']):>7}{fmt(g['flip_content_T3']):>7}   "
              f"{fmt(g['flip_structure_T1']):>7}{fmt(g['flip_structure_T2']):>7}{fmt(g['flip_structure_T3']):>7}   "
              f"{fmt(g['alpha_mean_T1']):>6}{fmt(g['alpha_mean_T2']):>6}{fmt(g['alpha_mean_T3']):>6}")
    P()

    # --- E3
    e3 = df[df.exp == "E3"]
    P("=" * 100)
    P("E3  Stability across seeds (mean over all seed pairs)")
    P("=" * 100)
    P(f"{'dataset':<18}{'variant':<9}{'ARI':>7}{'NMI':>7}{'alphabin':>9}{'ANFISterm':>10}{'flag':>7}{'all3':>7}{'rule_same':>10}{'rule_w':>8}{'#bins':>7}")
    for ds in DATASETS:
        for v in VARIANTS:
            s = e3[(e3.dataset == ds) & (e3.variant == v)]
            g = s.groupby("metric")["value"].mean()
            if g.empty:
                continue
            P(f"{ds:<18}{v:<9}{fmt(g.get('ari')):>7}{fmt(g.get('nmi')):>7}{fmt(g.get('same_alpha_bin')):>9}"
              f"{fmt(g.get('same_term')):>10}{fmt(g.get('same_flag')):>7}{fmt(g.get('same_explanation')):>7}"
              f"{fmt(g.get('rule_same_dominant')):>10}{fmt(g.get('rule_same_dominant_weighted')):>8}{g.get('rule_bins_common', np.nan):7.1f}")
    P()

    # --- E4
    e4 = df[df.exp == "E4"]
    P("=" * 100)
    P("E4  Explanations under injected missing modalities (mode 'both', nested MCAR masks, seeds 0-4)")
    P("=" * 100)
    P("Flag consistency = fraction of nodes whose reported modality flag equals the modality actually left after masking.")
    P(f"{'dataset':<18}{'variant':<9}{'p':>5}{'flag_ok':>9}   mean alpha by true status: {'both':>6}{'text':>7}{'image':>7}{'none':>7}   {'rho(alpha)':>10}{'flip_c':>8}{'flip_s':>8}")
    for ds in E4_DS:
        for v in VARIANTS:
            for p in E4_P:
                s = e4[(e4.dataset == ds) & (e4.variant == v)]
                base = s[s.cond == f"p{p}"].groupby("metric")["value"].mean()
                def ma(st):
                    x = s[(s.cond == f"p{p}|{st}") & (s.metric == "mean_alpha")]["value"]
                    return x.mean() if len(x) else np.nan
                P(f"{ds:<18}{v:<9}{p:5.1f}{fmt(base.get('flag_consistency')):>9}   {'':<28}{fmt(ma('both')):>6}{fmt(ma('text_only')):>7}"
                  f"{fmt(ma('image_only')):>7}{fmt(ma('none')):>7}   {fmt(base.get('spearman_alpha_delta')):>10}"
                  f"{fmt(base.get('flip_content')):>8}{fmt(base.get('flip_structure')):>8}")
    P()
    P("Fraction of nodes with alpha exactly at alpha_min, by true status (default variant):")
    for ds in E4_DS:
        for p in E4_P:
            s = e4[(e4.dataset == ds) & (e4.variant == "default")]
            row = []
            for st in ("both", "text_only", "image_only", "none"):
                x = s[(s.cond == f"p{p}|{st}") & (s.metric == "frac_alpha_at_min")]["value"]
                row.append(f"{st}={x.mean():.2f}" if len(x) else f"{st}=n/a")
            P(f"  {ds:<18} p={p}: " + "  ".join(row))

    text = "\n".join(L) + "\n"
    _atomic_text(LOG, text)
    print(text)
    do_plots(df)


def do_plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e1 = df[df.exp == "E1"]
    dss = ["crisismmd", "fakeddit", "fakeddit_real_lcc", "pheme"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharey=False)
    styles = {"A": ("tab:blue", "alpha/confidence/flag only (A)"),
              "B": ("tab:green", "A + structure (B)"),
              "C": ("tab:orange", "structure only (C)")}
    for r, v in enumerate(VARIANTS):
        for c, ds in enumerate(dss):
            ax = axes[r, c]
            s = e1[(e1.dataset == ds) & (e1.variant == v)]
            for name, (col, lab) in styles.items():
                ys = [s[(s.cond == f"tree_{name}_L{L}") & (s.metric == "fit_acc")]["value"].mean() for L in LEAVES]
                ax.plot(LEAVES, ys, "-o", ms=3, color=col, label=lab)
            maj = s[(s.cond == "tree") & (s.metric == "majority_all")]["value"].mean()
            ax.axhline(maj, color="gray", ls="--", lw=1, label="majority class")
            nr = s[(s.cond == "rules") & (s.metric == "n_rules")]["value"].mean()
            fd = s[(s.cond == "rules") & (s.metric == "fidelity")]["value"].mean()
            ax.plot([nr], [fd], "r*", ms=13, label="NF-MCD global rules")
            ax.set_title(f"{ds} ({v})", fontsize=10)
            ax.set_xscale("log")
            ax.set_xlabel("number of rules / leaves")
            if c == 0:
                ax.set_ylabel("fidelity to model's hard community")
            ax.set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("E1: how well can rules explain the model's community assignment?")
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "explain_fidelity_curve.png"), dpi=140)
    plt.close(fig)

    e2 = df[df.exp == "E2"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    labs, w = [], 0.26
    cats = [(ds, v) for ds in dss for v in VARIANTS]
    x = np.arange(len(cats))
    for j, (met, col, lab) in enumerate([("spearman_alpha_delta", "tab:blue", "alpha"),
                                         ("spearman_posthoc_delta", "tab:orange", "post-hoc margin"),
                                         ("spearman_normshare_delta", "tab:gray", "norm share")]):
        ys = [e2[(e2.dataset == ds) & (e2.variant == v) & (e2.metric == met)]["value"].mean() for ds, v in cats]
        ax.bar(x + (j - 1) * w, np.nan_to_num(ys), w, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n{v}" for ds, v in cats], fontsize=7, rotation=0)
    ax.set_ylabel("Spearman with intervention effect (content - structure)")
    ax.set_title("E2: does alpha predict which block the assignment depends on?")
    ax.legend(fontsize=8)
    ax = axes[1]
    for ds in dss:
        for v, ls in (("default", "-"), ("robust", "--")):
            s = e2[(e2.dataset == ds) & (e2.variant == v)]
            ys = [s[s.metric == f"flip_content_T{q}"]["value"].mean() for q in (1, 2, 3)]
            ax.plot([1, 2, 3], ys, ls, marker="o", label=f"{ds} {v}")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["low alpha", "mid alpha", "high alpha"])
    ax.set_ylabel("top-1 flip rate when content removed")
    ax.set_title("Content sensitivity by alpha tercile")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "explain_alpha_faithfulness.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "e3", "summary", "all"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if a.cmd in ("run", "all"):
        do_run(a.workers)
    if a.cmd in ("e3", "all"):
        do_e3()
    if a.cmd in ("summary", "all"):
        do_summary()


if __name__ == "__main__":
    main()
