"""
Evaluation of the neighbourhood-based explanations (nf_mcd.explain additions).

  (a) fidelity/coverage/compactness of the neighbourhood rules vs the old
      (confidence, alpha) rules vs a depth-limited decision tree on neighbour
      fractions vs the majority baseline, plus a fidelity-vs-#rules curve
  (b) faithfulness of content_sensitivity(): per-node flip indicator between the
      full model and an actual content-less refit (Hungarian-aligned), AUC/Spearman
      of content_sensitivity vs alpha vs content share of the feature norm, and
      the seed-to-seed flip rate as a noise floor
  (c) stability across seeds of the new vs old explanations
  (d) missing-modality consistency (mask 'both', p in {0.2, 0.6})
  (e) rendered example explanations

    py run_nbr_explain.py run --workers 4    # core + e4 units (resumable)
    py run_nbr_explain.py stability          # (c) from saved artifacts
    py run_nbr_explain.py examples           # (e)
    py run_nbr_explain.py summary            # tables + plots

These explanations mostly restate homophily on structure-dominated assignments and
say nothing about whether the model is right; no human study was done.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import pickle
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
CSV = os.path.join(EXP, "nbr_explain_results.csv")
PROG = os.path.join(EXP, "nbr_explain_progress.md")
LOG = os.path.join(EXP, "nbr_explain_summary.log")
EXAMPLES = os.path.join(EXP, "nbr_explain_examples.txt")

DATASETS = ["crisismmd", "fakeddit", "fakeddit_real_lcc", "pheme", "dblp", "amazon"]
VARIANTS = ["default", "robust"]
SEEDS = [0, 1, 2, 3, 4]
E4_DS = ["crisismmd", "fakeddit_real_lcc"]
E4_P = [0.2, 0.6]
E4_SEEDS = [0, 1, 2]
LEAVES = [2, 3, 4, 6, 8, 12, 16, 24]
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


def all_units():
    u = [(f"core|{ds}|{v}|{s}", "core", ds, v, s, None) for ds in DATASETS for v in VARIANTS for s in SEEDS]
    u += [(f"e4|{ds}|{v}|p{p}|{s}", "e4", ds, v, s, p) for ds in E4_DS for v in VARIANTS for p in E4_P for s in E4_SEEDS]
    return u


def write_progress(note=""):
    units = all_units()
    done = done_units()
    n_done = sum(1 for x in units if x[0] in done)
    n_stab = sum(1 for r in read_rows() if r[0] == "C" and r[5] == "_done")
    _atomic_text(PROG, (
        "# Neighbourhood-explanation evaluation progress\n\n"
        f"- Units done: **{n_done}/{len(units)}** (core fits + missing-modality units)\n"
        f"- Stability groups done: **{n_stab}/{len(DATASETS) * len(VARIANTS)}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished units are skipped automatically)\n\n"
        "```\ncd C:\\Users\\HP\\Projects\\Paper_SSC\\nfmcd_impl\n"
        "py run_nbr_explain.py run --workers 4\npy run_nbr_explain.py stability\n"
        "py run_nbr_explain.py examples\npy run_nbr_explain.py summary\n```\n\n"
        "Results: experiments/nbr_explain_results.csv (append-only, fsynced). Summary: experiments/nbr_explain_summary.log.\n"
    ))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def align_labels(ref, other, k):
    """Map `other` community ids onto `ref` ids (Hungarian on the confusion matrix).
    Returns (aligned other labels, mapping other->ref)."""
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((k, k))
    for x, y in zip(ref, other):
        M[int(x), int(y)] += 1
    r, c = linear_sum_assignment(-M)
    mapping = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mapping.get(int(y), -1) for y in other]), mapping


def auc(score, target):
    from sklearn.metrics import roc_auc_score
    t = np.asarray(target).astype(int)
    if t.min() == t.max() or np.ptp(score) < 1e-12:
        return float("nan")
    return float(roc_auc_score(t, score))


def fit_variant(d, seed, variant):
    from run_explain import fit_model
    return fit_model(d, seed, variant)


def refit_without_content(d, seed, variant):
    """Structure-only refit with the same variant; falls back to the default config if
    the variant cannot run with no content at all. Returns (model, used_fallback)."""
    n = d["G"].number_of_nodes()
    d0 = dict(d)
    d0["e_t"] = [None] * n
    d0["e_v"] = [None] * n
    try:
        return fit_variant(d0, seed, variant), 0
    except Exception:  # noqa: BLE001
        return fit_variant(d0, seed, "default"), 1


def tree_curve(model, hard, k, seed):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from run_explain import neighbor_features
    deg, frac = neighbor_features(model.G_, hard, k)
    X = np.hstack([deg[:, None], frac])
    out = {}
    cv = KFold(5, shuffle=True, random_state=seed)
    for L in LEAVES:
        clf = DecisionTreeClassifier(max_leaf_nodes=L, random_state=seed)
        try:
            cvacc = float(cross_val_score(clf, X, hard, cv=cv).mean())
        except ValueError:
            cvacc = float("nan")
        clf.fit(X, hard)
        out[L] = (float(clf.score(X, hard)), cvacc)
    return out


# ---------------------------------------------------------------------------
# core unit: (a), (b) rows and the (c) artifact
# ---------------------------------------------------------------------------

def run_core(ds, variant, seed, add):
    from nf_mcd import explain as ex
    from run_explain import artifact, content_block, fused_features, spearman
    from run_realgraph import get_dataset

    d = get_dataset(ds, seed)
    model = fit_variant(d, seed, variant)
    k = d["n_communities"]
    hard = model.predict_hard()
    deg, counts, wcounts = model._neighbour_shares()
    sens = model._block_sensitivity()
    rules = model.global_neighbourhood_rules()
    fid = model.neighbourhood_rule_fidelity()

    # (a) rules
    for key, val in fid.items():
        add("A", "nbr", key, val)
    add("A", "old", "fidelity", float(model.rule_fidelity()))
    add("A", "old", "n_rules", float(len(model.global_rules(min_support=3))))
    for r in range(1, len(rules) + 1):
        f = ex.neighbourhood_rule_fidelity(rules[:r], deg, counts, hard)
        add("A", f"curve|r{r}", "fidelity", f["fidelity"])
        add("A", f"curve|r{r}", "coverage", f["coverage"])
    for L, (tr, cv) in tree_curve(model, hard, k, seed).items():
        add("A", f"tree|L{L}", "train", tr)
        add("A", f"tree|L{L}", "cv", cv)

    # (b) faithfulness of content_sensitivity against an actual content-less refit
    flags = model.modality_flags_
    has = np.array([f != "none" for f in flags])
    add("B", "all", "n_with_content", int(has.sum()))
    add("B", "all", "n_nodes", int(len(has)))
    art = artifact(model)
    if has.sum() >= 10:
        F = model.fused_features_
        nc = model.n_content_cols_
        normshare = (F[:, :nc] ** 2).sum(1) / np.maximum((F ** 2).sum(1), 1e-12)
        m0, fb = refit_without_content(d, seed, variant)
        a0, _ = align_labels(hard, m0.predict_hard(), k)
        flip = (a0 != hard)
        m1 = fit_variant(d, seed + 1, variant)
        a1, _ = align_labels(hard, m1.predict_hard(), k)
        flip_noise = (a1 != hard)
        add("B", "all", "refit_fallback", fb)
        add("B", "all", "flip_rate_refit", float(flip[has].mean()))
        add("B", "all", "flip_rate_seed_noise", float(flip_noise[has].mean()))
        tv_c = sens["tv_content"]
        for name, score in (("content_sensitivity", tv_c), ("content_flip_margin", sens["margin_no_content"]),
                            ("alpha", model.alpha_), ("normshare", normshare),
                            ("structure_sensitivity_neg", -sens["tv_structure"])):
            add("B", name, "auc_flip", auc(score[has], flip[has]))
            add("B", name, "spearman_flip", spearman(score[has], flip[has].astype(float)))
        # does the intervention's own flip indicator predict the refit flip?
        pf = sens["flip_content"][has]
        tf = flip[has]
        add("B", "intervention_flip", "agreement", float((pf == tf).mean()))
        add("B", "intervention_flip", "precision", float(tf[pf].mean()) if pf.any() else float("nan"))
        add("B", "intervention_flip", "recall", float(pf[tf].mean()) if tf.any() else float("nan"))
        add("B", "all", "mean_content_sensitivity", float(tv_c[has].mean()))
        add("B", "all", "mean_structure_sensitivity", float(sens["tv_structure"][has].mean()))
    else:
        add("B", "all", "no_content", 1)

    # (c) artifact
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(deg > 0, counts[np.arange(len(hard)), hard] / np.maximum(deg, 1), np.nan)
    share_bin = np.where(deg == 0, -1, np.where(share < 1 / 3, 0, np.where(share < 2 / 3, 1, 2)))
    agrees = np.where(has, (sens["content_only_top"] == hard).astype(int), -1)
    art.update(
        share_bin=share_bin, agrees=agrees, flip_content=sens["flip_content"].astype(int),
        rule_pred=ex.apply_neighbourhood_rules(rules, deg, counts),
        rule_thr={r.community: r.threshold for r in rules},
    )
    _atomic_pickle(art_path(ds, variant, seed), art)


def art_path(ds, variant, seed):
    return os.path.join(CACHE, f"nbr_fit_{ds}_{variant}_{seed}.pkl")


# ---------------------------------------------------------------------------
# e4 unit: missing modalities
# ---------------------------------------------------------------------------

PHRASE = {
    "none": "No content (text or image) was available",
    "text_only": "The image was unavailable",
    "image_only": "The text was unavailable",
}


def run_e4(ds, variant, seed, p, add):
    from run_missing import apply_mask
    from run_realgraph import get_dataset

    d = get_dataset(ds, seed)
    dm = apply_mask(d, "both", p, seed)
    model = fit_variant(dm, seed, variant)
    n = len(dm["e_t"])
    has_t = np.array([dm["e_t"][i] is not None for i in range(n)])
    has_v = np.array([dm["e_v"][i] is not None for i in range(n)])
    status = np.where(has_t & has_v, "both", np.where(has_t, "text_only", np.where(has_v, "image_only", "none")))
    sens = model._block_sensitivity()
    deg, counts, wcounts = model._neighbour_shares()
    nodes = model.nodes_
    ok = {s: 0 for s in ("both", "text_only", "image_only", "none")}
    tot = {s: 0 for s in ok}
    for i, v in enumerate(nodes):
        e = model.explain_node_neighbourhood(v)
        st = status[i]
        tot[st] += 1
        txt = e.text
        if st == "both":
            good = (("Content supports" in txt) or ("Content contradicts" in txt)) and not any(
                ph in txt for ph in PHRASE.values())
        elif st == "none":
            good = PHRASE["none"] in txt and "Content supports" not in txt and "Content contradicts" not in txt
        else:
            good = PHRASE[st] in txt and (("Content supports" in txt) or ("Content contradicts" in txt))
        ok[st] += int(good)
    cond = f"p{p}"
    for st in ok:
        add("D", f"{cond}|{st}", "n_nodes", tot[st])
        if tot[st]:
            sel = status == st
            add("D", f"{cond}|{st}", "text_consistency", ok[st] / tot[st])
            add("D", f"{cond}|{st}", "mean_content_sensitivity", float(sens["tv_content"][sel].mean()))
            add("D", f"{cond}|{st}", "mean_structure_sensitivity", float(sens["tv_structure"][sel].mean()))
            add("D", f"{cond}|{st}", "flip_content", float(sens["flip_content"][sel].mean()))
            add("D", f"{cond}|{st}", "content_sens_zero", float((sens["tv_content"][sel] <= 1e-9).mean()))
    add("D", cond, "text_consistency_all", float(sum(ok.values()) / max(sum(tot.values()), 1)))


def run_unit(unit):
    uid, kind, ds, variant, seed, p = unit
    t0 = time.monotonic()
    rows = []

    def add(exp, cond, metric, value):
        rows.append([exp, ds, variant, seed, cond, metric, value])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "core":
            run_core(ds, variant, seed, add)
        else:
            run_e4(ds, variant, seed, p, add)
    rows.append(["_", ds, variant, seed, uid, "_done", round(time.monotonic() - t0, 2)])
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
# (c) stability
# ---------------------------------------------------------------------------

def do_stability():
    from sklearn.metrics import adjusted_rand_score
    from run_explain import rule_agreement
    done = done_units()
    for ds in DATASETS:
        for v in VARIANTS:
            gid = f"c|{ds}|{v}"
            if gid in done:
                continue
            arts = {}
            for s in SEEDS:
                pth = art_path(ds, v, s)
                if os.path.exists(pth):
                    with open(pth, "rb") as f:
                        arts[s] = pickle.load(f)
            if len(arts) < 2:
                print(f"  C {ds}/{v}: fewer than 2 artifacts, skipped", flush=True)
                continue
            k = int(max(a["hard"].max() for a in arts.values())) + 1
            rows = []
            ss = sorted(arts)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    a, b = arts[ss[i]], arts[ss[j]]
                    cond = f"pair{ss[i]}-{ss[j]}"

                    def add(metric, val):
                        rows.append(["C", ds, v, -1, cond, metric, val])

                    add("ari", float(adjusted_rand_score(a["hard"], b["hard"])))
                    old = (a["alpha_bin"] == b["alpha_bin"]) & (a["term"] == b["term"]) & (a["flag"] == b["flag"])
                    add("old_same_type", float(old.mean()))
                    same_bin = a["share_bin"] == b["share_bin"]
                    same_ag = a["agrees"] == b["agrees"]
                    same_fl = a["flip_content"] == b["flip_content"]
                    add("new_same_share_bin", float(same_bin.mean()))
                    add("new_same_content_agrees", float(same_ag.mean()))
                    add("new_same_flip_content", float(same_fl.mean()))
                    add("new_same_type", float((same_bin & same_ag & same_fl).mean()))
                    add("new_same_type_structure", float((same_bin & same_ag).mean()))
                    _, mapping = align_labels(a["hard"], b["hard"], k)
                    pb = np.array([mapping.get(int(x), -1) if x >= 0 else -1 for x in b["rule_pred"]])
                    add("rule_pred_agreement", float((pb == a["rule_pred"]).mean()))
                    both = (pb >= 0) & (a["rule_pred"] >= 0)
                    add("rule_pred_agreement_both_covered", float((pb[both] == a["rule_pred"][both]).mean()) if both.any() else float("nan"))
                    diffs = [abs(a["rule_thr"][mapping[cb]] - tb) for cb, tb in b["rule_thr"].items()
                             if cb in mapping and mapping[cb] in a["rule_thr"]]
                    add("rule_threshold_absdiff", float(np.mean(diffs)) if diffs else float("nan"))
                    fr, wfr, nb = rule_agreement(a, b)
                    add("old_rule_same_dominant", fr)
            rows.append(["C", ds, v, -1, gid, "_done", len(ss)])
            append_rows(rows)
            print(f"  C {ds}/{v}: {len(ss)} seeds", flush=True)
    write_progress("stability done")


# ---------------------------------------------------------------------------
# (e) examples
# ---------------------------------------------------------------------------

def do_examples():
    from run_missing import apply_mask
    from run_realgraph import get_dataset
    lines = [
        "Rendered neighbourhood explanations (nf_mcd.explain.explain_node_neighbourhood).",
        "They restate the neighbourhood and report measured content/structure sensitivity.",
        "On structure-dominated assignments they mostly restate homophily; they do not say the model is right.\n",
    ]

    def pick(model, cond):
        for i, v in enumerate(model.nodes_):
            e = model.explain_node_neighbourhood(v)
            if cond(e):
                return e
        return None

    plans = [
        ("crisismmd", None, "default", "consistent: content agrees, strong neighbourhood",
         lambda e: e.content_agrees is True and e.share_top >= 0.7 and e.n_neighbours >= 6),
        ("crisismmd", None, "default", "content contradicts the neighbourhood",
         lambda e: e.content_agrees is False and e.share_top >= 0.5 and e.n_neighbours >= 6),
        ("crisismmd", None, "robust", "assignment flips if content is removed",
         lambda e: e.flips_without_content and e.n_neighbours >= 6),
        ("fakeddit_real_lcc", None, "default", "real-relations graph, weak neighbourhood (low lift)",
         lambda e: e.n_neighbours >= 4 and e.lift < 1.5),
        ("pheme", None, "default", "isolated node (no neighbours)",
         lambda e: e.n_neighbours == 0),
        ("pheme", None, "default", "text-only dataset, ordinary node",
         lambda e: e.n_neighbours >= 3 and e.share_top >= 0.6),
        ("crisismmd", 0.6, "default", "missing modality (60% of modalities removed): text only",
         lambda e: e.modality_flag == "text_only" and e.n_neighbours >= 4),
        ("crisismmd", 0.6, "default", "missing modality: no content at all",
         lambda e: e.modality_flag == "none" and e.n_neighbours >= 4),
    ]
    cache = {}
    for ds, p, variant, title, cond in plans:
        key = (ds, p, variant)
        if key not in cache:
            d = get_dataset(ds, 0)
            if p is not None:
                d = apply_mask(d, "both", p, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache[key] = fit_variant(d, 0, variant)
        e = pick(cache[key], cond)
        lines.append(f"[{ds}{'' if p is None else f', both modalities removed with p={p}'}; {variant}; seed 0] {title}")
        lines.append("  " + (e.text if e else "(no node matched this case)"))
        lines.append("")
    _atomic_text(EXAMPLES, "\n".join(lines))
    print(f"wrote {EXAMPLES}", flush=True)


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


def f3(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:6.3f}"


def do_summary():
    import pandas as pd
    df = load_df()
    if df.empty:
        print("no results yet")
        return
    out = []
    P = out.append

    def mean_of(exp, cond, metric, ds, v):
        s = df[(df.exp == exp) & (df.cond == cond) & (df.metric == metric) & (df.dataset == ds) & (df.variant == v)]
        return float(s["value"].mean()) if len(s) else float("nan")

    P("Neighbourhood-based explanations: evaluation (mean over seeds; k = ground-truth count)")
    P("=" * 96)
    P("\n(a) Global rules: fidelity to the model's hard assignments")
    P("    new = neighbourhood rules; strict fidelity counts no-rule nodes as misses; old = (confidence, alpha) rules;")
    P("    tree = depth-limited decision tree on [degree, neighbour fractions], 5-fold CV accuracy at 6 and 12 leaves")
    hd = f"{'dataset':<18}{'variant':<9}{'majority':>9}{'old_fid':>8}{'new_fid':>8}{'cover':>7}{'prec':>7}{'w/fallb':>8}{'#rules':>7}{'tree6cv':>8}{'tree12cv':>9}"
    P(hd)
    P("-" * len(hd))
    for ds in DATASETS:
        for v in VARIANTS:
            g = lambda cond, m: mean_of("A", cond, m, ds, v)
            P(f"{ds:<18}{v:<9}{f3(g('nbr','majority')):>9}{f3(g('old','fidelity')):>8}{f3(g('nbr','fidelity')):>8}"
              f"{f3(g('nbr','coverage')):>7}{f3(g('nbr','precision_covered')):>7}{f3(g('nbr','fidelity_with_fallback')):>8}"
              f"{f3(g('nbr','n_rules')):>7}{f3(g('tree|L6','cv')):>8}{f3(g('tree|L12','cv')):>9}")
    P("\n    fidelity vs number of neighbourhood rules (rules ranked by support; default variant)")
    hd = f"{'dataset':<18}" + "".join(f"{'r='+str(r):>7}" for r in range(1, 11))
    P(hd)
    for ds in DATASETS:
        P(f"{ds:<18}" + "".join(f"{f3(mean_of('A', f'curve|r{r}', 'fidelity', ds, 'default')):>7}" for r in range(1, 11)))

    P("\n(b) Faithfulness of content_sensitivity(): does it predict which nodes change when the model is actually refit WITHOUT content?")
    P("    flip = node's aligned community differs between the full model and the content-less refit;")
    P("    noise = same comparison between two seeds of the full model (what flips anyway). AUC 0.5 = chance.")
    P("    scores compared: margin = content_flip_margin (best rival minus own community after removing content),")
    P("    sens = content_sensitivity (total-variation shift), alpha, norm = content share of the feature norm;")
    P("    iv = the intervention's own flip indicator (precision/recall against the refit flip).")
    hd = f"{'dataset':<18}{'variant':<9}{'flip':>7}{'noise':>7}{'AUC_margin':>11}{'AUC_sens':>9}{'AUC_alpha':>10}{'AUC_norm':>9}{'rho_margin':>11}{'iv_prec':>8}{'iv_rec':>7}"
    P(hd)
    P("-" * len(hd))
    for ds in DATASETS:
        for v in VARIANTS:
            g = lambda cond, m: mean_of("B", cond, m, ds, v)
            if np.isnan(g("all", "flip_rate_refit")):
                P(f"{ds:<18}{v:<9}  (no content in this dataset: sensitivity is 0 by construction)")
                continue
            P(f"{ds:<18}{v:<9}{f3(g('all','flip_rate_refit')):>7}{f3(g('all','flip_rate_seed_noise')):>7}"
              f"{f3(g('content_flip_margin','auc_flip')):>11}{f3(g('content_sensitivity','auc_flip')):>9}"
              f"{f3(g('alpha','auc_flip')):>10}{f3(g('normshare','auc_flip')):>9}"
              f"{f3(g('content_flip_margin','spearman_flip')):>11}"
              f"{f3(g('intervention_flip','precision')):>8}{f3(g('intervention_flip','recall')):>7}")
    fb = df[(df.exp == "B") & (df.metric == "refit_fallback")]
    if len(fb) and fb["value"].sum() > 0:
        P(f"\n    note: {int(fb['value'].sum())} content-less refits fell back to the default config (robust cannot run with no content).")

    P("\n(c) Stability across seeds 0-4 (mean over seed pairs): same node, same explanation?")
    hd = f"{'dataset':<18}{'variant':<9}{'ARI':>7}{'old_type':>9}{'new_share':>10}{'new_agree':>10}{'new_flip':>9}{'new_type':>9}{'rule_pred':>10}{'thr_diff':>9}{'old_rule':>9}"
    P(hd)
    P("-" * len(hd))
    for ds in DATASETS:
        for v in VARIANTS:
            s = df[(df.exp == "C") & (df.dataset == ds) & (df.variant == v)]
            if s.empty:
                continue
            mm = lambda m: float(s[s.metric == m]["value"].mean())
            P(f"{ds:<18}{v:<9}{f3(mm('ari')):>7}{f3(mm('old_same_type')):>9}{f3(mm('new_same_share_bin')):>10}"
              f"{f3(mm('new_same_content_agrees')):>10}{f3(mm('new_same_flip_content')):>9}{f3(mm('new_same_type')):>9}"
              f"{f3(mm('rule_pred_agreement')):>10}{f3(mm('rule_threshold_absdiff')):>9}{f3(mm('old_rule_same_dominant')):>9}")

    P("\n(d) Missing modalities ('both' mode): is the explanation text consistent with what was removed, and does sensitivity respond?")
    hd = f"{'dataset':<18}{'variant':<9}{'p':>5}{'status':>11}{'n':>7}{'text_ok':>8}{'content_sens':>13}{'struct_sens':>12}{'sens=0':>8}"
    P(hd)
    P("-" * len(hd))
    for ds in E4_DS:
        for v in VARIANTS:
            for p in E4_P:
                for st in ("both", "text_only", "image_only", "none"):
                    c = f"p{p}|{st}"
                    s = df[(df.exp == "D") & (df.dataset == ds) & (df.variant == v) & (df.cond == c)]
                    if s.empty:
                        continue
                    mm = lambda m: float(s[s.metric == m]["value"].mean())
                    P(f"{ds:<18}{v:<9}{p:>5}{st:>11}{mm('n_nodes'):>7.0f}{f3(mm('text_consistency')):>8}"
                      f"{f3(mm('mean_content_sensitivity')):>13}{f3(mm('mean_structure_sensitivity')):>12}{f3(mm('content_sens_zero')):>8}")

    P("\nWhat these explanations do not claim:")
    P("  - On structure-dominated assignments (all datasets with informative graphs, and especially the leaky CrisisMMD/Fakeddit graphs)")
    P("    the neighbourhood rule mostly restates homophily: high fidelity here reflects that the clustering is structure-driven,")
    P("    not that the assignment is correct.")
    P("  - Neighbour shares are computed from the model's own assignments (self-referential).")
    P("  - Zeroing a feature block is off-distribution. content_sensitivity is an intervention measure, not a causal effect on the truth.")
    P("  - No human study: usefulness to a person is untested. 5 seeds (3 for missing-modality).")
    _atomic_text(LOG, "\n".join(out) + "\n")
    print("\n".join(out))
    do_plots(df)


def do_plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ci, ds in enumerate(DATASETS):
        for ax, v in zip(axes, VARIANTS):
            xs, ys = [], []
            for r in range(1, 13):
                s = df[(df.exp == "A") & (df.dataset == ds) & (df.variant == v) & (df.cond == f"curve|r{r}") & (df.metric == "fidelity")]
                if len(s):
                    xs.append(r)
                    ys.append(s["value"].mean())
            ax.plot(xs, ys, marker="o", color=cmap(ci), label=ds)
            old = df[(df.exp == "A") & (df.dataset == ds) & (df.variant == v) & (df.cond == "old") & (df.metric == "fidelity")]["value"].mean()
            ax.scatter([xs[-1] if xs else 1], [old], marker="x", s=60, color=cmap(ci))
    for ax, v in zip(axes, VARIANTS):
        ax.set_title(f"{v}: fidelity vs # neighbourhood rules (x = old (confidence, alpha) rules)")
        ax.set_xlabel("number of rules")
        ax.set_ylabel("fidelity to model assignment")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP, "nbr_explain_fidelity_curve.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ds_c = [ds for ds in DATASETS if not df[(df.exp == "B") & (df.dataset == ds) & (df.metric == "flip_rate_refit")].empty]
    for ax, v in zip(axes, VARIANTS):
        w = 0.26
        w = 0.2
        for oi, (name, lab) in enumerate((("content_flip_margin", "content_flip_margin"), ("content_sensitivity", "content_sensitivity (TV)"),
                                          ("alpha", "alpha"), ("normshare", "content share of norm"))):
            vals = [df[(df.exp == "B") & (df.dataset == ds) & (df.variant == v) & (df.cond == name) & (df.metric == "auc_flip")]["value"].mean() for ds in ds_c]
            ax.bar(np.arange(len(ds_c)) + (oi - 1.5) * w, vals, w, label=lab)
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(ds_c)))
        ax.set_xticklabels(ds_c, rotation=15, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC for predicting flips on content-less refit")
        ax.set_title(f"{v}: which score predicts flips?")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP, "nbr_explain_sensitivity.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "stability", "examples", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if a.cmd == "run":
        do_run(a.workers)
    elif a.cmd == "stability":
        do_stability()
    elif a.cmd == "examples":
        do_examples()
    else:
        do_summary()


if __name__ == "__main__":
    main()
