"""Diagnosis only (no library changes): why does full NF-MCD underperform
text-only / image-only / structure-only on Fakeddit, and fixed alpha on CrisisMMD?

Crash-safe: every result row is appended to experiments/fusion_diagnosis.csv
with flush+fsync; units already in the CSV are skipped on rerun. Plots are
written last via temp+rename.

Run:  py diagnose_fusion.py [fakeddit crisismmd]
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
import time
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
CSV_PATH = os.path.join(EXP, "fusion_diagnosis.csv")
LOG_PATH = os.path.join(EXP, "fusion_diagnosis.log")

from nf_mcd import NFMCD  # noqa: E402
from nf_mcd import community_detection as cd  # noqa: E402
from nf_mcd import metrics as mx  # noqa: E402
from nf_mcd import topology as topo  # noqa: E402
from nf_mcd.fuzzy_fusion import ANFISAgreement, NeuroFuzzyFusion  # noqa: E402

DATASETS = [a for a in sys.argv[1:]] or ["fakeddit", "crisismmd"]
SEEDS = (0, 1, 2)
COLS = ["diagnostic", "dataset", "seed", "variant", "metric", "value"]


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done():
    done, rows = set(), []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["diagnostic"], r["dataset"], r["seed"], r["variant"]))
                rows.append(r)
    return done, rows


DONE, ROWS = load_done()


def emit(diag, ds, seed, variant, **metrics):
    new = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        for k, v in metrics.items():
            w.writerow([diag, ds, seed, variant, k, f"{float(v):.6f}"])
            ROWS.append(dict(diagnostic=diag, dataset=ds, seed=str(seed), variant=variant,
                             metric=k, value=f"{float(v):.6f}"))
        f.flush()
        os.fsync(f.fileno())
    DONE.add((diag, ds, str(seed), variant))


def unit_done(diag, ds, seed, variant):
    return (diag, ds, str(seed), variant) in DONE


def load_cache(name):
    with open(os.path.join(CACHE, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Helpers mirroring NFMCD.fit stages 3-4 / evaluate, so a fusion result can be
# reused across alpha values (fusion does not depend on alpha).
# ----------------------------------------------------------------------------

def views_from_U(U, threshold=0.2, subset=None):
    node_view = cd.overlapping_communities(U, threshold)
    k = U.shape[1]
    view = [set() for _ in range(k)]
    for i, comms in enumerate(node_view):
        if subset is not None and i not in subset:
            continue
        for c in comms:
            view[c].add(i)
    return view


def views_from_labels(labels, k, subset=None):
    view = [set() for _ in range(k)]
    for i, c in enumerate(labels):
        if subset is not None and i not in subset:
            continue
        view[int(c)].add(i)
    return view


def true_view(true, k, subset=None):
    view = [set() for _ in range(k)]
    for i, comms in enumerate(true):
        if subset is not None and i not in subset:
            continue
        for c in comms:
            if 0 <= c < k:
                view[c].add(i)
    return view


def onmi(pred_view, tview, n):
    return mx.overlapping_nmi(pred_view, tview, n)


def cluster_from(d, fusion_result, seed, amin=0.15, amax=0.85, m=1.5, alpha=None):
    k = d["n_communities"]
    G = d["G"]
    conf = fusion_result.confidence
    a = np.full(len(conf), alpha) if alpha is not None else topo.compute_alpha(conf, amin, amax)
    Zs = topo.compute_structural_embedding(G, dim=k, seed=seed)
    feats = topo.fuse_features(fusion_result.fused, Zs, a)
    res = cd.FuzzyCMeans(n_clusters=k, m=m, seed=seed).fit(feats)
    n = G.number_of_nodes()
    score = onmi(views_from_U(res.U), true_view(d["true"], k), n)
    return score, res.U, a


def fuse(d, seed, e_t=None, e_v=None, common_dim=8, cls=NeuroFuzzyFusion, **kw):
    e_t = d["e_t"] if e_t is None else e_t
    e_v = d["e_v"] if e_v is None else e_v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = cls(common_dim=common_dim, anfis=ANFISAgreement(), seed=seed, **kw)
        return f, f.fuse(e_t, e_v)


class FusionVar(NeuroFuzzyFusion):
    """Same as NeuroFuzzyFusion but the PCA rank cap divisor (default 4 in the
    library: safe_rank = n_paired // 4) is configurable. Script-only."""

    def __init__(self, *a, rank_div=4, **kw):
        super().__init__(*a, **kw)
        self.rank_div = rank_div

    def _fit_alignment(self, Xt_paired, Xv_paired):
        n_paired = Xt_paired.shape[0]
        if n_paired < self.min_paired_samples:
            return False
        self._xt_mean = Xt_paired.mean(axis=0)
        self._xt_std = Xt_paired.std(axis=0) + 1e-8
        self._xv_mean = Xv_paired.mean(axis=0)
        self._xv_std = Xv_paired.std(axis=0) + 1e-8
        Xt_std = (Xt_paired - self._xt_mean) / self._xt_std
        Xv_std = (Xv_paired - self._xv_mean) / self._xv_std
        safe_rank = max(2, n_paired // self.rank_div)
        pca_dim_t = max(1, min(safe_rank, Xt_std.shape[1], n_paired - 1))
        pca_dim_v = max(1, min(safe_rank, Xv_std.shape[1], n_paired - 1))
        self._pca_t = PCA(n_components=pca_dim_t, random_state=self._seed).fit(Xt_std)
        self._pca_v = PCA(n_components=pca_dim_v, random_state=self._seed).fit(Xv_std)
        Zt = self._pca_t.transform(Xt_std)
        Zv = self._pca_v.transform(Xv_std)
        self._zt_mean, self._zt_std = Zt.mean(axis=0), Zt.std(axis=0) + 1e-8
        self._zv_mean, self._zv_std = Zv.mean(axis=0), Zv.std(axis=0) + 1e-8
        Zt_std = (Zt - self._zt_mean) / self._zt_std
        Zv_std = (Zv - self._zv_mean) / self._zv_std
        n_components = max(1, min(self.common_dim, Zt_std.shape[1], Zv_std.shape[1], n_paired - 1))
        self._cca = CCA(n_components=n_components, scale=False)
        self._cca.fit(Zt_std, Zv_std)
        self._fitted = True
        return True


def qstats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=len(x), mean=x.mean(), sd=x.std(), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4])


def bimodality(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return {}
    g = stats.skew(x)
    kx = stats.kurtosis(x)  # excess
    bc = (g ** 2 + 1) / (kx + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    gm1 = GaussianMixture(1, random_state=0).fit(x[:, None])
    gm2 = GaussianMixture(2, random_state=0).fit(x[:, None])
    return dict(bimodality_coef=bc, bic_gain_2_vs_1=gm1.bic(x[:, None]) - gm2.bic(x[:, None]))


def hungarian_correct(labels, primary, k):
    labels = np.asarray(labels)
    ks = int(max(labels.max(), primary.max())) + 1
    conf = np.zeros((ks, ks))
    for l, t in zip(labels, primary):
        conf[int(l), int(t)] += 1
    r, c = linear_sum_assignment(-conf)
    m = dict(zip(r, c))
    return np.array([m.get(int(l), -1) == int(t) for l, t in zip(labels, primary)])


def norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def diag_sanity(ds, d):
    """The helper pipeline must reproduce NFMCD.evaluate exactly."""
    for seed in SEEDS[:1]:
        if unit_done("sanity", ds, seed, "default"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NFMCD(n_communities=d["n_communities"], seed=seed)
            m.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            ref = m.evaluate(true_communities_per_node=d["true"])["overlapping_nmi"]
            m2 = NFMCD(n_communities=d["n_communities"], seed=seed, alpha_min=0.5, alpha_max=0.5)
            m2.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            ref2 = m2.evaluate(true_communities_per_node=d["true"])["overlapping_nmi"]
        _, fr = fuse(d, seed)
        mine, _, _ = cluster_from(d, fr, seed)
        mine2, _, _ = cluster_from(d, fr, seed, alpha=0.5)
        log(f"[{ds}] sanity seed={seed}: NFMCD={ref:.4f} helper={mine:.4f} | alpha=0.5 NFMCD={ref2:.4f} helper={mine2:.4f}")
        emit("sanity", ds, seed, "default", nfmcd=ref, helper=mine, nfmcd_a05=ref2, helper_a05=mine2)


def diag1_distributions(ds, d):
    e_none = [None] * len(d["e_t"])
    variants = {"full": (d["e_t"], d["e_v"]), "text_only": (d["e_t"], e_none), "image_only": (e_none, d["e_v"])}
    for name, (et, ev) in variants.items():
        if name == "image_only" and all(v is None for v in d["e_v"]):
            continue
        if name == "text_only" and all(v is None for v in d["e_t"]):
            continue
        for seed in SEEDS:
            if unit_done("d1_alpha_conf", ds, seed, name):
                continue
            _, fr = fuse(d, seed, et, ev)
            score, U, alpha = cluster_from(d, fr, seed)
            met = {"onmi": score}
            for k, v in qstats(alpha).items():
                met[f"alpha_{k}"] = v
            for k, v in qstats(fr.confidence).items():
                met[f"conf_{k}"] = v
            met["n_flag_both"] = sum(1 for f in fr.modality_flags if f == "both")
            met["n_flag_single"] = sum(1 for f in fr.modality_flags if f in ("text_only", "image_only"))
            emit("d1_alpha_conf", ds, seed, name, **met)
            log(f"[{ds}] d1 {name:10s} seed={seed}: ONMI={score:.3f} alpha mean={met['alpha_mean']:.3f} "
                f"q05/50/95={met['alpha_q05']:.2f}/{met['alpha_q50']:.2f}/{met['alpha_q95']:.2f} "
                f"conf mean={met['conf_mean']:.3f} both={met['n_flag_both']} single={met['n_flag_single']}")


ALPHAS = (0.02, 0.15, 0.3, 0.5, 0.7, 0.85, 0.98)


def diag2_alpha_sweep(ds, d):
    for seed in SEEDS:
        if all(unit_done("d2_alpha_sweep", ds, seed, f"a={a}") for a in ALPHAS):
            continue
        _, fr = fuse(d, seed)
        for a in ALPHAS:
            if unit_done("d2_alpha_sweep", ds, seed, f"a={a}"):
                continue
            score, _, _ = cluster_from(d, fr, seed, alpha=a)
            emit("d2_alpha_sweep", ds, seed, f"a={a}", onmi=score)
            log(f"[{ds}] d2 const alpha={a:.2f} seed={seed}: ONMI={score:.3f}")


def diag3_content_only(ds, d):
    k = d["n_communities"]
    n = d["G"].number_of_nodes()
    et, ev = d["e_t"], d["e_v"]
    paired = [i for i in range(n) if et[i] is not None and ev[i] is not None]
    if len(paired) < 20:
        return
    subset = set(paired)
    tv = true_view(d["true"], k, subset)
    T = norm_rows(np.stack([et[i] for i in paired]))
    V = norm_rows(np.stack([ev[i] for i in paired]))
    for seed in SEEDS:
        if unit_done("d3_content_only", ds, seed, "all"):
            continue
        _, fr = fuse(d, seed)
        Fz = fr.fused[paired]
        srcs = {"cca_fused": Fz, "raw_text": T, "raw_image": V, "raw_text+image": np.hstack([T, V])}
        met = {}
        for name, X in srcs.items():
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            lab = np.full(n, -1)
            for j, i in enumerate(paired):
                lab[i] = km.labels_[j]
            sc_km = onmi(views_from_labels(lab, k, subset), tv, n)
            res = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=seed).fit(X)
            Ufull = np.zeros((n, k))
            Ufull[paired] = res.U
            sc_fcm = onmi(views_from_U(Ufull, subset=subset), tv, n)
            met[f"{name}_kmeans"] = sc_km
            met[f"{name}_fcm"] = sc_fcm
        emit("d3_content_only", ds, seed, "all", **met)
        log(f"[{ds}] d3 content-only (paired n={len(paired)}) seed={seed}: " +
            " ".join(f"{k_}={v:.3f}" for k_, v in met.items()))


def diag4_confidence_informative(ds, d):
    k = d["n_communities"]
    n = d["G"].number_of_nodes()
    et, ev = d["e_t"], d["e_v"]
    paired = [i for i in range(n) if et[i] is not None and ev[i] is not None]
    if len(paired) < 20:
        return
    primary = np.array([min(d["true"][i]) if d["true"][i] else -1 for i in paired])
    for seed in SEEDS:
        if unit_done("d4_confidence", ds, seed, "all"):
            continue
        _, fr = fuse(d, seed)
        agr = fr.agreement[paired]
        conf = fr.confidence[paired]
        srcs = {"cca_fused": fr.fused[paired],
                "raw_text": norm_rows(np.stack([et[i] for i in paired]))}
        met = {}
        for name, X in srcs.items():
            lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_
            ok = hungarian_correct(lab, primary, k).astype(float)
            met[f"{name}_acc_overall"] = ok.mean()
            met[f"{name}_spearman_agreement"] = stats.spearmanr(agr, ok)[0]
            met[f"{name}_spearman_conf"] = stats.spearmanr(conf, ok)[0]
            edges = np.quantile(agr, [0, .25, .5, .75, 1.0])
            for q in range(4):
                lo, hi = edges[q], edges[q + 1]
                mask = (agr >= lo) & ((agr <= hi) if q == 3 else (agr < hi))
                met[f"{name}_acc_agq{q + 1}"] = ok[mask].mean() if mask.any() else float("nan")
                met[f"agq{q + 1}_mean_agreement"] = agr[mask].mean() if mask.any() else float("nan")
        emit("d4_confidence", ds, seed, "all", **met)
        log(f"[{ds}] d4 seed={seed}: acc overall cca={met['cca_fused_acc_overall']:.3f} raw_text={met['raw_text_acc_overall']:.3f}; "
            f"spearman(agreement,correct) cca={met['cca_fused_spearman_agreement']:+.3f} raw_text={met['raw_text_spearman_agreement']:+.3f}; "
            f"acc by agreement quartile (cca) " + "/".join(f"{met[f'cca_fused_acc_agq{q}']:.2f}" for q in (1, 2, 3, 4)))


def diag5_cca_sensitivity(ds, d):
    for seed in SEEDS:
        for cdim in (2, 4, 8, 16):
            for div in (16, 8, 4):
                var = f"cd={cdim},div={div}"
                if unit_done("d5_cca", ds, seed, var):
                    continue
                _, fr = fuse(d, seed, common_dim=cdim, cls=FusionVar, rank_div=div)
                score, _, alpha = cluster_from(d, fr, seed)
                met = {"onmi": score, "alpha_mean": alpha.mean(), "conf_mean": fr.confidence.mean()}
                for k, v in qstats(fr.agreement).items():
                    met[f"agr_{k}"] = v
                emit("d5_cca", ds, seed, var, **met)
                log(f"[{ds}] d5 {var} seed={seed}: ONMI={score:.3f} agreement mean={met['agr_mean']:.3f} sd={met['agr_sd']:.3f} alpha mean={alpha.mean():.3f}")


def diag6_agreement_dist(ds, d):
    for seed in SEEDS[:1]:
        if unit_done("d6_agreement", ds, seed, "full"):
            continue
        _, fr = fuse(d, seed)
        a = fr.agreement[np.isfinite(fr.agreement)]
        met = dict(qstats(a))
        met.update(bimodality(a))
        emit("d6_agreement", ds, seed, "full", **met)
        log(f"[{ds}] d6 agreement: mean={met['mean']:.3f} sd={met['sd']:.3f} q05/50/95={met['q05']:.2f}/{met['q50']:.2f}/{met['q95']:.2f} "
            f"BC={met['bimodality_coef']:.3f} (>0.555 suggests bimodal) BIC gain 2v1 comp={met['bic_gain_2_vs_1']:.1f}")
        np.save(os.path.join(EXP, f"fusion_agreement_{ds}.npy"), a)


def diag7_mixed_space(ds, d):
    """Nodes missing one modality get the first `out_dim` RAW embedding
    coordinates in a vector whose other rows live in CCA space. Test the effect."""
    n = d["G"].number_of_nodes()
    for seed in SEEDS:
        if unit_done("d7_mixed_space", ds, seed, "all"):
            continue
        f, fr = fuse(d, seed)
        single = [i for i, fl in enumerate(fr.modality_flags) if fl in ("text_only", "image_only")]
        if not single:
            emit("d7_mixed_space", ds, seed, "all", n_single=0)
            log(f"[{ds}] d7: no single-modality nodes -> n/a")
            continue
        base, _, _ = cluster_from(d, fr, seed)
        zeroed = fr.fused.copy()
        zeroed[single] = 0.0
        proj = fr.fused.copy()
        for i in single:
            if fr.modality_flags[i] == "text_only":
                p = f._project_text(d["e_t"][i][None, :])[0]
            else:
                p = f._project_image(d["e_v"][i][None, :])[0]
            proj[i, :len(p)] = p / (np.linalg.norm(p) + 1e-12)
        from copy import copy
        fz, fp = copy(fr), copy(fr)
        fz.fused, fp.fused = zeroed, proj
        s_zero, _, _ = cluster_from(d, fz, seed)
        s_proj, _, _ = cluster_from(d, fp, seed)
        emit("d7_mixed_space", ds, seed, "all", n_single=len(single), onmi_asis=base, onmi_zeroed=s_zero, onmi_projected=s_proj)
        log(f"[{ds}] d7 seed={seed}: {len(single)} single-modality nodes; ONMI as-is={base:.3f} zeroed={s_zero:.3f} cca-projected={s_proj:.3f}")


def main():
    open(LOG_PATH, "a").close()
    t0 = time.monotonic()
    log(f"=== fusion diagnosis start {time.strftime('%Y-%m-%d %H:%M:%S')} datasets={DATASETS} seeds={SEEDS} ===")
    for ds in DATASETS:
        d = load_cache(ds)
        n_t = sum(v is not None for v in d["e_t"])
        n_v = sum(v is not None for v in d["e_v"])
        log(f"\n--- {ds}: {d['G'].number_of_nodes()} nodes, k={d['n_communities']}, text={n_t}, image={n_v} ---")
        for fn in (diag_sanity, diag1_distributions, diag2_alpha_sweep, diag3_content_only,
                   diag4_confidence_informative, diag6_agreement_dist, diag7_mixed_space, diag5_cca_sensitivity):
            fn(ds, d)
    make_plots()
    log(f"=== done in {time.monotonic() - t0:.0f}s ===")


def agg(diag, ds, variant_prefix=None):
    out = {}
    for r in ROWS:
        if r["diagnostic"] == diag and r["dataset"] == ds:
            out.setdefault((r["variant"], r["metric"]), []).append(float(r["value"]))
    return out


def save_fig(fig, path):
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, path)


def make_plots():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds in DATASETS:
        a = agg("d2_alpha_sweep", ds)
        xs = [al for al in ALPHAS if (f"a={al}", "onmi") in a]
        if not xs:
            continue
        mu = [np.mean(a[(f"a={al}", "onmi")]) for al in xs]
        sd = [np.std(a[(f"a={al}", "onmi")]) for al in xs]
        ln = ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3, label=f"{ds}: constant alpha")
        d1 = agg("d1_alpha_conf", ds)
        for vname, ls in (("full", "--"), ("text_only", ":")):
            if (vname, "onmi") in d1:
                ax.axhline(np.mean(d1[(vname, "onmi")]), color=ln[0].get_color(), linestyle=ls, alpha=0.6,
                           label=f"{ds}: {vname} (fuzzy alpha)")
    ax.set_xlabel("constant content weight alpha (0=structure only, 1=content only)")
    ax.set_ylabel("LFK overlapping NMI")
    ax.set_title("ONMI vs content weight (k = ground-truth, mean +- sd over 3 seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(EXP, "fusion_alpha_sweep.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.38
    for ds_i, ds in enumerate(DATASETS):
        d4 = agg("d4_confidence", ds)
        if not d4:
            continue
        for j, src in enumerate(("cca_fused", "raw_text")):
            vals = [np.mean(d4[("all", f"{src}_acc_agq{q}")]) for q in (1, 2, 3, 4)]
            axes[0].bar(np.arange(4) + ds_i * 2.4 + j * w - w / 2, vals, w, label=f"{ds} {src}" if True else None)
    axes[0].set_xticks([0, 1, 2, 3, 2.4, 3.4, 4.4, 5.4])
    axes[0].set_xticklabels(["Q1", "Q2", "Q3", "Q4"] * 2, fontsize=7)
    axes[0].set_ylabel("content-only k-means accuracy (Hungarian)")
    axes[0].set_xlabel("agreement quartile (Q1 lowest) | left: first dataset, right: second")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Does high agreement mean more accurate content?")
    for ds in DATASETS:
        p = os.path.join(EXP, f"fusion_agreement_{ds}.npy")
        if os.path.exists(p):
            axes[1].hist(np.load(p), bins=40, alpha=0.5, label=ds)
    axes[1].set_xlabel("cross-modal agreement (cosine in CCA space)")
    axes[1].set_title("Agreement distribution")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(EXP, "fusion_confidence_vs_correctness.png"))


if __name__ == "__main__":
    main()
