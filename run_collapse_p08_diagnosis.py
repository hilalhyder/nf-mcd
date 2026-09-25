"""
Diagnose why NFMCD.robust() collapses (mean_umax -> exactly 1/k) at k=32,
n=8000 combined with 80% "both"-mode missing content (experiments/compound_summary.log,
condition="missing"), when it resists the SAME k=32 combined with noisy
structure or cross-modal mismatch. Read-only w.r.t. nf_mcd/*; this file only
calls existing code. Reuses run_compound.py's exact graph construction
(p_in_out_for, generate_synthetic_multimodal_graph defaults) and
run_missing.apply_mask so the graph/masks are identical to the ones that
produced the ONMI=0.000 finding.

Usage:
    py run_collapse_p08_diagnosis.py run       # all items, resumable
    py run_collapse_p08_diagnosis.py summary   # (re)build summary.log + plots
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "collapsep08_results.csv")
SUMMARY_LOG = os.path.join(EXP, "collapsep08_summary.log")
PROGRESS_MD = os.path.join(EXP, "collapsep08_progress.md")

sys.path.insert(0, HERE)

BASE_K = 32
BASE_N = 8000
TARGET_DEGREE = 18.0
BASE_RATIO = 0.02 / 0.18

FIELDS = ["item", "p", "k", "n", "seed", "alpha_forced", "sub", "onmi", "mean_umax",
          "m_used", "extra_json"]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(note):
    atomic_write_text(PROGRESS_MD, (
        "# Collapse-at-p=0.8 diagnosis progress\n\n"
        f"- {note}\n- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n"
        f"```\ncd {HERE}\npy run_collapse_p08_diagnosis.py run\n"
        "py run_collapse_p08_diagnosis.py summary\n```\n"
    ))


def load_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rkey(item, p, k, seed, alpha_forced, sub):
    return (item, f"{float(p):.2f}", int(k), int(seed), str(alpha_forced), str(sub))


_csv_file = None
_csv_writer = None


def _open_csv():
    global _csv_file, _csv_writer
    new = not os.path.exists(RESULTS_CSV)
    _csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=FIELDS)
    if new:
        _csv_writer.writeheader()
        _csv_file.flush()
        os.fsync(_csv_file.fileno())


def _append(row):
    for k in FIELDS:
        row.setdefault(k, "")
    _csv_writer.writerow(row)
    _csv_file.flush()
    os.fsync(_csv_file.fileno())


# ---------------------------------------------------------------------------
# dataset construction (identical to run_compound.py's "missing" condition)
# ---------------------------------------------------------------------------

def _base_dataset(seed: int, k: int = BASE_K, n: int = BASE_N):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph
    from run_scale import p_in_out_for
    p_in, p_out = p_in_out_for(n, n_communities=k, target_degree=TARGET_DEGREE, ratio=BASE_RATIO)
    data = generate_synthetic_multimodal_graph(n_nodes=n, n_communities=k, p_in=p_in, p_out=p_out, seed=seed)
    return dict(G=data.G, e_t=data.text_embeddings, e_v=data.image_embeddings,
                true=data.true_communities, n_communities=k)


def _masked(seed: int, p: float, k: int = BASE_K, n: int = BASE_N):
    d = _base_dataset(seed, k, n)
    if p <= 0:
        return d
    from run_missing import apply_mask
    return apply_mask(d, "both", p, seed)


def _score_U(U: np.ndarray, true_communities, k: int, n: int) -> float:
    from nf_mcd.community_detection import overlapping_communities
    from nf_mcd.metrics import overlapping_nmi
    pred_node = overlapping_communities(U, threshold=0.2)
    pred_view = [set() for _ in range(k)]
    for i, comms in enumerate(pred_node):
        for c in comms:
            pred_view[c].add(i)
    true_view = [set() for _ in range(k)]
    for i, comms in enumerate(true_communities):
        for c in comms:
            if 0 <= c < k:
                true_view[c].add(i)
    return overlapping_nmi(pred_view, true_view, n_nodes=n)


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

def item1_2(rows_done, seed, p):
    """Census (modality flags, raw-PCA content zero/nonzero rows, effective rank)
    + which m adaptive-m FCM actually tries and its mean(U.max) at each rung,
    using the SAME fused_features robust() would build."""
    from nf_mcd import topology as topo
    from nf_mcd.clusterers import _kmeans, _fcm_from_centers, M_LADDER, NONDEGENERACY
    from nf_mcd.pipeline import NFMCD

    d = _masked(seed, p)
    n = d["G"].number_of_nodes()
    e_t, e_v = d["e_t"], d["e_v"]
    has_t = np.array([v is not None for v in e_t])
    has_v = np.array([v is not None for v in e_v])
    flag_both = float((has_t & has_v).mean())
    flag_text = float((has_t & ~has_v).mean())
    flag_img = float((~has_t & has_v).mean())
    flag_none = float((~has_t & ~has_v).mean())

    Z_c = topo.raw_pca_content(e_t, e_v, dim=16, seed=seed)
    row_norm = np.linalg.norm(Z_c, axis=1)
    frac_zero_rows = float((row_norm < 1e-9).mean())
    eff_rank = int(np.linalg.matrix_rank(Z_c[row_norm > 1e-9])) if (row_norm > 1e-9).sum() > 1 else 0

    key = rkey("census", p, BASE_K, seed, "", "flags")
    if key not in rows_done:
        _append(dict(item="census", p=p, k=BASE_K, n=n, seed=seed, alpha_forced="", sub="flags",
                     extra_json=json.dumps(dict(flag_both=flag_both, flag_text=flag_text,
                                                 flag_img=flag_img, flag_none=flag_none,
                                                 content_dim=Z_c.shape[1], frac_zero_content_rows=frac_zero_rows,
                                                 effective_rank_nonzero_rows=eff_rank))))
        rows_done.add(key)

    key = rkey("mselect", p, BASE_K, seed, "", "fit")
    if key not in rows_done:
        model = NFMCD.robust(n_communities=BASE_K, seed=seed)
        model.fit(d["G"], text_embeddings=e_t, image_embeddings=e_v)
        onmi = _score_U(model.U_, d["true"], BASE_K, n)
        mean_umax = float(model.U_.max(axis=1).mean())
        _append(dict(item="mselect", p=p, k=BASE_K, n=n, seed=seed, alpha_forced="", sub="robust_fit",
                     onmi=onmi, mean_umax=mean_umax, m_used=getattr(model.fcm_result_, "m_used", ""),
                     extra_json=""))
        rows_done.add(key)

        # Replay the adaptive-m ladder on the SAME fused_features to see mean(U.max)
        # at every rung, not just the one it stopped on.
        X = model.fused_features_
        C0 = _kmeans(X, BASE_K, seed).cluster_centers_
        floor = 1.0 / BASE_K + NONDEGENERACY * (1.0 - 1.0 / BASE_K)
        ladder = {}
        for m in M_LADDER:
            U, C, n_iter, hist = _fcm_from_centers(X, C0.copy(), m)
            ladder[m] = float(U.max(axis=1).mean())
        key2 = rkey("mladder", p, BASE_K, seed, "", "ladder")
        _append(dict(item="mladder", p=p, k=BASE_K, n=n, seed=seed, alpha_forced="", sub="ladder",
                     extra_json=json.dumps(dict(floor=floor, ladder={str(k2): v for k2, v in ladder.items()}))))
        rows_done.add(key2)
    return d


def item3_structure_alone(rows_done, seed):
    """Structure-only ablation (Z_c forced to zero, alpha=0) -- graph is
    identical at every p (missingness never touches edges), so this is a
    single per-seed number."""
    key = rkey("structure_alone", 0, BASE_K, seed, "", "struct")
    if key in rows_done:
        return
    from nf_mcd import topology as topo
    from nf_mcd.clusterers import fcm_adaptive_m

    d = _base_dataset(seed, BASE_K, BASE_N)
    n = d["G"].number_of_nodes()
    Z_s = topo.compute_structural_embedding(d["G"], dim=BASE_K, seed=seed)
    Z_c = np.zeros((n, 16))
    alpha = np.zeros(n)
    X = topo.fuse_features(Z_c, Z_s, alpha)
    res = fcm_adaptive_m(X, BASE_K, seed)
    onmi = _score_U(res.U, d["true"], BASE_K, n)
    mean_umax = float(res.U.max(axis=1).mean())
    _append(dict(item="structure_alone", p=0, k=BASE_K, n=n, seed=seed, alpha_forced="0.0", sub="struct",
                 onmi=onmi, mean_umax=mean_umax, m_used=getattr(res, "m_used", "")))
    rows_done.add(key)


def item4_content_alone(rows_done, seed, p=0.8):
    """Content-only ablation at p=0.8 (Z_s forced to zero, alpha=1)."""
    key = rkey("content_alone", p, BASE_K, seed, "", "content")
    if key in rows_done:
        return
    from nf_mcd import topology as topo
    from nf_mcd.clusterers import fcm_adaptive_m

    d = _masked(seed, p)
    n = d["G"].number_of_nodes()
    Z_c = topo.raw_pca_content(d["e_t"], d["e_v"], dim=16, seed=seed)
    Z_s = np.zeros((n, BASE_K))
    alpha = np.ones(n)
    X = topo.fuse_features(Z_c, Z_s, alpha)
    res = fcm_adaptive_m(X, BASE_K, seed)
    onmi = _score_U(res.U, d["true"], BASE_K, n)
    mean_umax = float(res.U.max(axis=1).mean())
    _append(dict(item="content_alone", p=p, k=BASE_K, n=n, seed=seed, alpha_forced="1.0", sub="content",
                 onmi=onmi, mean_umax=mean_umax, m_used=getattr(res, "m_used", "")))
    rows_done.add(key)


def item5_alpha_sweep(rows_done, seed, p=0.8):
    from nf_mcd.pipeline import NFMCD
    d = _masked(seed, p)
    n = d["G"].number_of_nodes()
    for a in (0.05, 0.15, 0.3, 0.5, 0.7, 0.85):
        key = rkey("alpha_sweep", p, BASE_K, seed, a, "fixed_alpha")
        if key in rows_done:
            continue
        model = NFMCD.robust(n_communities=BASE_K, seed=seed, alpha_min=a, alpha_max=a)
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
        onmi = _score_U(model.U_, d["true"], BASE_K, n)
        mean_umax = float(model.U_.max(axis=1).mean())
        _append(dict(item="alpha_sweep", p=p, k=BASE_K, n=n, seed=seed, alpha_forced=a, sub="fixed_alpha",
                     onmi=onmi, mean_umax=mean_umax, m_used=getattr(model.fcm_result_, "m_used", "")))
        rows_done.add(key)


def item6_onset(rows_done, seed):
    from nf_mcd.pipeline import NFMCD
    for p in (0.0, 0.2, 0.4, 0.6, 0.7, 0.8):
        key = rkey("onset", p, BASE_K, seed, "", "robust")
        if key in rows_done:
            continue
        d = _masked(seed, p)
        n = d["G"].number_of_nodes()
        model = NFMCD.robust(n_communities=BASE_K, seed=seed)
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
        onmi = _score_U(model.U_, d["true"], BASE_K, n)
        mean_umax = float(model.U_.max(axis=1).mean())
        _append(dict(item="onset", p=p, k=BASE_K, n=n, seed=seed, alpha_forced="", sub="robust",
                     onmi=onmi, mean_umax=mean_umax, m_used=getattr(model.fcm_result_, "m_used", "")))
        rows_done.add(key)


def item7_k_interaction(rows_done, seed, p=0.8, seeds_cap=2):
    if seed >= seeds_cap:
        return
    from nf_mcd.pipeline import NFMCD
    for k in (4, 8, 16, 32):
        n = 250 * k
        key = rkey("k_interaction", p, k, seed, "", "robust")
        if key in rows_done:
            continue
        d = _masked(seed, p, k=k, n=n)
        model = NFMCD.robust(n_communities=k, seed=seed)
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
        onmi = _score_U(model.U_, d["true"], k, n)
        mean_umax = float(model.U_.max(axis=1).mean())
        _append(dict(item="k_interaction", p=p, k=k, n=n, seed=seed, alpha_forced="", sub="robust",
                     onmi=onmi, mean_umax=mean_umax, m_used=getattr(model.fcm_result_, "m_used", "")))
        rows_done.add(key)


def cmd_run():
    import warnings
    warnings.filterwarnings("ignore")
    existing = load_results()
    rows_done = {rkey(r["item"], r["p"], r["k"], r["seed"], r["alpha_forced"], r["sub"]) for r in existing}
    _open_csv()
    t0 = time.monotonic()
    try:
        for seed in (0, 1, 2):
            for p in (0.0, 0.4, 0.8):
                item1_2(rows_done, seed, p)
                print(f"[census/mselect] seed={seed} p={p} done ({time.monotonic()-t0:.0f}s)", flush=True)
            item3_structure_alone(rows_done, seed)
            print(f"[structure_alone] seed={seed} done ({time.monotonic()-t0:.0f}s)", flush=True)
            item4_content_alone(rows_done, seed)
            print(f"[content_alone] seed={seed} done ({time.monotonic()-t0:.0f}s)", flush=True)
            item5_alpha_sweep(rows_done, seed)
            print(f"[alpha_sweep] seed={seed} done ({time.monotonic()-t0:.0f}s)", flush=True)
            item6_onset(rows_done, seed)
            print(f"[onset] seed={seed} done ({time.monotonic()-t0:.0f}s)", flush=True)
            item7_k_interaction(rows_done, seed, seeds_cap=2)
            print(f"[k_interaction] seed={seed} done ({time.monotonic()-t0:.0f}s)", flush=True)
            write_progress(f"finished seed={seed} ({time.monotonic()-t0:.0f}s elapsed)")
    finally:
        _csv_file.close()
    write_progress(f"done ({time.monotonic()-t0:.0f}s total)")
    print(f"Run complete in {time.monotonic()-t0:.0f}s.")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def cmd_summary():
    rows = load_results()
    if not rows:
        print("No results yet.")
        return

    def fv(r, k):
        v = r.get(k, "")
        if v in ("", None):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    L = []
    L.append("Diagnosis: why does NFMCD.robust() collapse at k=32/n=8000 + 80% missing content?")
    L.append(f"1/k = {1/BASE_K:.4f}. jobs: {len(rows)}")
    L.append("=" * 92)

    L.append("\n1. Census at p=0/0.4/0.8 (mean over seeds 0-2)")
    L.append("-" * 92)
    for p in (0.0, 0.4, 0.8):
        rs = [r for r in rows if r["item"] == "census" and fv(r, "p") == p]
        if not rs:
            continue
        ex = [json.loads(r["extra_json"]) for r in rs]
        keys = ex[0].keys()
        agg = {k: float(np.mean([e[k] for e in ex])) for k in keys}
        L.append(f"  p={p}: flag both={agg['flag_both']:.3f} text_only={agg['flag_text']:.3f} "
                  f"image_only={agg['flag_img']:.3f} none={agg['flag_none']:.3f} | "
                  f"raw-PCA content_dim={agg['content_dim']:.0f} frac_zero_rows={agg['frac_zero_content_rows']:.3f} "
                  f"effective_rank(nonzero rows)={agg['effective_rank_nonzero_rows']:.1f}")

    L.append("\n2. Adaptive-m ladder: mean(U.max) at each m, on the SAME fused_features robust() built")
    L.append("-" * 92)
    for p in (0.0, 0.4, 0.8):
        rs = [r for r in rows if r["item"] == "mladder" and fv(r, "p") == p]
        if not rs:
            continue
        ex = [json.loads(r["extra_json"]) for r in rs]
        floor = ex[0]["floor"]
        ms = sorted(ex[0]["ladder"].keys(), key=float)
        agg = {m: float(np.mean([e["ladder"][m] for e in ex])) for m in ms}
        L.append(f"  p={p}: non-degeneracy floor={floor:.4f}")
        for m in ms:
            passed = "PASS" if agg[m] >= floor else "fail"
            L.append(f"      m={m}: mean(U.max)={agg[m]:.4f}  [{passed}]")
        fit_rs = [r for r in rows if r["item"] == "mselect" and fv(r, "p") == p]
        if fit_rs:
            onmi = np.mean([fv(r, "onmi") for r in fit_rs])
            umax = np.mean([fv(r, "mean_umax") for r in fit_rs])
            m_used = fit_rs[0]["m_used"]
            L.append(f"    -> robust() actually chose m={m_used}: mean ONMI={onmi:.4f}, mean(U.max)={umax:.4f}")

    L.append("\n3. Structure-alone ablation (content zeroed, alpha=0; graph independent of p)")
    L.append("-" * 92)
    rs = [r for r in rows if r["item"] == "structure_alone"]
    if rs:
        onmi = np.mean([fv(r, "onmi") for r in rs])
        umax = np.mean([fv(r, "mean_umax") for r in rs])
        L.append(f"  mean ONMI={onmi:.4f}, mean(U.max)={umax:.4f}, n_seeds={len(rs)}")
        L.append("  -> if this is already low/collapsed, k=32 structural separability is itself marginal;")
        L.append("     if it's solid, the p=0.8 collapse comes from adding degenerate content, not from k=32 alone.")

    L.append("\n4. Content-alone ablation at p=0.8 (structure zeroed, alpha=1)")
    L.append("-" * 92)
    rs = [r for r in rows if r["item"] == "content_alone"]
    if rs:
        onmi = np.mean([fv(r, "onmi") for r in rs])
        umax = np.mean([fv(r, "mean_umax") for r in rs])
        L.append(f"  mean ONMI={onmi:.4f}, mean(U.max)={umax:.4f}, n_seeds={len(rs)}  (content-only floor at p=0.8)")

    L.append("\n5. Fixed-alpha sweep at p=0.8 -- does ANY alpha avoid collapse?")
    L.append("-" * 92)
    for a in (0.05, 0.15, 0.3, 0.5, 0.7, 0.85):
        rs = [r for r in rows if r["item"] == "alpha_sweep" and fv(r, "alpha_forced") == a]
        if not rs:
            continue
        onmi = np.mean([fv(r, "onmi") for r in rs])
        umax = np.mean([fv(r, "mean_umax") for r in rs])
        L.append(f"  alpha={a}: mean ONMI={onmi:.4f}, mean(U.max)={umax:.4f}  [{'collapsed' if umax < 1/BASE_K+0.05 else 'ok'}]")

    L.append("\n6. Onset curve: robust() as-is, fine-grained p at k=32")
    L.append("-" * 92)
    L.append(f"{'p':>6}{'onmi':>10}{'mean_umax':>12}{'status':>12}")
    onset_pts = []
    for p in (0.0, 0.2, 0.4, 0.6, 0.7, 0.8):
        rs = [r for r in rows if r["item"] == "onset" and fv(r, "p") == p]
        if not rs:
            continue
        onmi = np.mean([fv(r, "onmi") for r in rs])
        umax = np.mean([fv(r, "mean_umax") for r in rs])
        status = "collapsed" if umax < 1 / BASE_K + 0.05 else "ok"
        L.append(f"{p:>6.2f}{onmi:>10.4f}{umax:>12.4f}{status:>12}")
        onset_pts.append((p, onmi, umax))

    L.append("\n7. k-interaction at p=0.8: does reducing k avoid the collapse? (seeds 0-1)")
    L.append("-" * 92)
    L.append(f"{'k':>6}{'n':>8}{'onmi':>10}{'mean_umax':>12}{'status':>12}")
    k_pts = []
    for k in (4, 8, 16, 32):
        rs = [r for r in rows if r["item"] == "k_interaction" and int(r["k"]) == k]
        if not rs:
            continue
        onmi = np.mean([fv(r, "onmi") for r in rs])
        umax = np.mean([fv(r, "mean_umax") for r in rs])
        status = "collapsed" if umax < 1 / k + 0.05 else "ok"
        L.append(f"{k:>6}{250*k:>8}{onmi:>10.4f}{umax:>12.4f}{status:>12}")
        k_pts.append((k, onmi, umax))

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if onset_pts:
            fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
            ps, onmis, umaxs = zip(*onset_pts)
            ax1.plot(ps, onmis, "o-", color="#0B6E6B", label="ONMI")
            ax1.set_xlabel("missing rate p (both modalities independently)")
            ax1.set_ylabel("LFK ONMI", color="#0B6E6B")
            ax1.axhline(0, color="gray", lw=0.5)
            ax2 = ax1.twinx()
            ax2.plot(ps, umaxs, "s--", color="#C2410C", label="mean(U.max)")
            ax2.axhline(1 / BASE_K, color="#C2410C", linestyle=":", lw=1)
            ax2.set_ylabel("mean(U.max)  (dotted = 1/k)", color="#C2410C")
            ax1.set_title(f"Collapse onset: NFMCD.robust() at k={BASE_K}, n={BASE_N}")
            fig.tight_layout()
            fig.savefig(os.path.join(EXP, "collapsep08_onset.png"), dpi=150)
            plt.close(fig)

        if k_pts:
            fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
            ks, onmis, umaxs = zip(*k_pts)
            ax1.plot(ks, onmis, "o-", color="#0B6E6B", label="ONMI")
            ax1.set_xlabel("k (n = 250*k), p=0.8 both-missing")
            ax1.set_ylabel("LFK ONMI", color="#0B6E6B")
            ax2 = ax1.twinx()
            ax2.plot(ks, umaxs, "s--", color="#C2410C", label="mean(U.max)")
            ax2.plot(ks, [1 / k for k in ks], ":", color="gray", lw=1, label="1/k")
            ax2.set_ylabel("mean(U.max)", color="#C2410C")
            ax1.set_title("Does lowering k avoid the collapse at p=0.8?")
            fig.tight_layout()
            fig.savefig(os.path.join(EXP, "collapsep08_k_interaction.png"), dpi=150)
            plt.close(fig)
        print("Plots written: collapsep08_onset.png, collapsep08_k_interaction.png")
    except Exception as exc:
        print(f"(plotting skipped: {exc})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run()
        cmd_summary()
    elif cmd == "summary":
        cmd_summary()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
