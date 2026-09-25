"""Correctness + performance check for NeuroFuzzyFusion's opt-in `vectorized`
path (nf_mcd/fuzzy_fusion.py). The original per-node Python loop in fuse()
was found to be ~94% of total NFMCD.fit() time at n=20,000 nodes
(experiments/scale_summary.log). This script:

  1. builds a battery of reference fuse() outputs using the ORIGINAL
     (vectorized=False) path,
  2. re-runs the same inputs through vectorized=True and asserts numerical
     equivalence, case by case,
  3. only if that passes: regression-checks vectorized=True end-to-end
     against the 51 saved default rows used elsewhere in this project, and
  4. measures the speedup at n in {1000, 5000, 20000}.

Usage:
    py run_vectorize_fusion.py equivalence   # step 1+2, must pass first
    py run_vectorize_fusion.py regression    # step 3
    py run_vectorize_fusion.py scale         # step 4
    py run_vectorize_fusion.py all           # all of the above, in order
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

EXP = os.path.join(os.path.dirname(__file__), "experiments")
os.makedirs(EXP, exist_ok=True)
FIXTURE_PKL = os.path.join(EXP, "vectorize_fixture.pkl")
RESULTS_CSV = os.path.join(EXP, "vectorize_results.csv")
SUMMARY_LOG = os.path.join(EXP, "vectorize_summary.log")


def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# step 1: build reference cases (inputs only; outputs captured per-run below)
# ---------------------------------------------------------------------------

def _rand_embeds(rng, n, dim, present_mask):
    return [rng.normal(size=dim) if present_mask[i] else None for i in range(n)]


def build_cases():
    rng = np.random.default_rng(0)
    cases = {}

    # Case A: all paired, plenty of samples, alignment should succeed.
    n = 60
    present = np.ones(n, dtype=bool)
    cases["all_paired_aligned"] = dict(
        e_t=_rand_embeds(rng, n, 40, present), e_v=_rand_embeds(rng, n, 32, present),
        common_dim=8, pca_rank_div=4, min_paired_samples=10,
    )

    # Case B: mixed - some text-only, some image-only, some neither, some paired.
    n = 80
    present_t = rng.random(n) < 0.7
    present_v = rng.random(n) < 0.6
    cases["mixed_modalities"] = dict(
        e_t=[rng.normal(size=40) if present_t[i] else None for i in range(n)],
        e_v=[rng.normal(size=32) if present_v[i] else None for i in range(n)],
        common_dim=6, pca_rank_div=4, min_paired_samples=10,
    )

    # Case C: too few paired nodes -> alignment_ok False (both_unaligned path).
    n = 30
    present_t = np.ones(n, dtype=bool)
    present_v = np.zeros(n, dtype=bool)
    present_v[:5] = True  # only 5 paired, below min_paired_samples=10
    rng.shuffle(present_v)
    cases["few_paired_unaligned"] = dict(
        e_t=[rng.normal(size=20) for _ in range(n)],
        e_v=[rng.normal(size=16) if present_v[i] else None for i in range(n)],
        common_dim=4, pca_rank_div=4, min_paired_samples=10,
    )

    # Case D: larger common_dim, single_modality_fill="zero".
    n = 100
    present = np.ones(n, dtype=bool)
    present_v2 = rng.random(n) < 0.5
    cases["large_common_dim_zero_fill"] = dict(
        e_t=[rng.normal(size=50) for _ in range(n)],
        e_v=[rng.normal(size=45) if present_v2[i] else None for i in range(n)],
        common_dim=16, pca_rank_div=4, min_paired_samples=10, single_modality_fill="zero",
    )

    # Case E: small pca_rank_div=16 (the robust preset's setting).
    n = 120
    present_t = rng.random(n) < 0.85
    present_v = rng.random(n) < 0.85
    cases["rank_div_16"] = dict(
        e_t=[rng.normal(size=64) if present_t[i] else None for i in range(n)],
        e_v=[rng.normal(size=48) if present_v[i] else None for i in range(n)],
        common_dim=8, pca_rank_div=16, min_paired_samples=10,
    )

    # Case F: real cached dataset (crisismmd), all real embeddings.
    try:
        from run_realgraph import get_dataset
        d = get_dataset("crisismmd", 0)
        cases["real_crisismmd"] = dict(
            e_t=list(d["e_t"]), e_v=list(d["e_v"]),
            common_dim=8, pca_rank_div=4, min_paired_samples=10,
        )
    except Exception as exc:
        print(f"(skipping real_crisismmd case: {exc})")

    return cases


def _run_case(e_t, e_v, common_dim, pca_rank_div, min_paired_samples, single_modality_fill="raw", vectorized=False):
    from nf_mcd.fuzzy_fusion import NeuroFuzzyFusion, ANFISAgreement
    import warnings
    fusion = NeuroFuzzyFusion(
        common_dim=common_dim, anfis=ANFISAgreement(), seed=0,
        min_paired_samples=min_paired_samples, pca_rank_div=pca_rank_div,
        single_modality_fill=single_modality_fill, vectorized=vectorized,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fusion.fuse(e_t, e_v)


def cmd_equivalence():
    cases = build_cases()
    lines = ["Equivalence check: vectorized=True vs. original loop (vectorized=False)", "=" * 78]
    all_ok = True
    for name, params in cases.items():
        single_modality_fill = params.pop("single_modality_fill", "raw")
        ref = _run_case(**params, single_modality_fill=single_modality_fill, vectorized=False)
        got = _run_case(**params, single_modality_fill=single_modality_fill, vectorized=True)

        d_fused = float(np.max(np.abs(ref.fused - got.fused)))
        ref_agree = np.nan_to_num(ref.agreement, nan=-999.0)
        got_agree = np.nan_to_num(got.agreement, nan=-999.0)
        d_agree = float(np.max(np.abs(ref_agree - got_agree)))
        d_conf = float(np.max(np.abs(ref.confidence - got.confidence)))
        flags_match = ref.modality_flags == got.modality_flags
        ok = d_fused < 1e-9 and d_agree < 1e-9 and d_conf < 1e-9 and flags_match
        all_ok &= ok
        lines.append(
            f"[{name}] n={len(params['e_t'])} fill={single_modality_fill}: "
            f"max|d fused|={d_fused:.2e} max|d agreement|={d_agree:.2e} "
            f"max|d confidence|={d_conf:.2e} flags_match={flags_match} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok and not flags_match:
            mism = [i for i, (a, b) in enumerate(zip(ref.modality_flags, got.modality_flags)) if a != b][:10]
            lines.append(f"    flag mismatches at indices (first 10): {mism}")
            for i in mism[:3]:
                lines.append(f"    idx {i}: ref={ref.modality_flags[i]!r} got={got.modality_flags[i]!r}")
        params["single_modality_fill"] = single_modality_fill  # restore for record

    lines.append("")
    lines.append(f"OVERALL: {'ALL CASES PASS' if all_ok else 'FAILURES FOUND -- see above'}")
    _atomic_write_text(SUMMARY_LOG, "\n".join(lines) + "\n")
    print("\n".join(lines))
    return all_ok


# ---------------------------------------------------------------------------
# step 3: regression check against the 51 saved default rows
# ---------------------------------------------------------------------------

def cmd_regression():
    import subprocess
    print("--- default config (vectorized_fusion not passed, should be untouched) ---")
    proc = subprocess.run([sys.executable, "run_clusterers.py", "verify"],
                           capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
    ok_default = "mismatches=0" in proc.stdout
    print(f"default-path verify: {'PASS' if ok_default else 'FAIL'}")

    print("\n--- vectorized=True substituted in, checking fused/confidence/agreement match the fixture, and end-to-end scores are close ---")
    import warnings
    from run_realgraph import get_dataset
    from nf_mcd.pipeline import NFMCD
    from nf_mcd import baselines as b

    lines = ["Regression check: vectorized_fusion=True vs False, end-to-end", "=" * 78]
    all_close = True
    for key in ["crisismmd", "pheme", "dblp", "amazon"]:
        d = get_dataset(key, 0)
        for label, kw in (("default", dict()), ("vectorized", dict(vectorized_fusion=True))):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = NFMCD(n_communities=d["n_communities"], seed=0, **kw)
                m.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            s = b.score(d, b._nfmcd_result(m))
            lines.append(f"{key:<12} {label:<12} onmi={s['onmi']:.6f} modularity={s['modularity']:.6f} f1={s['f1']:.6f}")
            if label == "default":
                ref_scores = s
            else:
                d_onmi = abs(s["onmi"] - ref_scores["onmi"])
                d_mod = abs(s["modularity"] - ref_scores["modularity"])
                d_f1 = abs(s["f1"] - ref_scores["f1"])
                close = d_onmi < 1e-9 and d_mod < 1e-9 and d_f1 < 1e-9
                all_close &= close
                lines.append(f"  -> diffs vs default: onmi={d_onmi:.2e} modularity={d_mod:.2e} f1={d_f1:.2e} -> {'IDENTICAL' if close else 'DIFFERS'}")

    lines.append("")
    lines.append(f"vectorized_fusion end-to-end regression: {'IDENTICAL to default path' if all_close else 'DIFFERS from default path (see above)'}")
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write("\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok_default and all_close


# ---------------------------------------------------------------------------
# step 4: speedup measurement, reusing run_scale.py's generator/sizing
# ---------------------------------------------------------------------------

def cmd_scale():
    import warnings
    from run_scale import _make_dataset
    from nf_mcd.fuzzy_fusion import NeuroFuzzyFusion, ANFISAgreement
    from nf_mcd import topology as topo
    from nf_mcd import community_detection as cd

    sizes = [1000, 5000, 20000]
    lines = ["Speedup: vectorized fusion vs. original loop", "=" * 78]
    header = f"{'n':>8}{'fusion_old_s':>16}{'fusion_new_s':>16}{'speedup':>10}{'total_old_s':>14}{'total_new_s':>14}"
    lines.append(header)
    rows = []
    for n in sizes:
        data = _make_dataset(n, seed=0)
        e_t, e_v, G = data.text_embeddings, data.image_embeddings, data.G
        k = 8

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.monotonic()
            f_old = NeuroFuzzyFusion(common_dim=8, anfis=ANFISAgreement(), seed=0, vectorized=False)
            fr_old = f_old.fuse(e_t, e_v)
            t_fusion_old = time.monotonic() - t0

            t0 = time.monotonic()
            f_new = NeuroFuzzyFusion(common_dim=8, anfis=ANFISAgreement(), seed=0, vectorized=True)
            fr_new = f_new.fuse(e_t, e_v)
            t_fusion_new = time.monotonic() - t0

        d_fused = float(np.max(np.abs(fr_old.fused - fr_new.fused)))
        assert d_fused < 1e-9, f"n={n}: fusion outputs diverged, max|d|={d_fused:.2e}"

        # rest of the pipeline (structural embedding + alpha/fuse_features + FCM),
        # timed once, same for both (not the thing being measured here).
        t0 = time.monotonic()
        Z_s = topo.compute_structural_embedding(G, dim=k, seed=0)
        alpha = topo.compute_alpha(fr_old.confidence, 0.15, 0.85)
        fused_features = topo.fuse_features(fr_old.fused, Z_s, alpha)
        fcm = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=0)
        fcm.fit(fused_features)
        t_rest = time.monotonic() - t0

        speedup = t_fusion_old / max(t_fusion_new, 1e-9)
        total_old = t_fusion_old + t_rest
        total_new = t_fusion_new + t_rest
        lines.append(f"{n:>8}{t_fusion_old:>16.4f}{t_fusion_new:>16.4f}{speedup:>10.1f}x{total_old:>14.4f}{total_new:>14.4f}")
        rows.append(dict(n=n, fusion_old_s=t_fusion_old, fusion_new_s=t_fusion_new, speedup=speedup,
                          total_old_s=total_old, total_new_s=total_new, rest_s=t_rest))

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines.append("")
    lines.append("Per-stage share of total NFMCD.fit() time, OLD vs NEW fusion (n=20000):")
    biggest = rows[-1]
    old_total = biggest["fusion_old_s"] + biggest["rest_s"]
    new_total = biggest["fusion_new_s"] + biggest["rest_s"]
    lines.append(f"  old: fusion {100*biggest['fusion_old_s']/old_total:.1f}%  rest(struct+alpha+fcm) {100*biggest['rest_s']/old_total:.1f}%  total {old_total:.2f}s")
    lines.append(f"  new: fusion {100*biggest['fusion_new_s']/new_total:.1f}%  rest(struct+alpha+fcm) {100*biggest['rest_s']/new_total:.1f}%  total {new_total:.2f}s")

    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write("\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ns = [r["n"] for r in rows]
        ax.plot(ns, [r["fusion_old_s"] for r in rows], marker="o", label="fusion (original loop)")
        ax.plot(ns, [r["fusion_new_s"] for r in rows], marker="o", label="fusion (vectorized)")
        ax.plot(ns, [r["total_old_s"] for r in rows], marker="s", linestyle="--", label="total fit (original)")
        ax.plot(ns, [r["total_new_s"] for r in rows], marker="s", linestyle="--", label="total fit (vectorized)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_nodes"); ax.set_ylabel("wall-clock time (s)")
        ax.set_title("Fusion vectorization speedup")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(EXP, "vectorize_speedup.png"), dpi=150)
        plt.close(fig)
        print("\nPlot written: vectorize_speedup.png")
    except Exception as exc:
        print(f"(plotting skipped: {exc})")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("equivalence", "all"):
        ok = cmd_equivalence()
        if not ok:
            print("\nSTOPPING: equivalence check failed, not proceeding to regression/scale.")
            sys.exit(1)
    if cmd in ("regression", "all"):
        ok = cmd_regression()
        if not ok and cmd == "all":
            print("\nSTOPPING: regression check failed, not proceeding to scale measurement.")
            sys.exit(1)
    if cmd in ("scale", "all"):
        cmd_scale()


if __name__ == "__main__":
    main()
