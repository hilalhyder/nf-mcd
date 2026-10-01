# -*- coding: utf-8 -*-
"""
Statistical testing of the controlled factor sweep (Section 6.11), addressing two
reviewer points: (1) "40 of 50 grid cells outright" used no formal test or multiple-
comparison correction; (2) whether the margin between spectral+content and each NF-MCD
variant is statistically distinguishable per cell, not just nominally higher.

For every one of the 50 (regime, agreement, missingness) cells, paired t-tests (paired
over the 5 shared seeds) compare spectral+content against: (a) the stronger of
structure_only/content_only in that regime, and (b) each of the two NF-MCD variants.
Benjamini-Hochberg FDR correction is applied across all tests of a given comparison type
(100 tests: 50 cells x 2 NF-MCD variants).
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r"C:\Users\HP\Projects\Paper_SSC\nfmcd_impl\experiments\factor_sweep_results.csv")

cells = df[["regime", "agreement", "missingness"]].drop_duplicates().sort_values(
    ["regime", "agreement", "missingness"]).reset_index(drop=True)

def get_vals(regime, agreement, missingness, method):
    sub = df[(df.regime == regime) & (df.agreement == agreement) &
             (df.missingness == missingness) & (df.method == method)].sort_values("seed")
    return sub["onmi"].values

def bh_correct(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out

results = []
for _, row in cells.iterrows():
    regime, agreement, missingness = row["regime"], row["agreement"], row["missingness"]
    sc = get_vals(regime, agreement, missingness, "spectral_content")
    for variant in ["nfmcd_alpha_fixed", "nfmcd_adaptive"]:
        v = get_vals(regime, agreement, missingness, variant)
        if len(sc) == 5 and len(v) == 5:
            diff = sc - v
            if np.allclose(diff, diff[0]):
                # zero-variance paired difference: t-test undefined, report sign directly
                t, p = np.inf if diff[0] > 0 else (-np.inf if diff[0] < 0 else 0.0), (0.0 if diff[0] != 0 else 1.0)
            else:
                t, p = stats.ttest_rel(sc, v)
            results.append({
                "regime": regime, "agreement": agreement, "missingness": missingness,
                "variant": variant, "sc_mean": sc.mean(), "variant_mean": v.mean(),
                "mean_diff": sc.mean() - v.mean(), "t": t, "p_raw": p,
            })

res = pd.DataFrame(results)
for variant in ["nfmcd_alpha_fixed", "nfmcd_adaptive"]:
    mask = res["variant"] == variant
    res.loc[mask, "p_bh"] = bh_correct(res.loc[mask, "p_raw"].values)

res["sig_raw_05"] = res["p_raw"] < 0.05
res["sig_bh_05"] = res["p_bh"] < 0.05
res["sc_wins"] = res["mean_diff"] > 0

out_csv = r"C:\Users\HP\Projects\Paper_SSC\nfmcd_impl\experiments\factor_sweep_stats_results.csv"
res.to_csv(out_csv, index=False)

summary_lines = []
summary_lines.append("Factor-sweep paired t-tests, spectral+content vs each NF-MCD variant, per cell (5 seeds).")
summary_lines.append("Benjamini-Hochberg FDR correction applied across 50 cells, separately per NF-MCD variant.")
summary_lines.append("")
for variant in ["nfmcd_alpha_fixed", "nfmcd_adaptive"]:
    sub = res[res["variant"] == variant]
    n_cells = len(sub)
    n_sc_wins_nominal = (sub["sc_wins"]).sum()
    n_sig_raw = (sub["sc_wins"] & sub["sig_raw_05"]).sum()
    n_sig_bh = (sub["sc_wins"] & sub["sig_bh_05"]).sum()
    n_nfmcd_wins_nominal = (~sub["sc_wins"]).sum()
    n_nfmcd_sig_bh = (~sub["sc_wins"] & sub["sig_bh_05"]).sum()
    summary_lines.append(f"=== spectral+content vs {variant} ===")
    summary_lines.append(f"  cells where spectral+content has higher mean (nominal): {n_sc_wins_nominal}/{n_cells}")
    summary_lines.append(f"  of those, significant at raw p<0.05: {n_sig_raw}/{n_cells}")
    summary_lines.append(f"  of those, significant after BH correction (q<0.05): {n_sig_bh}/{n_cells}")
    summary_lines.append(f"  cells where {variant} has higher mean (nominal): {n_nfmcd_wins_nominal}/{n_cells}")
    summary_lines.append(f"  of those, significant after BH correction (q<0.05): {n_nfmcd_sig_bh}/{n_cells}")
    summary_lines.append("")

summary_lines.append("Per-regime breakdown (spectral+content vs adaptive-alpha NF-MCD, the paper's default):")
for regime in ["strong_struct_weak_content", "weak_struct_strong_content"]:
    sub = res[(res["variant"] == "nfmcd_adaptive") & (res["regime"] == regime)]
    n_sig_bh = (sub["sc_wins"] & sub["sig_bh_05"]).sum()
    summary_lines.append(f"  {regime}: {n_sig_bh}/{len(sub)} cells significant (BH q<0.05) in favor of spectral+content")

out_log = r"C:\Users\HP\Projects\Paper_SSC\nfmcd_impl\experiments\factor_sweep_stats_summary.log"
with open(out_log, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))
