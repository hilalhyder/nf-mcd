# Factor-sweep statistical testing — findings

Added after reviewer feedback that the original "spectral+content wins 40 of 50 grid
cells outright" headline used no formal significance test and no correction for
multiple comparisons, unlike the rest of the paper's statistical reporting.

## Method

`run_factor_sweep_stats.py` reads the existing `factor_sweep_results.csv` (no new
experiments run) and performs a paired t-test (paired over the 5 shared seeds) between
spectral+content and each of the two NF-MCD variants, independently in each of the 50
(regime, agreement, missingness) cells. Benjamini-Hochberg FDR correction is applied
across the 50 cells, separately for each NF-MCD variant comparison (100 tests total).

## Results

- vs fixed-alpha NF-MCD: spectral+content nominally higher in 46/50 cells; 41/50
  remain significant after BH correction (q<0.05). Fixed-alpha NF-MCD nominally higher
  in the remaining 4/50, none significant.
- vs adaptive-alpha NF-MCD (the default): spectral+content nominally higher in 46/50
  cells; 39/50 remain significant after BH correction. Adaptive-alpha NF-MCD nominally
  higher in the remaining 4/50, none significant (all four in the near-ceiling
  strong-structure/weak-content regime, raw p = 0.15-0.36 — noise, not signal).
- By regime (vs adaptive-alpha): weak-structure/strong-content is 25/25 significant;
  strong-structure/weak-content is only 14/25 significant, because spectral+content
  and structure-only are both near the ONMI ceiling there and their margin is often
  within noise. This matches the earlier manual check (quoting the paper's own
  Table 2 tie-rule: larger of 0.02 or summed per-cell SDs) that most of this regime
  would read as ties under that threshold.

## Conclusion

The original "40 of 50" headline undercounts slightly relative to a rigorous,
multiple-comparison-corrected test against the default (adaptive-alpha) NF-MCD
variant specifically (39/50), and overcounts slightly relative to the fixed-alpha
comparison (41/50) landing close by coincidence. The qualitative conclusion holds
and is now on firmer footing: neither NF-MCD variant has a single cell where it
significantly beats spectral+content, and the asymmetry between regimes (strong
general support in the weak-structure/strong-content regime, weak/noisy support in
the strong-structure/weak-content regime) is itself informative and now quantified
rather than asserted.
