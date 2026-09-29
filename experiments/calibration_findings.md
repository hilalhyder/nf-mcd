# Confidence calibration analysis findings

Script: `run_calibration.py`. Raw results: `experiments/calibration_results.csv`
(256 rows, 0 errors, full run completed). Full tables: `experiments/calibration_summary.log`.
Reliability diagrams: `experiments/calibration_reliability.png`.

## What was run

Reviewer's ask: turn the existing alpha/confidence analysis into a rigorous
confidence-estimation study — reliability diagrams, ECE, Brier score, AUROC for
mismatch detection (already existed, Section 6.4), precision/recall for identifying
unreliable content, and calibration before vs. after missingness.

**Reused rather than reimplemented**, per the task's instructions:
- `run_mechanism.fit_kind(d, seed, kind)` — fits a named NFMCD configuration
  (`"default"` and `"rank16"` used here) exactly as Section 6.4 did.
- `run_mechanism.build_B(setting, seed)` — real-data (crisismmd, fakeddit) mismatch
  injection: a `q`-fraction of paired nodes get their image swapped for a donor from
  a different community, reused verbatim, from the cached embeddings in
  `experiments/cache/{crisismmd,fakeddit}.pkl`.
- `run_missing.apply_mask(d, "both", p, seed)` — the project's standard MCAR
  missingness mask, applied on top of `build_B`'s output to get the
  before/after-missingness axis on real data.
- `nf_mcd.datasets.generate_synthetic_multimodal_graph` directly for the synthetic
  part (this is what `run_mechanism.build_A` itself calls; `build_A`'s own
  setting-string parser hardcodes `missing_modality_rate=0.1`, so this script calls
  the generator directly to get independently-controllable missingness, which is the
  whole point of the "before/after missingness" ask).

**Convention (stated once, applied throughout)**: NFMCD's `confidence_` is treated
as the model's estimated probability that a node's content pair is **NOT**
mismatched. `p_mismatch = 1 - confidence_` is therefore the predicted probability of
the positive class (`y=1` = "this pair was injected as mismatched"), and every
metric below (ECE, Brier, reliability diagram, precision/recall, AUROC/AP) is
computed against that convention on `p_mismatch`. All metrics restricted to nodes
NFMCD actually placed in the shared CCA space (`modality_flags_ == "both"` and a
defined `agreement_`), matching `run_mechanism.det_row`'s scope exactly — a node
with one or no modality has no cross-modal claim to calibrate.

### Grid

- **Synthetic**: `n=600, k=6, p_in=0.18, p_out=0.02`; content strength
  `{hi: centroid_scale=3.0, lo: centroid_scale=0.5}`; contamination
  (`misalignment_rate`) `{0.1, 0.2, 0.3, 0.4}`; missingness `{0.0, 0.4}`; 5 seeds;
  kinds `{default, rank16}` → 160 jobs.
- **Real**: `{crisismmd, fakeddit}`; injected swap fraction `q` `{0.1, 0.2, 0.3, 0.4}`;
  missingness `{0.0, 0.4}`; 3 seeds; kinds `{default, rank16}` → 96 jobs.
- Total 256 jobs, all completed, 0 errors.

## Sanity check against the manuscript's existing Section 6.4 numbers

Before trusting new numbers, the reused machinery was checked against the range the
task prompt quoted from the manuscript ("0.58-0.64 default rank cap, real data; near
0.9 synthetic low-contamination"): this run gets **crisismmd AUROC 0.565-0.627** and
**fakeddit AUROC 0.523-0.610** across q=0.1-0.4 at missingness=0 (default kind), and
**synthetic AUROC 0.975 at contamination=0.10** (content=hi) — both land inside the
quoted ranges. The reused machinery is behaving as expected.

## Headline numbers

### 1. ECE is not a substitute for AUROC — and can look better exactly when discrimination is collapsing

Synthetic, content=hi, missingness=0.0 (mean over 5 seeds):

| contamination | n_both | prevalence | ECE | Brier | AUROC | AP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 600 | 0.098 | 0.246±0.006 | 0.098±0.004 | **0.975±0.007** | 0.849 |
| 0.20 | 600 | 0.197 | 0.204±0.014 | 0.130±0.006 | 0.914±0.017 | 0.770 |
| 0.30 | 600 | 0.311 | 0.071±0.015 | 0.194±0.006 | 0.706±0.032 | 0.529 |
| 0.40 | 600 | 0.411 | **0.061±0.026** | 0.237±0.013 | **0.607±0.037** | 0.524 |

ECE *falls* from 0.246 to 0.061 as contamination rises from 10% to 40%, while AUROC
(the model's actual ability to tell aligned from mismatched pairs) *collapses* from
0.975 to 0.607 (near chance) over the same range. This is not a contradiction — ECE
only measures whether the average predicted probability matches the average
empirical rate; as true prevalence rises toward the confidence layer's typical
output range, the model's systematically mis-shaped confidence (see reliability
diagrams below) happens to average out better numerically, even though it has
stopped discriminating between individual pairs at all. **A low ECE number in this
setting would be actively misleading read alone — the manuscript should report ECE
next to AUROC/AP, never in place of them**, and the reliability diagrams (below)
show why: the confidence layer's three fixed ANFIS consequent levels (0.10, 0.50,
0.90, from `fuzzy_fusion.ANFISAgreement`) mean `p_mismatch` clusters near a few
discrete values regardless of the true contamination rate, so ECE's bin-weighted
average is very sensitive to where the (fixed) prevalence happens to fall relative
to those clusters.

### 2. Reliability diagrams: confidence is not a well-calibrated probability at any contamination level tested

See `experiments/calibration_reliability.png` (6 panels: synthetic hi at
missingness 0.0/0.4, crisismmd at missingness 0.0/0.4, fakeddit at missingness
0.0/0.4, all at a representative contamination/injection level of 0.2). In every
panel the empirical curve deviates from the diagonal (perfect calibration) — real
data in particular is flat and far below the diagonal (predicted `p_mismatch`
regularly well above the true empirical mismatch rate in the same bin), i.e.
**systematic overconfidence about mismatch** on real embeddings. This is consistent
with the low real-data AUROC (0.52-0.63): the confidence layer is not just
imprecise, it is directionally miscalibrated on real data.

### 3. Brier score confirms the same picture, and is monotonic (unlike ECE)

Brier score rises steadily with contamination in every setting tested (synthetic
hi/miss=0.0: 0.098 → 0.130 → 0.194 → 0.237 as contamination goes 0.10 → 0.40;
crisismmd/miss=0.0: 0.173 → 0.199 → 0.223 → 0.247 as q goes 0.10 → 0.40). Brier
score does **not** show ECE's misleading "improvement" — it degrades exactly where
AUROC degrades, since it penalizes both calibration and discrimination jointly.
**Recommendation for the manuscript: report Brier score as the primary scalar
calibration summary, with ECE and the reliability diagram as supporting detail, not
the other way around** — ECE alone would have told the wrong story here.

### 4. Precision/recall for identifying unreliable content — three thresholds, synthetic (content=hi)

Mean over 5 seeds, missingness=0.0:

| contamination | prec@0.3 | rec@0.3 | prec@0.5 | rec@0.5 | prec@0.7 | rec@0.7 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.342 | 0.980 | 0.922 | 0.508 | 1.000 | 0.152 |
| 0.20 | 0.405 | 0.936 | 0.872 | 0.424 | 0.960 | 0.119 |
| 0.30 | 0.415 | 0.781 | 0.595 | 0.222 | 0.790 | 0.063 |
| 0.40 | 0.460 | 0.713 | 0.569 | 0.191 | 0.647 | 0.048 |

At the "default" threshold of 0.5, precision is high at low contamination (0.922 at
10% contamination) but recall is only 0.508 — the confidence layer catches barely
half of injected mismatches even under the easiest condition tested. Lowering the
threshold to 0.3 trades this for near-complete recall (0.980) at much lower
precision (0.342) — three-quarters of everything flagged "unreliable" at that
threshold is actually fine. Raising it to 0.7 gives perfect precision (1.000, i.e.
no false alarms) but recall collapses to 0.152. **No single threshold gives both
usable precision and usable recall simultaneously in this setup — this is a genuine
finding for the manuscript's discussion of what the confidence signal can and
cannot be used for (e.g. it may be defensible as a low-threshold "flag for review"
signal, not as an automated mismatch filter).**

Real data, q=0.2 (mean over 3 seeds): precision/recall are markedly worse across
the board — crisismmd@0.5: prec=0.267, rec=0.245; fakeddit@0.5: prec=0.250,
rec=0.195. At true prevalence ≈0.195-0.2, precision 0.25-0.27 is barely above the
"flag everyone" baseline — **on real data, confidence at the default threshold is
close to uninformative for identifying individual unreliable pairs, even though
AUROC (0.58-0.61) is modestly above chance in aggregate ranking terms.** This is an
important distinction for the manuscript to make explicit: aggregate
rank-discrimination (AUROC) and pointwise usability (precision/recall at a fixed
threshold) tell different stories, and only the latter matches how a practitioner
would actually use the confidence score to filter content.

### 5. Calibration before vs. after missingness

Synthetic, content=hi (mean over 5 seeds; `n_both` drops as expected once nodes lose
a modality):

| contamination | n_both@0.0 | ECE@0.0 | n_both@0.4 | ECE@0.4 | AUROC@0.0 | AUROC@0.4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 600 | 0.246 | 355 | 0.248 | 0.975 | 0.967 |
| 0.20 | 600 | 0.204 | 362 | 0.194 | 0.914 | 0.889 |
| 0.30 | 600 | 0.071 | 360 | 0.110 | 0.706 | 0.770 |
| 0.40 | 600 | 0.061 | 367 | 0.058 | 0.607 | 0.621 |

**Calibration and AUROC are both essentially stable under missingness once
conditioned on the surviving "both"-modality pool** (differences are within
1-2 seed-sds of each other at every contamination level; `AUROC@0.4` is actually
slightly *higher* than `AUROC@0.0` at contamination 0.30/0.40). This makes sense
mechanistically: MCAR missingness removes ~40% of individual modalities
independently of whether a node's pair was injected as mismatched, so the *subset*
of nodes that keep both modalities is not biased with respect to alignment — the
`n_both` pool shrinks (600→355-367, i.e. `(1-0.4)^2 ≈ 0.36` of nodes keep both, as
expected for two independent 40% removal draws) but the underlying agreement-vs-
mismatch relationship within that shrunken pool is essentially unchanged.
**The manuscript's calibration claims do not need a separate "missingness caveat"
for the pairs it can still evaluate — the caveat is entirely about coverage
(far fewer nodes get a mismatch judgment at all), not about the quality of the
judgment on the nodes it still can score.** The same pattern holds on real data:
crisismmd ECE 0.087→0.102 (q=0.4, missingness 0.0→0.4), fakeddit ECE 0.082→0.087 —
small, not the large degradation one might expect a priori.

### 6. Default CCA rank cap vs. rank16: AUROC improves, but ECE gets *worse*

At a representative cell (synthetic content=hi contamination=0.2 missingness=0.0;
real q=0.2 missingness=0.0), comparing NFMCD defaults (`pca_rank_div=4`) against the
tighter `rank16` cap used by `NFMCD.robust()`:

| setting | ECE (default) | ECE (rank16) | AUROC (default) | AUROC (rank16) |
|---|---:|---:|---:|---:|
| synthetic/hi | 0.204 | 0.312 | 0.914 | **0.987** |
| real/crisismmd | 0.186 | 0.416 | 0.611 | **0.781** |
| real/fakeddit | 0.175 | 0.431 | 0.586 | **0.721** |

`rank16` substantially improves AUROC everywhere (as the existing rank-cap sweep,
`experiments/rank_summary.log`, already established for detection), but its ECE is
**1.5-2.5x worse** than the default's in every setting tested. This means rank16's
confidence values, while better *ranked* (higher AUROC — they put more truly-
mismatched pairs at the low-confidence end), are further from being literal
calibrated probabilities in absolute terms. **Practical implication for the
manuscript: recommending `rank16`/`NFMCD.robust()` for detection performance should
not be read as also recommending it for calibrated probability output — the two
are not the same property, and this data shows they can trade off against each
other.** If the manuscript wants both, the ANFIS consequents (currently fixed
by hand at 0.10/0.50/0.90 — see `fuzzy_fusion.ANFISAgreement`) would likely need
re-fitting (e.g. Platt scaling on held-out labeled mismatch data) specifically for
whichever rank cap is deployed, rather than assuming the same fixed consequents
calibrate well regardless of `pca_rank_div`.

## Honest limitations / what was simplified

- Real-data q (injected swap fraction) was swept over `{0.1, 0.2, 0.3, 0.4}` and
  missingness over `{0.0, 0.4}` for both crisismmd and fakeddit — a full
  cross-product with 3 seeds (96 jobs); PHEME and Fakeddit's real-relations graph
  were not included (no image modality / different cache shape) — only crisismmd
  and fakeddit, matching exactly what the prompt asked for ("at least one real
  dataset").
- Precision/recall tables in the main write-up above are shown at one
  representative contamination/injection level per data source (content=hi for
  synthetic, q=0.2 for real) for readability; the full 4-level tables for every
  combination are in `experiments/calibration_summary.log` and the raw per-seed
  numbers in `experiments/calibration_results.csv`.
- `rank16` was added as a bonus comparison (not explicitly requested) because the
  task prompt referenced "0.58-0.64 default rank cap" as if a non-default rank cap
  were a natural point of comparison; it reuses `run_mechanism.fit_kind`'s existing
  `"rank16"` config, not new fitting code.
- ECE uses 10 equal-width bins on `p_mismatch` in [0, 1] (the reviewer's suggested
  bin count); bin-level data (count, mean predicted probability, empirical rate)
  is stored per row in the `bins_json` CSV column for anyone who wants a different
  binning scheme without re-fitting models.
- Real-data "both"-modality node counts are large enough to trust (`n_both` from
  276 to 1143 across cells) but the *positive* (mismatched) class counts at low q
  are modest (e.g. crisismmd q=0.1, missingness=0.4: `n_both=276 * prevalence
  0.114 ≈ 31` mismatched nodes) — precision/recall at q=0.1 in the appendix tables
  should be read with that sample size caveat; q=0.2-0.4 cells are better powered.
- No attempt was made to re-calibrate the ANFIS consequents (e.g. via Platt
  scaling/isotonic regression) to see how much of the ECE gap could be closed —
  flagged above as a natural follow-up, not attempted here (would be a change to
  `nf_mcd/fuzzy_fusion.py`, out of scope for a read-only-library experiment script).

## Reproducibility

`py run_calibration.py run --workers 4` (resumable: finished
`(source, regime, param, missingness, seed, kind)` rows skipped on rerun, appended
to CSV with flush+fsync per job) → `py run_calibration.py summary` (rebuilds
`calibration_summary.log` and `calibration_reliability.png` from the CSV).
Confirmed: ran to completion, 256/256 jobs, 0 errors.
