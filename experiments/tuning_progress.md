# Tuning progress

- Phase reached: **B**  (915/915 fits of this phase done)
- Updated: 2026-09-21 17:30:39
- phase fits complete

## Resume (finished fits are skipped automatically)

```
cd C:\Users\HP\Projects\Paper_SSC\nfmcd_impl
py tune_hyperparams.py cache      # only if experiments/cache/*.pkl are missing
py tune_hyperparams.py phaseA --workers 6
py tune_hyperparams.py phaseB --workers 6
```

Results: experiments/tuning_results.csv (append-only, fsynced per fit). Summary: experiments/tuning_summary.log.
