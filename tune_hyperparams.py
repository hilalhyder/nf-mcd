"""
Hyperparameter sensitivity + leave-one-dataset-out (LODO) tuning for NF-MCD.

Usage (from nfmcd_impl/):
    py tune_hyperparams.py cache            # encode once, cache to experiments/cache/
    py tune_hyperparams.py phaseA [--workers N]
    py tune_hyperparams.py phaseB [--workers N] [--n-random 60]

Crash-safe and resumable: every finished fit is appended (and fsynced) to
experiments/tuning_results.csv immediately; on restart, fits already in the
CSV are skipped. Cached files/summaries are written temp-then-rename.
See experiments/tuning_progress.md for the resume command.

Protocol: k fixed to the number of ground-truth communities; seeds 0,1,2;
primary metric LFK ONMI (nf_mcd.metrics.overlapping_nmi), overlap threshold
fixed at 0.2 (not tuned). PHEME is evaluated and reported but excluded from
the selection score.
"""
from __future__ import annotations

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] != "cache":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
SUMMARY_LOG = os.path.join(EXP, "tuning_summary.log")
RESULTS_CSV = os.path.join(EXP, "tuning_results.csv")
PROGRESS_MD = os.path.join(EXP, "tuning_progress.md")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
CONTENT = {"crisismmd", "pheme", "fakeddit"}
SELECTION = ["crisismmd", "fakeddit", "dblp", "amazon"]   # PHEME excluded
SEEDS = (0, 1, 2)

DEFAULT = dict(m=1.5, alpha=(0.15, 0.85), sd=1.0, cd=8, anfis="default")
GRID = dict(
    m=[1.2, 1.3, 1.5, 1.75, 2.0],
    alpha=[(0.05, 0.95), (0.15, 0.85), (0.25, 0.75)],
    sd=[0.5, 1.0, 1.5, 2.0],
    cd=[4, 8, 16, 32],
    anfis=["default", "shift-0.1", "shift+0.1", "scale0.6", "scale1.5", "pct"],
)
APPLIES = {
    "m": set(DATASETS), "sd": set(DATASETS),
    "alpha": CONTENT, "cd": CONTENT, "anfis": CONTENT,
}

FIELDS = ["dataset", "seed", "phase", "param", "cfg", "m", "alpha_min", "alpha_max", "sd_mult",
          "common_dim", "anfis", "modularity", "onmi", "f1", "error", "secs"]


# --------------------------------------------------------------------------
# Crash-safe I/O helpers
# --------------------------------------------------------------------------

def atomic_write_bytes(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text: str):
    atomic_write_bytes(path, text.encode("utf-8"))


def log(msg=""):
    print(msg, flush=True)
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Tuning progress\n\n"
        f"- Phase reached: **{phase}**  ({done}/{total} fits of this phase done)\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished fits are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py tune_hyperparams.py cache      # only if experiments/cache/*.pkl are missing\n"
        "py tune_hyperparams.py phaseA --workers 6\n"
        "py tune_hyperparams.py phaseB --workers 6\n"
        "```\n\n"
        "Results: experiments/tuning_results.csv (append-only, fsynced per fit). "
        "Summary: experiments/tuning_summary.log.\n"
    ))


def cfg_key(c):
    return json.dumps(c, sort_keys=True, default=list)


def load_results():
    """Read completed (error-free) fits from the CSV, tolerating a torn last line."""
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "dataset":
                continue
            row = dict(zip(FIELDS, r))
            if row["error"]:
                continue
            try:
                for k in ("modularity", "onmi", "f1"):
                    row[k] = float(row[k])
                row["seed"] = int(row["seed"])
            except ValueError:
                continue
            res[(row["dataset"], row["cfg"], row["seed"])] = row
    return res


def open_results_for_append():
    new = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not new:
        with open(RESULTS_CSV, "rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_nl = f.read(1) == b"\n"
        if not ends_nl:
            with open(RESULTS_CSV, "ab") as f:
                f.write(b"\n")
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(FIELDS)
        f.flush()
        os.fsync(f.fileno())
    return f, w


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

def cache_path(name):
    return os.path.join(CACHE, f"{name}.pkl")


def do_cache():
    from nf_mcd import datasets as ds_mod
    from nf_mcd.encoders import MultimodalEncoder

    os.makedirs(CACHE, exist_ok=True)
    loaders = {
        "crisismmd": lambda: ds_mod.load_crisismmd(max_nodes=800),
        "pheme": lambda: ds_mod.load_pheme(max_nodes=2000),
        "fakeddit": lambda: ds_mod.load_fakeddit(max_nodes=1200),
        "dblp": lambda: ds_mod.load_snap_community("dblp", n_communities=8),
        "amazon": lambda: ds_mod.load_snap_community("amazon", n_communities=8, min_community_size=15),
    }
    encoder = None
    for name in DATASETS:
        if os.path.exists(cache_path(name)):
            log(f"[cache] {name}: already cached, skipping")
            continue
        t0 = time.monotonic()
        data = loaders[name]()
        if name in CONTENT:
            if encoder is None:
                encoder = MultimodalEncoder()
            e_t, e_v = encoder.encode(data.texts, data.images)
        else:
            n = data.G.number_of_nodes()
            e_t, e_v = [None] * n, [None] * n
        payload = dict(G=data.G, e_t=e_t, e_v=e_v, true=data.true_communities,
                       n_communities=data.n_communities)
        atomic_write_bytes(cache_path(name), pickle.dumps(payload))
        n_t = sum(v is not None for v in e_t)
        n_v = sum(v is not None for v in e_v)
        log(f"[cache] {name}: {data.G.number_of_nodes()} nodes, {data.G.number_of_edges()} edges, "
            f"k={data.n_communities}, text emb={n_t}, image emb={n_v}, {time.monotonic() - t0:.0f}s")


_CACHE_MEM = {}
_PCT_MEM = {}


def load_cache(name):
    if name not in _CACHE_MEM:
        with open(cache_path(name), "rb") as f:
            _CACHE_MEM[name] = pickle.load(f)
    return _CACHE_MEM[name]


# --------------------------------------------------------------------------
# One evaluation
# --------------------------------------------------------------------------

def pct_centers(name):
    """Label-free ANFIS centers: 20/50/80th percentiles of observed agreement (default fit, seed 0)."""
    if name not in _PCT_MEM:
        from nf_mcd import NFMCD
        d = load_cache(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NFMCD(n_communities=d["n_communities"], seed=0)
            m.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
        a = m.agreement_[np.isfinite(m.agreement_)]
        _PCT_MEM[name] = np.percentile(a, [20, 50, 80]) if len(a) else None
    return _PCT_MEM[name]


def build_anfis(tag, name):
    from nf_mcd.fuzzy_fusion import ANFISAgreement
    a = ANFISAgreement()
    if tag == "shift-0.1":
        a.centers = a.centers - 0.1
    elif tag == "shift+0.1":
        a.centers = a.centers + 0.1
    elif tag == "scale0.6":
        a.widths = a.widths * 0.6
    elif tag == "scale1.5":
        a.widths = a.widths * 1.5
    elif tag == "pct":
        c = pct_centers(name)
        if c is not None:
            a.centers = np.asarray(c, dtype=float)
    return a


def run_one(job):
    ds, cfg, seed, phase, param = job
    t0 = time.monotonic()
    row = dict(dataset=ds, seed=seed, phase=phase, param=param, cfg=cfg_key(cfg),
               m=cfg["m"], alpha_min=cfg["alpha"][0], alpha_max=cfg["alpha"][1],
               sd_mult=cfg["sd"], common_dim=cfg["cd"], anfis=cfg["anfis"],
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"), error="")
    try:
        from nf_mcd import NFMCD
        d = load_cache(ds)
        k = d["n_communities"]
        model = NFMCD(
            n_communities=k, common_dim=cfg["cd"], structural_dim=max(2, int(round(cfg["sd"] * k))),
            fcm_m=cfg["m"], alpha_min=cfg["alpha"][0], alpha_max=cfg["alpha"][1],
            anfis=build_anfis(cfg["anfis"], ds), seed=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            s = model.evaluate(true_communities_per_node=d["true"])
        row.update(modularity=s["modularity"], onmi=s["overlapping_nmi"], f1=s["membership_f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def run_jobs(jobs, workers, phase_name):
    """Run only fits missing from the CSV; append + fsync each row as it finishes."""
    done = load_results()
    pending = [j for j in jobs if (j[0], cfg_key(j[1]), j[2]) not in done]
    total = len(jobs)
    log(f"  {total - len(pending)}/{total} fits already in CSV; running {len(pending)}")
    if not pending:
        return
    f, w = open_results_for_append()
    n_done = total - len(pending)
    n_err = 0
    t0 = time.monotonic()
    write_progress(phase_name, n_done, total, "running")
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, j) for j in pending]
            for fut in as_completed(futs):
                row = fut.result()
                w.writerow([row[k] for k in FIELDS])
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if row["error"]:
                    n_err += 1
                    if n_err <= 3:
                        log(f"  fit failed ({row['dataset']}, seed {row['seed']}): {row['error']}")
                if n_done % 25 == 0:
                    write_progress(phase_name, n_done, total, "running")
    finally:
        f.close()
    write_progress(phase_name, n_done, total, "phase fits complete")
    log(f"  ran {len(pending)} fits in {time.monotonic() - t0:.0f}s with {workers} workers ({n_err} errors)")


def cfg_with(**kw):
    c = dict(DEFAULT)
    c.update(kw)
    return c


def scores(res, ds, cfg, field="onmi"):
    vals = [res[(ds, cfg_key(cfg), s)][field] for s in SEEDS if (ds, cfg_key(cfg), s) in res]
    return np.array(vals, dtype=float)


def mu_sd(a):
    return (float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0) if len(a) else (float("nan"), float("nan"))


# --------------------------------------------------------------------------
# Phase A: one-at-a-time sensitivity
# --------------------------------------------------------------------------

def phase_a(workers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log("=" * 78)
    log("PHASE A: one-at-a-time sensitivity around defaults")
    log("=" * 78)
    jobs = [(ds, cfg_with(), s, "A", "default") for ds in DATASETS for s in SEEDS]
    for param, values in GRID.items():
        for v in values:
            if v == DEFAULT[param]:
                continue
            for ds in DATASETS:
                if ds in APPLIES[param]:
                    jobs += [(ds, cfg_with(**{param: v}), s, "A", param) for s in SEEDS]
    run_jobs(jobs, workers, "A")
    res = load_results()

    base = {ds: mu_sd(scores(res, ds, cfg_with())) for ds in DATASETS}
    log("\nDefault-config ONMI (mean +/- sd over seeds 0,1,2):")
    for ds in DATASETS:
        log(f"  {ds:<10} {base[ds][0]:.4f} +/- {base[ds][1]:.4f}")

    mattered, max_delta = {}, {}
    for param, values in GRID.items():
        log(f"\n--- {param} ---")
        best = 0.0
        for ds in DATASETS:
            if ds not in APPLIES[param]:
                continue
            parts = []
            for v in values:
                if v == DEFAULT[param]:
                    parts.append(f"{v}*: {base[ds][0]:.3f}+/-{base[ds][1]:.3f}")
                    continue
                mu, sd = mu_sd(scores(res, ds, cfg_with(**{param: v})))
                parts.append(f"{v}: {mu:.3f}+/-{sd:.3f}")
                if ds != "pheme":
                    delta = abs(mu - base[ds][0])
                    best = max(best, delta)
                    if delta > max(0.02, 2 * max(sd, base[ds][1])):
                        mattered.setdefault(param, []).append(f"{ds}({v}: {mu - base[ds][0]:+.3f})")
            log(f"  {ds:<10} " + " | ".join(parts))
        max_delta[param] = best

    log("\nRule: a parameter 'mattered' if on some non-PHEME dataset a value moved mean ONMI by more than "
        "max(0.02, 2*sd) from the default.")
    log("Mattered: " + ("; ".join(f"{p}: {', '.join(v)}" for p, v in mattered.items()) or "none"))
    kept = list(mattered)
    if len(kept) < 2:
        extra = [p for p in sorted(max_delta, key=max_delta.get, reverse=True) if p not in kept]
        kept += extra[: 2 - len(kept)]
        log(f"Fewer than 2 passed; also keeping the largest-|delta| parameters for Phase B.")
    atomic_write_text(os.path.join(EXP, "tuning_phaseA_kept.json"), json.dumps(kept))
    log(f"Kept for Phase B: {kept}")

    for param, values in GRID.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for ds in DATASETS:
            if ds not in APPLIES[param]:
                continue
            mus, sds = [], []
            for v in values:
                mu, sd = base[ds] if v == DEFAULT[param] else mu_sd(scores(res, ds, cfg_with(**{param: v})))
                mus.append(mu); sds.append(sd)
            ax.errorbar(range(len(values)), mus, yerr=np.nan_to_num(sds), marker="o", capsize=3, label=ds)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=20)
        ax.set_xlabel(f"{param} (default = {DEFAULT[param]})")
        ax.set_ylabel("LFK ONMI (mean +/- sd, 3 seeds)")
        ax.set_title(f"Sensitivity: {param}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(EXP, f"tuning_sensitivity_{param}.png")
        fig.savefig(out + ".tmp", dpi=130, format="png")
        os.replace(out + ".tmp", out)
        plt.close(fig)
    log("Saved experiments/tuning_sensitivity_<param>.png")


# --------------------------------------------------------------------------
# Phase B: random search + leave-one-dataset-out selection
# --------------------------------------------------------------------------

def phase_b(workers, n_random):
    log("\n" + "=" * 78)
    log("PHASE B: random search + leave-one-dataset-out (LODO) selection")
    log("=" * 78)
    with open(os.path.join(EXP, "tuning_phaseA_kept.json")) as f:
        kept = json.load(f)
    log(f"Searching over: {kept}")
    rng = np.random.default_rng(0)
    configs = [cfg_with()]
    seen = {cfg_key(configs[0])}
    tries = 0
    while len(configs) < n_random + 1 and tries < 10000:
        tries += 1
        c = cfg_with()
        for p in kept:
            c[p] = GRID[p][int(rng.integers(len(GRID[p])))]
        if cfg_key(c) not in seen:
            seen.add(cfg_key(c))
            configs.append(c)
    log(f"{len(configs)} configs (default + {len(configs) - 1} random) x {len(DATASETS)} datasets x {len(SEEDS)} seeds")

    jobs = [(ds, c, s, "B", "search") for c in configs for ds in DATASETS for s in SEEDS]
    run_jobs(jobs, workers, "B")
    res = load_results()

    M = {(i, ds): mu_sd(scores(res, ds, c)) for i, c in enumerate(configs) for ds in DATASETS}

    log("\nLODO selection (score = mean ONMI over the other non-PHEME datasets; default is config 0)")
    header = f"{'held-out':<11}{'chosen cfg':>11}{'tuned':>16}{'default':>16}{'delta':>9}{'2*noise':>9}  verdict"
    log(header)
    log("-" * len(header))
    deltas, chosen = [], {}
    for h in SELECTION:
        others = [d for d in SELECTION if d != h]
        sc = [np.mean([M[(i, o)][0] for o in others]) for i in range(len(configs))]
        best = int(np.argmax(sc))
        chosen[h] = best
        (t_mu, t_sd), (d_mu, d_sd) = M[(best, h)], M[(0, h)]
        delta = t_mu - d_mu
        noise2 = 2 * float(np.sqrt(t_sd ** 2 + d_sd ** 2))
        thr = max(noise2, 0.02)
        verdict = ("default selected" if best == 0 else "tuned better" if delta > thr
                   else "tuned WORSE" if delta < -thr else "within noise")
        deltas.append(delta)
        log(f"{h:<11}{best:>11}{t_mu:>10.3f}+/-{t_sd:.3f}{d_mu:>10.3f}+/-{d_sd:.3f}{delta:>+9.3f}{noise2:>9.3f}  {verdict}")
        log(f"            cfg {best}: {configs[best]}")
    log(f"\nMean held-out delta (tuned - default) over {len(SELECTION)} datasets: {np.mean(deltas):+.4f}")

    log("\nPHEME (not used in selection) for each chosen config vs default:")
    for h, b in chosen.items():
        log(f"  chosen for {h:<10}: cfg {b}: PHEME ONMI {M[(b, 'pheme')][0]:.3f}+/-{M[(b, 'pheme')][1]:.3f} "
            f"(default {M[(0, 'pheme')][0]:.3f}+/-{M[(0, 'pheme')][1]:.3f})")
    log(f"  best PHEME ONMI over all {len(configs)} configs: {max(M[(i, 'pheme')][0] for i in range(len(configs))):.3f}")

    ins = sorted(range(len(configs)), key=lambda i: -np.mean([M[(i, d)][0] for d in SELECTION]))[:5]
    dflt = np.mean([M[(0, d)][0] for d in SELECTION])
    log("\nTop 5 configs by mean ONMI over the 4 selection datasets (in-sample, NOT held-out):")
    for i in ins:
        log(f"  cfg {i}: {np.mean([M[(i, d)][0] for d in SELECTION]):.4f}  (default {dflt:.4f})  {configs[i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "phaseA", "phaseB"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    t0 = time.monotonic()
    if args.cmd == "cache":
        write_progress("cache", 0, 5, "caching encodings")
        do_cache()
        write_progress("cache done", 5, 5, "cache complete; next: phaseA")
    elif args.cmd == "phaseA":
        phase_a(args.workers)
    else:
        phase_b(args.workers, args.n_random)
    log(f"[{args.cmd}] total {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
