"""
Label-independent-structure experiments for Fakeddit and CrisisMMD.

Usage (from nfmcd_impl/):
    py run_realgraph.py build            # sample Fakeddit relation graph, download/encode, cache
    py run_realgraph.py run [--workers N]
    py run_realgraph.py summary

Part 1: Fakeddit graph from real relations (author / domain / linked submission),
        as the full sampled graph ("fakeddit_real_full") and its largest
        components ("fakeddit_real_lcc").
Part 2: controlled homophily sweep (synthetic SBM over ground-truth groups,
        p_out/p_in in RATIOS, expected edge count equal to the original graph)
        for crisismmd and fakeddit. Synthetic sensitivity study, not a real graph.

Crash-safe: each finished job is appended to experiments/realgraph_results.csv
(flush+fsync); reruns skip finished jobs; other files are written temp+rename.
k = ground-truth community count, seeds 0,1,2, LFK ONMI + modularity + F1.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import json
import pickle
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "realgraph_results.csv")
SUMMARY_LOG = os.path.join(EXP, "realgraph_summary.log")
PROGRESS_MD = os.path.join(EXP, "realgraph_progress.md")
STATS_JSON = os.path.join(EXP, "realgraph_graph_stats.json")
FAKE_TSV = os.path.join(HERE, "data", "fakeddit", "multimodal_only_samples", "multimodal_train.tsv")
IMG_CACHE = os.path.join(HERE, "data", "fakeddit", "image_cache")

SEEDS = (0, 1, 2)
RATIOS = (0.02, 0.1, 0.25, 0.5, 1.0)
REAL = ["fakeddit_real_full", "fakeddit_real_lcc"]
SWEEP_BASE = ["crisismmd", "fakeddit"]
FIELDS = ["method", "dataset", "seed", "k_used", "modularity", "onmi", "f1", "note", "error", "secs"]

VARIANT_NAMES = ["default", "cons_a60", "agree_a50", "default_rank16", "all_changes"]
BASELINE_NAMES = ["louvain", "spectral_graph+content", "kmeans_content", "nfmcd_text_only", "nfmcd_structure_only"]
METHODS = [f"var:{v}" for v in VARIANT_NAMES] + [f"base:{b}" for b in BASELINE_NAMES]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Real-graph / homophily experiments progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished work is skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_realgraph.py build      # only if experiments/cache/fakeddit_real_*.pkl are missing\n"
        "py run_realgraph.py run --workers 4\n"
        "py run_realgraph.py summary\n"
        "```\n"
    ))


# ---------------------------------------------------------------------------
# Build: Fakeddit relation graph
# ---------------------------------------------------------------------------

def _download(url, timeout=5.0):
    import requests
    from io import BytesIO
    from PIL import Image
    path = os.path.join(IMG_CACHE, hashlib.sha256(url.encode("utf-8")).hexdigest() + ".jpg")
    if os.path.isfile(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research data collection; nf_mcd)"}, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(path, format="JPEG")
        return img
    except Exception:
        return None


def do_build():
    import networkx as nx
    import pandas as pd
    from nf_mcd import graphs as gr
    from nf_mcd.encoders import MultimodalEncoder

    os.makedirs(IMG_CACHE, exist_ok=True)
    if all(os.path.exists(os.path.join(CACHE, f"{n}.pkl")) for n in REAL) and os.path.exists(STATS_JSON):
        print("real-graph caches already exist; nothing to build")
        return

    df = pd.read_csv(FAKE_TSV, sep="\t")
    df = df.dropna(subset=["clean_title", "image_url", "subreddit", "author"])
    df = df[df["image_url"].astype(str).str.startswith("http")]
    top = df["subreddit"].value_counts().nlargest(10).index
    pool = df[df["subreddit"].isin(top)].reset_index(drop=True)

    sub, G, stats = gr.fakeddit_relation_sample(pool, n_target=1400, n_seeds=12, use_linked=True, seed=42)
    print("sample stats:", stats, flush=True)

    urls = sub["image_url"].astype(str).tolist()
    t0 = time.monotonic()
    images = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_download, u): i for i, u in enumerate(urls)}
        for n_done, fut in enumerate(as_completed(futs), 1):
            images[futs[fut]] = fut.result()
            if time.monotonic() - t0 > 1200:
                print("download time budget hit; remaining images left as missing", flush=True)
                break
    n_img = sum(1 for x in images if x is not None)
    print(f"images: {n_img}/{len(urls)} available ({time.monotonic() - t0:.0f}s)", flush=True)

    texts = sub["clean_title"].astype(str).tolist()
    enc = MultimodalEncoder()
    e_t, e_v = enc.encode(texts, images)
    print(f"encoded (text backend={enc.text_encoder._backend}, image backend={enc.image_encoder._backend})", flush=True)

    subs = sorted(sub["subreddit"].unique())
    sid = {s: i for i, s in enumerate(subs)}
    labels = [sid[s] for s in sub["subreddit"]]
    full = dict(G=G, e_t=e_t, e_v=e_v, true=[{l} for l in labels], n_communities=len(subs), names=subs)

    lcc_nodes = sorted(max(nx.connected_components(G), key=len))
    keep = lcc_nodes
    counts = defaultdict(int)
    for i in keep:
        counts[labels[i]] += 1
    keep = [i for i in keep if counts[labels[i]] >= 30]
    H = G.subgraph(keep)
    old2new = {o: n for n, o in enumerate(keep)}
    H = nx.relabel_nodes(H, old2new)
    lsubs = sorted({labels[i] for i in keep})
    lmap = {l: n for n, l in enumerate(lsubs)}
    lcc = dict(G=H, e_t=[e_t[i] for i in keep], e_v=[e_v[i] for i in keep],
               true=[{lmap[labels[i]]} for i in keep], n_communities=len(lsubs),
               names=[subs[l] for l in lsubs])

    lab_arr = np.array(labels)
    ledges = list(H.edges())
    lab_l = np.array([labels[i] for i in keep])
    stats.update(
        images_available=n_img, n_communities_full=len(subs), community_sizes_full={subs[i]: int((lab_arr == i).sum()) for i in range(len(subs))},
        lcc_nodes_kept=len(keep), lcc_edges=H.number_of_edges(), lcc_components=nx.number_connected_components(H),
        lcc_n_communities=len(lsubs), lcc_community_sizes={subs[l]: int((lab_l == l).sum()) for l in lsubs},
        lcc_homophily=float(np.mean([lab_l[u] == lab_l[v] for u, v in ledges])) if ledges else float("nan"),
    )
    atomic_write_pickle(os.path.join(CACHE, "fakeddit_real_full.pkl"), full)
    atomic_write_pickle(os.path.join(CACHE, "fakeddit_real_lcc.pkl"), lcc)
    atomic_write_text(STATS_JSON, json.dumps(stats, indent=2))
    print("build done:", json.dumps(stats, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Datasets (incl. SBM sweep)
# ---------------------------------------------------------------------------

_CACHE = {}


def _load_pkl(name):
    if name not in _CACHE:
        with open(os.path.join(CACHE, f"{name}.pkl"), "rb") as f:
            _CACHE[name] = pickle.load(f)
    return _CACHE[name]


def parse_ds(key):
    if "@r" in key:
        base, r = key.split("@r")
        return base, float(r)
    return key, None


def get_dataset(key, seed):
    base, ratio = parse_ds(key)
    d = dict(_load_pkl(base))
    if ratio is not None:
        from nf_mcd import graphs as gr
        groups = np.array([min(t) for t in d["true"]])
        m0 = d["G"].number_of_edges()
        n = len(groups)
        same = (groups[:, None] == groups[None, :])
        s_same = (np.triu(same, 1)).sum()
        s_diff = n * (n - 1) / 2 - s_same
        p_in = min(1.0, m0 / (s_same + ratio * s_diff))
        d["G"] = gr.sbm_graph(groups, p_in, ratio, seed=5000 + seed)
    return d


def edge_homophily(d):
    lab = np.array([min(t) for t in d["true"]])
    es = list(d["G"].edges())
    return float(np.mean([lab[u] == lab[v] for u, v in es])) if es else float("nan")


def run_one(job):
    method, key, seed = job
    t0 = time.monotonic()
    row = dict(method=method, dataset=key, seed=seed, k_used="", modularity=float("nan"),
               onmi=float("nan"), f1=float("nan"), note="", error="")
    try:
        from nf_mcd import baselines as b
        from nf_mcd.pipeline import NFMCD
        from run_variants import VARIANTS
        d = get_dataset(key, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kind, name = method.split(":", 1)
            if kind == "var":
                model = NFMCD(n_communities=d["n_communities"], seed=seed, **VARIANTS[name])
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = b._nfmcd_result(model)
            else:
                res = b.METHODS[name][0](d, seed)
            s = b.score(d, res)
        row.update(k_used=res.k_used, modularity=s["modularity"], onmi=s["onmi"], f1=s["f1"],
                   note=f"homophily={edge_homophily(d):.3f} edges={d['G'].number_of_edges()}")
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_datasets():
    return REAL + [f"{b}@r{r}" for b in SWEEP_BASE for r in RATIOS]


def all_jobs():
    return [(m, ds, s) for m in METHODS for ds in all_datasets() for s in SEEDS]


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "method":
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
            res[(row["method"], row["dataset"], row["seed"])] = row
    return res


def open_for_append():
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


def do_run(workers):
    for n in REAL:
        if not os.path.exists(os.path.join(CACHE, f"{n}.pkl")):
            sys.exit(f"missing cache {n}.pkl; run `py run_realgraph.py build` first")
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if j not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    write_progress("run", n_done, len(jobs), "running")
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(run_one, j) for j in pending]):
                row = fut.result()
                w.writerow([row[k] for k in FIELDS])
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if row["error"]:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  FAILED {row['method']}/{row['dataset']}/seed{row['seed']}: {row['error']}", flush=True)
                if n_done % 10 == 0:
                    write_progress("run", n_done, len(jobs), "running")
    finally:
        f.close()
    write_progress("run", n_done, len(jobs), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _ms(v):
    a = np.array(v, dtype=float)
    return f"{a.mean():.3f}Â±{a.std(ddof=0):.3f}"


def do_summary():
    res = load_results()
    agg = defaultdict(lambda: defaultdict(list))
    hom = defaultdict(list)
    for (m, ds, s), r in res.items():
        for k in ("modularity", "onmi", "f1"):
            agg[(m, ds)][k].append(r[k])
        note = r["note"]
        if "homophily=" in note:
            hom[ds].append(float(note.split("homophily=")[1].split()[0]))
    L = []
    if os.path.exists(STATS_JSON):
        st = json.load(open(STATS_JSON))
        L.append("Fakeddit real-relations graph (label-independent sampling; label used only as ground truth)")
        L.append("-" * 90)
        for k in ("nodes_sampled", "edges", "components", "lcc_nodes", "images_available",
                  "edges_author", "homophily_author", "edges_domain", "homophily_domain",
                  "edges_linked", "homophily_linked", "homophily_all", "homophily_random_baseline"):
            v = st.get(k)
            L.append(f"  {k:<28}{v:.3f}" if isinstance(v, float) else f"  {k:<28}{v}")
        L.append(f"  community sizes (full)       {st.get('community_sizes_full')}")
        L.append(f"  LCC subset: nodes={st.get('lcc_nodes_kept')} edges={st.get('lcc_edges')} "
                 f"components={st.get('lcc_components')} communities={st.get('lcc_n_communities')} "
                 f"homophily={st.get('lcc_homophily'):.3f}")
        L.append(f"  community sizes (LCC)        {st.get('lcc_community_sizes')}")
    L.append("")

    def table(metric, datasets, title):
        L.append(title)
        L.append("-" * len(title))
        L.append(f"{'method':<28}" + "".join(f"{d:>20}" for d in datasets))
        for m in METHODS:
            row = f"{m:<28}"
            for ds in datasets:
                row += f"{_ms(agg[(m, ds)][metric]):>20}" if (m, ds) in agg else f"{'missing':>20}"
            L.append(row)
        L.append("")

    for metric, label in (("onmi", "LFK ONMI"), ("modularity", "modularity"), ("f1", "membership F1")):
        table(metric, REAL, f"(a) Fakeddit real-relations graph: {label}")

    L.append("(b) Homophily sweep (synthetic SBM over ground-truth groups, equal expected edge count)")
    L.append("    realised edge homophily per ratio (mean over seeds):")
    for base in SWEEP_BASE:
        L.append("    " + base + ": " + "  ".join(
            f"r={r}:{np.mean(hom[f'{base}@r{r}']):.2f}" for r in RATIOS if hom.get(f"{base}@r{r}")))
    for base in SWEEP_BASE:
        ds_list = [f"{base}@r{r}" for r in RATIOS]
        table("onmi", ds_list, f"(b) {base}: LFK ONMI vs p_out/p_in (1.0 = structure carries no label information)")

    # (c) analysis helper: best method per dataset and NF-MCD default vs best baseline.
    L.append("(c) Best method per dataset by ONMI, and where full NF-MCD (var:default) ranks")
    for ds in all_datasets():
        cands = [(np.mean(agg[(m, ds)]["onmi"]), m) for m in METHODS if (m, ds) in agg]
        if not cands:
            continue
        cands.sort(reverse=True)
        rank = [m for _, m in cands].index("var:default") + 1
        best = cands[0]
        L.append(f"  {ds:<22} best={best[1]} ({best[0]:.3f})  var:default rank {rank}/{len(cands)} "
                 f"({np.mean(agg[('var:default', ds)]['onmi']):.3f})")
    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = load_results()
    agg = defaultdict(list)
    for (m, ds, s), r in res.items():
        agg[(m, ds)].append(r["onmi"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, base in zip(axes, SWEEP_BASE):
        for m in METHODS:
            ys = [np.mean(agg[(m, f"{base}@r{r}")]) if (m, f"{base}@r{r}") in agg else np.nan for r in RATIOS]
            ax.plot(RATIOS, ys, marker="o", label=m, linewidth=2 if m == "var:default" else 1.2)
        ax.set_xscale("log")
        ax.set_xlabel("p_out / p_in  (1.0 = structure has no label information)")
        ax.set_title(f"{base}: LFK ONMI vs structure informativeness (synthetic SBM)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LFK ONMI")
    axes[1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    out = os.path.join(EXP, "homophily_sweep.png")
    plt.savefig(out + ".tmp.png", dpi=140)
    os.replace(out + ".tmp.png", out)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "build":
        write_progress("build", 0, 1, "sampling, downloading, encoding")
        do_build()
        write_progress("build", 1, 1, "build complete; next: run")
    elif args.cmd == "run":
        do_run(args.workers)
        do_summary()
        do_plot()
    else:
        do_summary()
        do_plot()

