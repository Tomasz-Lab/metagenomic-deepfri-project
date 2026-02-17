import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from collections import defaultdict
    import re
    from glob import glob
    import importlib
    import random
    from functools import lru_cache
    import subprocess

    try:
        from IPython import get_ipython

        _ip = get_ipython()
        if _ip is not None:
            _ip.run_line_magic("load_ext", "autoreload")
            _ip.run_line_magic("autoreload", "2")
    except Exception:
        _ip = None

    # save and display plots in whitemode
    import matplotlib.style

    matplotlib.style.use("default")
    return importlib, lru_cache, mo, np, pd, plt, re, subprocess


@app.cell
def _(importlib):
    import helpers.gc_benchmark_helpers as gch

    importlib.reload(gch)
    return (gch,)


@app.cell
def _(importlib):
    import helpers.cmap_load_helpers as clh

    importlib.reload(clh)
    return (clh,)


@app.cell
def _(importlib):
    import helpers.visualization_helpers as vh

    importlib.reload(vh)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # mmseqs2 results analysis
    """)
    return


@app.cell
def _(pd):
    ### data loading ###
    _landscape_path = "data/source/landscape"

    # mmseqs2 results from mdeepfri (full afdb_v4 search)
    afdb_v4_full = pd.read_csv(
        f"{_landscape_path}/mmseqs2_search/afdb_uniprot_v4_results.tsv",
        sep="\t",
    )

    full_query_fasta = f"{_landscape_path}/landscape_afdb_50k.fasta"

    ground_truth_cmaps_dir = f"{_landscape_path}/sample_50k_cmaps"

    results_cmaps_dir = f"{_landscape_path}/results"

    pyopal_root = f"{_landscape_path}/pyopal_alignments"

    true_struct_root = f"{_landscape_path}/sample_50k_structures/"

    hit_struct_root = f"{_landscape_path}/structures"
    return (
        afdb_v4_full,
        full_query_fasta,
        ground_truth_cmaps_dir,
        hit_struct_root,
        pyopal_root,
        results_cmaps_dir,
        true_struct_root,
    )


@app.cell
def _(np):
    ### parameters ###
    n_all_queries = 50000

    generated_contacts = [0, 1, 2, 3, 4]

    all_bins = np.arange(0.1, 1.01, 0.1)
    return all_bins, n_all_queries


@app.cell
def _(all_bins):
    all_bins
    return


@app.cell
def _(afdb_v4_full, n_all_queries, np, plt):
    ### count how many self hits ###
    # in the analysis we exclude self-hits
    self_hits = afdb_v4_full[afdb_v4_full["query"] == afdb_v4_full["target"]]
    afdb_no_self = afdb_v4_full[
        afdb_v4_full["query"] != afdb_v4_full["target"]
    ].copy()

    # how many seqs have any hits and how many are only self-hits
    seq_any_hit = afdb_v4_full["query"].nunique()
    seq_only_self_hit = (
        afdb_v4_full["query"].nunique() - afdb_no_self["query"].nunique()
    )
    seq_good_hits = afdb_no_self["query"].nunique()

    print(
        f"Out of {n_all_queries} sequences, {seq_any_hit} have any hit in the afdb_v4\n",
        f"{seq_only_self_hit} sequences have only self-hits\n",
        f"{seq_good_hits} sequences have hits to other sequences in afdb_v4",
    )

    # fident_self should be a list/array of your self-hit fident values
    fident_self = self_hits["fident"].values
    fident = np.array(fident_self)

    # jitter for scatter
    x_scatter = np.ones_like(fident) + np.random.uniform(
        -0.4, 0.4, size=len(fident)
    )

    # Filter for values < 1.0
    fident_lt1 = fident[fident < 1.0]
    x_scatter_lt1 = np.ones_like(fident_lt1) + np.random.uniform(
        -0.4, 0.4, size=len(fident_lt1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # --- Subplot 1: Full distribution ---
    axes[0].boxplot(fident, positions=[1], showfliers=False)
    axes[0].scatter(x_scatter, fident, alpha=0.01)
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["Self-hits"])
    axes[0].set_ylabel("fident")
    axes[0].set_title("Full fident distribution")

    # --- Subplot 2: fident < 1.0 ---
    axes[1].boxplot(fident_lt1, positions=[1], showfliers=False)
    axes[1].scatter(x_scatter_lt1, fident_lt1, alpha=0.01)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Nonidentical self-hits"])
    axes[1].set_title("fident values < 1.0")

    plt.tight_layout()
    plt.savefig(
        "plots/landscape/self_hit_fident.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    return


@app.cell
def _(full_query_fasta):
    ### load the fasta with original queries ###
    # lets load the full list of query sequences
    fasta_ids = []
    with open(full_query_fasta) as _f:
        for _line in _f:
            if _line.startswith(">"):
                fasta_ids.append(_line[1:].strip())

    fasta_ids_unique = set(fasta_ids)

    print(
        f"Loaded fasta with {len(fasta_ids)} IDs\n",
        f"Out of these, {len(fasta_ids_unique)} are unique",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## CMAP ANALYSIS
    ### Gather all paths and results, takes like 30 minutes and loads of RAM
    """)
    return


@app.cell
def _(
    afdb_v4_full,
    all_bins,
    clh,
    gch,
    ground_truth_cmaps_dir,
    hit_struct_root,
    pd,
    pyopal_root,
    results_cmaps_dir,
    true_struct_root,
):
    # 0) Prepare mmseqs2 results as a results basket
    main_results_df = afdb_v4_full.query("query != target")

    idbin_labels = [
        f"{all_bins[i]:.2f}-{all_bins[i + 1]:.2f}"
        for i in range(len(all_bins) - 1)
    ]
    # ['0.00-0.10', '0.10-0.20', ..., '0.90-1.00'] like in directory names

    main_results_df["idbin"] = pd.cut(
        main_results_df["fident"],
        bins=all_bins,
        labels=idbin_labels,
        include_lowest=True,
        right=False,  # [a,b) intervals
    ).astype(str)

    # 1) Prepare df with pyopal results
    pyopal_df = gch.parse_pyopal_to_df(pyopal_root)
    # drop hits with perfect identity over the entire sequence
    pyopal_df = pyopal_df.query(
        "not (pyopal_identity == 1 and pyopal_coverage == 1)"
    )

    # 2) Merge pyopal results with mmseqs2 results
    main_results_df = main_results_df.merge(
        pyopal_df,
        on=["idbin", "query", "target"],
        how="inner",
    )

    # 3) Add cmaps
    cmap_paths_by_pid, gc_values = clh.discover_pred_cmaps(results_cmaps_dir)
    main_results_df = clh.add_cmap_path_columns(
        main_results_df,
        cmap_paths_by_pid,
        gc_values,
        query_col="query",
        idbin_col="idbin",
    )
    ground_truth_paths = clh.discover_ground_truth_cmap_paths(
        ground_truth_cmaps_dir
    )
    main_results_df["true_cmap_path"] = main_results_df["query"].map(
        ground_truth_paths
    )
    # 4) Add structures
    true_struct_paths = gch.discover_true_structures(true_struct_root)
    hit_struct_paths = gch.discover_hit_structures(hit_struct_root)

    main_results_df = gch.add_structure_paths_to_df(
        main_results_df,
        true_struct_paths,
        hit_struct_paths,
    )

    # 5) Load actual cmaps and calculate f1 scores and prec and recc
    main_results_df = gch.add_prf1_columns_from_paths(main_results_df, gc_values)

    # 6) Calculate match/gap-specific f1 scores
    main_results_df = gch.add_prf1_with_masks_from_paths(
        main_results_df, gc_values
    )
    return (main_results_df,)


@app.cell
def _(main_results_df):
    main_results_df
    return


@app.cell
def _(lru_cache, np, pd, re, subprocess):
    ### Calculate TM-scores ###
    USALIGN_BIN = "/home/FilipS/software/USalign/USalign"

    _tm_regex = re.compile(r"TM-score\s*=\s*([0-9.]+)")


    @lru_cache(None)
    def run_usalign(path_true: str, path_pred: str) -> float:
        try:
            result = subprocess.run(
                [
                    USALIGN_BIN,
                    path_pred,
                    path_true,
                ],  # pred first, true second is standard
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            print(f"[ERROR] USalign binary not found at {USALIGN_BIN}")
            return np.nan

        if result.returncode != 0:
            print(
                f"[WARN] USalign failed for:\n  pred={path_pred}\n  true={path_true}\n  stderr={result.stderr}"
            )
            return np.nan

        m = _tm_regex.search(result.stdout)
        if not m:
            print(
                f"[WARN] Could not parse TM-score from USalign output for:\n  pred={path_pred}\n  true={path_true}"
            )
            return np.nan

        return float(m.group(1))


    def compute_tm_for_row(row):
        true_path = row["query_true_struct_path"]
        pred_path = row["target_hit_struct_path"]

        if pd.isna(true_path) or pd.isna(pred_path):
            return np.nan

        return run_usalign(true_path, pred_path)
    return (run_usalign,)


@app.cell
def _(main_results_df, run_usalign):
    ### Run TM-align for each unique query-target pair ###
    pair_df = main_results_df[
        ["query_true_struct_path", "target_hit_struct_path"]
    ].drop_duplicates()

    pair_df["tm_score"] = pair_df.apply(
        lambda row: run_usalign(
            row["query_true_struct_path"], row["target_hit_struct_path"]
        ),
        axis=1,
    )

    # merge back
    main_results_df_tm = main_results_df.merge(
        pair_df,
        on=["query_true_struct_path", "target_hit_struct_path"],
        how="left",
    )
    return (main_results_df_tm,)


@app.cell
def _(main_results_df_tm):
    main_results_df_tm
    return


@app.cell
def _(main_results_df_tm):
    ### save the file ###
    main_results_df_tm.to_csv(
        "data/generated/gc_benchmark_with_tm.csv", index=False, header=True
    )
    return


if __name__ == "__main__":
    app.run()
