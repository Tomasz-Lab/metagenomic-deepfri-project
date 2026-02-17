import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style
    import importlib
    from matplotlib.colors import LogNorm
    import pandas as pd

    # save and display plots in whitemode
    matplotlib.style.use("default")
    return LogNorm, importlib, mo, np, pd, plt


@app.cell
def _(importlib):
    ### custom functions ###
    import helpers.visualization_helpers as vh

    importlib.reload(vh)
    return (vh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data preparation
    """)
    return


@app.cell
def _(pd):
    ### load the results prepared in gc_data_harmonization.py ###
    full_results_df = pd.read_csv("data/generated/gc_benchmark_with_tm.csv")

    # the idbin is really mmseqs2-identity based bin
    full_results_df = full_results_df.rename(columns={"idbin": "id_bin_mmseqs2"})
    return (full_results_df,)


@app.cell
def _(full_results_df):
    full_results_df
    return


@app.cell
def _(full_results_df, np, pd):
    ### summarize f1 scores and flatten ###
    metrics = [
        ("base", "f1_gc{}", False),  # naive
        ("inherited", "f1_inh_gc{}", False),
        ("synthetic", "f1_syn_gc{}", True),  # allow NaNs due to no gaps
    ]

    gc_values = [0, 1, 2, 3, 4]

    records = []
    for gc in gc_values:
        for metric_name, pattern, drop_all_nans in metrics:
            col = pattern.format(gc)
            tmp = full_results_df[
                ["pyopal_identity", "tm_score", "query", col]
            ].copy()
            tmp = tmp.rename(columns={col: "f1"})
            tmp["gc"] = gc
            tmp["metric"] = metric_name

            if drop_all_nans:
                tmp = tmp.dropna(subset=["f1"])
            records.append(tmp)

    f1_long = pd.concat(records, ignore_index=True)

    # Create dense bins for median calculation
    bins = np.arange(0.225, 1.026, 0.05)
    # bins[-1] = 1
    f1_long["id_bin_pyopal"] = pd.cut(
        f1_long["pyopal_identity"], bins=bins, include_lowest=True
    )

    f1_long["id_bin_tmscore"] = pd.cut(
        f1_long["tm_score"], bins=bins, include_lowest=True
    )
    return f1_long, gc_values


@app.cell
def _(f1_long):
    f1_long.to_csv("data/generated/gc_benchmark_f1_scores.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bootstrap helper
    """)
    return


@app.cell
def _(np):
    def bootstrap_ci(x, n_boot=500, alpha=0.05):
        x = np.asarray(x)
        if x.size == 0:
            return np.nan, np.nan, np.nan
        boots = np.median(
            np.random.choice(x, size=(n_boot, x.size), replace=True), axis=1
        )
        boots.sort()
        med = np.median(boots)
        low = np.percentile(boots, 100 * alpha / 2)
        high = np.percentile(boots, 100 * (1 - alpha / 2))
        return med, low, high
    return (bootstrap_ci,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pyopal vs MMseqs2 identity
    """)
    return


@app.cell
def _(full_results_df, np, plt):
    # --- Filter raw data for gc = 2 and base metric ---
    _subset = full_results_df[["query", "fident", "pyopal_identity"]]

    plt.figure(figsize=(11, 5))

    # --- Raw scatter ---
    plt.scatter(
        _subset["pyopal_identity"],
        _subset["fident"],
        alpha=0.1,
        s=10,
        color="#08bdba",
        edgecolors="none",
    )

    # ---- Polynomial trendline ----
    x = _subset["pyopal_identity"].values
    y = _subset["fident"].values

    degree = 3

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    # Create smooth x-grid for a clean curve
    x_sorted = np.linspace(x.min(), x.max(), 1000)

    # Plot polynomial line
    plt.plot(x_sorted, poly(x_sorted), color="black", linewidth=2)

    plt.xlabel("PyOpal identity")
    plt.xlim(0.2, 1)
    plt.ylabel("MMseqs2 identity")
    plt.title("Identity comparison")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Group size plot
    """)
    return


@app.cell
def _(full_results_df, plt):
    # count unique queries per id_bin
    _counts = (
        full_results_df.groupby("id_bin_mmseqs2")["query"].nunique().reset_index()
    )
    _counts = _counts.sort_values("id_bin_mmseqs2")

    # rename for clarity
    _counts.columns = ["id_bin_mmseqs2", "hit_count"]

    # convert to %, using 50k (all queries) as 100%
    total = 50000
    _counts["percent"] = (_counts["hit_count"] / total) * 100

    # plot
    plt.figure(figsize=(8, 4))
    plt.bar(_counts["id_bin_mmseqs2"], _counts["percent"])
    plt.xticks(rotation=45)
    plt.xlabel("Identity bin")
    plt.ylabel("% of all queries")
    plt.title("Succesful structure hits")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## F1 vs pyopal identity scatterplot for gc=2 boxplots
    """)
    return


@app.cell
def _(f1_long, pd, plt):
    # --- Filter raw data for gc = 2 and base metric ---
    _gc2_subset = f1_long[
        (f1_long["gc"] == 2) & (f1_long["metric"] == "base")
    ].copy()

    # --- Prepare 0.1-wide bins across PyOpal identity ---
    _edges = [i / 10 for i in range(0, 11)]  # 0.0, 0.1, ..., 1.0
    _bin_cats = pd.cut(
        _gc2_subset["pyopal_identity"],
        bins=_edges,
        include_lowest=True,
        right=True,
    )

    # Collect F1 values per bin and compute positions/labels
    _data_by_bin = []
    _positions = []
    _xtick_labels = []
    _n_labels = []
    for _interval in _bin_cats.cat.categories:
        _mask = _bin_cats == _interval
        _vals = _gc2_subset.loc[_mask, "f1"].values
        if len(_vals) == 0:
            continue
        _data_by_bin.append(_vals)
        _center = (_interval.left + _interval.right) / 2
        _positions.append(_center)
        # Format as ".2 - .3" (remove leading zeros, use space-dash-space)
        _left_str = f"{_interval.left:.1f}".replace("0.", ".")
        _right_str = f"{_interval.right:.1f}".replace("0.", ".")
        _xtick_labels.append(f"{_left_str}-{_right_str}")
        _n_labels.append(_vals.size)

    _fig = plt.figure(figsize=(5, 5))
    _ax = plt.gca()

    # --- Boxplot across 0.1-binned identities ---
    _bp = _ax.boxplot(
        _data_by_bin,
        positions=_positions,
        widths=0.075,
        patch_artist=True,
        boxprops=dict(facecolor="#1192e8", color="black", alpha=0.50),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(
            marker="o",
            markersize=2,
            markerfacecolor="black",
            markeredgecolor="none",
            alpha=0.3,
        ),
    )

    # --- Labels / aesthetics ---
    _ax.set_xlabel("Pyopal identity", fontsize=11, labelpad=30)  # <-- add labelpad
    _ax.set_ylabel("F1-score", fontsize=11)
    _ax.set_title("Pyopal identity vs F1-score", fontsize=12)

    _ax.set_xlim(0.2, 1.0)

    if _positions:
        _ax.set_xticks(_positions)
        _ax.set_xticklabels(_xtick_labels, rotation=0.2, fontsize=10)

        # Give tick labels a bit of space
        _ax.tick_params(axis="x", pad=6)

        # Put n-labels lower than before
        for _pos, _n in zip(_positions, _n_labels):
            _ax.text(
                _pos,
                -0.10,
                f"n={_n}",
                ha="center",
                va="top",
                fontsize=9,
                rotation=-30,
                transform=_ax.get_xaxis_transform(),
                clip_on=False,  # <-- ensures it's not clipped
            )

    _ax.grid(alpha=0.3)

    # Increase bottom margin so the external text has room
    _fig.tight_layout(rect=[0, 0.22, 1, 0.98])
    plt.savefig("plots/pyopal_vs_f1score.svg", format="svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Global F1 vs pyopal heatmap
    """)
    return


@app.cell
def _(LogNorm, f1_long, np, plt):
    # --- Filter raw data for gc = 2 and base metric ---
    _gc2_subset = f1_long[
        (f1_long["gc"] == 2) & (f1_long["metric"] == "base")
    ].copy()

    _x = _gc2_subset["pyopal_identity"].to_numpy()
    _y = _gc2_subset["f1"].to_numpy()

    plt.figure(figsize=(11, 5))

    # --- Heatmap (2D histogram) ---
    _xbins = np.linspace(0.2, 1.0, 80)
    _ybins = np.linspace(0.0, 1.0, 80)

    _h = plt.hist2d(
        _x,
        _y,
        bins=[_xbins, _ybins],
        cmap="BuPu",
        norm=LogNorm(vmin=1),
    )

    _cbar = plt.colorbar(_h[3])
    _cbar.set_label("Count (log scale)")

    # --- Median trendline (reuse existing pyopal bins) ---
    _median_by_bin = _gc2_subset.groupby("id_bin_pyopal", observed=True)[
        "f1"
    ].median()

    _centers = [interval.mid for interval in _median_by_bin.index]

    plt.plot(
        _centers,
        _median_by_bin.values,
        color="black",
        linewidth=2,
        label="binned median F1",
    )

    # --- Labels / aesthetics ---
    plt.xlabel("PyOpal identity")
    plt.xlim(0.3, 1.0)
    plt.ylabel("F1 score (global)")
    plt.ylim(0.5, 1.0)
    plt.title("Global F1 vs PyOpal identity for gc=2")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## global F1 vs TM-score
    """)
    return


@app.cell
def _(f1_long, plt):
    # --- Filter raw data for gc = 0 and base metric ---
    _gc0_subset = f1_long[
        (f1_long["gc"] == 0) & (f1_long["metric"] == "base")
    ].copy()

    plt.figure(figsize=(5, 5))

    # --- Raw scatter ---
    plt.scatter(
        _gc0_subset["tm_score"],
        _gc0_subset["f1"],
        alpha=0.1,
        s=10,
        color="#009d9a",
        edgecolors="none",
    )

    # --- Compute median trendline (based on binning) ---
    _median_by_bin = _gc0_subset.groupby("id_bin_tmscore", observed=True)[
        "f1"
    ].median()

    # bin centers for plotting
    _centers = [interval.mid for interval in _median_by_bin.index]

    plt.plot(
        _centers,
        _median_by_bin.values,
        color="#004144",
        linewidth=2,
        label="0.01-binned median F1",
    )

    # --- Labels / aesthetics ---
    plt.xlabel("TM-score")
    plt.xlim(0.0, 1)
    plt.ylabel("Whole-cmap F1 score")
    plt.title("Whole-cmap F1 score for gc=0 vs TM-score")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## global F1 vs TM-score as a heatmap
    """)
    return


@app.cell
def _(LogNorm, f1_long, np, plt):
    # --- Filter raw data for gc = 0 and base metric ---
    _gc0_subset = f1_long[
        (f1_long["gc"] == 0) & (f1_long["metric"] == "base")
    ].copy()

    _x = _gc0_subset["tm_score"].to_numpy()
    _y = _gc0_subset["f1"].to_numpy()

    plt.figure(figsize=(6, 6))

    # --- 2D histogram heatmap (counts per bin) ---
    # Tune bins to taste; these are usually a good start
    xbins = np.linspace(0.3, 1.0, 100)
    ybins = np.linspace(0.5, 1.0, 100)

    h = plt.hist2d(_x, _y, bins=[xbins, ybins], norm=LogNorm(), cmap="YlOrRd")

    cbar = plt.colorbar(h[3])
    cbar.set_label("Count (log scale)")


    # --- Median trendline (based on existing id_bin_tmscore binning) ---
    _median_by_tmscore = _gc0_subset.groupby("id_bin_tmscore", observed=True)[
        "f1"
    ].median()
    _centers_tmscore = [interval.mid for interval in _median_by_tmscore.index]

    plt.plot(
        _centers_tmscore,
        _median_by_tmscore.values,
        linewidth=2,
        color="black",
        label="Median F1 by TM-score bins",
    )

    # --- Labels / aesthetics ---
    plt.xlabel("TM-score")
    plt.xlim(0.3, 1)
    plt.ylabel("Whole-cmap F1 score")
    plt.ylim(0.6, 1)
    plt.title("Whole-cmap F1 score for gc=0 vs TM-score")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inherited F1 vs TM-score
    """)
    return


@app.cell
def _(f1_long, plt):
    # --- Filter raw data for gc = 2 and base metric ---
    gc0_subset_inh = f1_long[
        (f1_long["gc"] == 0) & (f1_long["metric"] == "inherited")
    ].copy()

    plt.figure(figsize=(5, 5))

    # --- Raw scatter ---
    plt.scatter(
        gc0_subset_inh["tm_score"],
        gc0_subset_inh["f1"],
        alpha=0.1,
        s=10,
        color="#24a148",
        edgecolors="none",
    )

    # --- Compute median trendline (based on binning) ---
    _median_by_bin = gc0_subset_inh.groupby("id_bin_tmscore", observed=True)[
        "f1"
    ].median()

    # bin centers for plotting
    _centers = [interval.mid for interval in _median_by_bin.index]

    plt.plot(
        _centers,
        _median_by_bin.values,
        color="black",
        linewidth=2,
        label="binned median F1",
    )

    # --- Labels / aesthetics ---
    plt.xlabel("TM-score")
    plt.xlim(0.3, 1)
    plt.ylabel("F1 score (inherited)")
    plt.title("Inherited F1 vs TM-score")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TM-score vs Pyopal identity
    """)
    return


@app.cell
def _(f1_long, plt):
    # --- Filter raw data for gc = 0 and base metric ---
    plt.figure(figsize=(5, 5))

    _gc0_subset = f1_long[
        (f1_long["gc"] == 0) & (f1_long["metric"] == "base")
    ].copy()

    # --- Raw scatter ---
    plt.scatter(
        _gc0_subset["pyopal_identity"],
        _gc0_subset["tm_score"],
        alpha=0.1,
        s=10,
        color="#1192e8",
        edgecolors="none",
    )

    # --- Compute median trendline (based on binning) ---
    _median_by_bin = _gc0_subset.groupby("id_bin_pyopal", observed=True)[
        "tm_score"
    ].median()
    # bin centers for plotting
    _centers = [interval.mid for interval in _median_by_bin.index]
    plt.plot(
        _centers,
        _median_by_bin.values,
        color="#003a6d",
        linewidth=2,
        label="0.01-binned median TM-score",
    )

    # --- Labels / aesthetics ---
    plt.xlabel("Pyopal identity")
    plt.xlim(0.2, 1)
    plt.ylabel("TM-score")
    plt.title("Pyopal identity vs TM-score")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## TM-score vs Pyopal identity boxplots
    """)
    return


@app.cell
def _(f1_long, pd, plt):
    # --- Filter raw data for gc = 0 and base metric ---

    _gc0_subset = f1_long[
        (f1_long["gc"] == 0) & (f1_long["metric"] == "base")
    ].copy()

    # --- Prepare 0.1 identity bins ---
    _edges = [i / 10 for i in range(0, 11)]  # 0.0, 0.1, ..., 1.0
    _bin_cats = pd.cut(
        _gc0_subset["pyopal_identity"],
        bins=_edges,
        include_lowest=True,
        right=True,
    )

    _data_by_bin = []
    _positions = []
    _xtick_labels = []
    _n_labels = []
    for _interval in _bin_cats.cat.categories:
        _mask = _bin_cats == _interval
        _vals = _gc0_subset.loc[_mask, "tm_score"].values
        if len(_vals) == 0:
            continue
        _data_by_bin.append(_vals)
        _center = (_interval.left + _interval.right) / 2
        _positions.append(_center)
        # Format as ".2-.3" (remove leading zeros, use dash)
        _left_str = f"{_interval.left:.1f}".replace("0.", ".")
        _right_str = f"{_interval.right:.1f}".replace("0.", ".")
        _xtick_labels.append(f"{_left_str}-{_right_str}")
        _n_labels.append(_vals.size)

    _fig = plt.figure(figsize=(5, 5))
    _ax = plt.gca()

    # --- Boxplot across 0.1-binned identities ---
    _bp = _ax.boxplot(
        _data_by_bin,
        positions=_positions,
        widths=0.075,
        patch_artist=True,
        boxprops=dict(facecolor="#009d9a", color="black", alpha=0.50),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(
            marker="o",
            markersize=2,
            markerfacecolor="black",
            markeredgecolor="none",
            alpha=0.3,
        ),
    )

    # --- Labels / aesthetics ---
    _ax.set_xlabel("Pyopal identity", fontsize=11, labelpad=30)
    _ax.set_ylabel("TM-score", fontsize=11)
    _ax.set_title("Pyopal identity vs TM-score", fontsize=12)

    _ax.set_xlim(0.2, 1.0)

    if _positions:
        _ax.set_xticks(_positions)
        _ax.set_xticklabels(_xtick_labels, rotation=0.2, fontsize=10)

        # Give tick labels a bit of space
        _ax.tick_params(axis="x", pad=6)

        # Put n-labels lower than before
        for _pos, _n in zip(_positions, _n_labels):
            _ax.text(
                _pos,
                -0.10,
                f"n={_n}",
                ha="center",
                va="top",
                fontsize=9,
                rotation=-30,
                transform=_ax.get_xaxis_transform(),
                clip_on=False,
            )

    _ax.grid(alpha=0.3)

    # Increase bottom margin so the external text has room
    _fig.tight_layout(rect=[0, 0.22, 1, 0.98])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mismatch-derived F1 score vs gc values
    """)
    return


@app.cell
def _(bootstrap_ci, f1_long, plt):
    # --- filter synthetic only ---
    synthetic_bootstrapped = (
        f1_long[f1_long["metric"] == "synthetic"]
        .groupby(["gc", "id_bin_pyopal"], observed=True)["f1"]
        .agg(
            median_f1=lambda x: bootstrap_ci(x)[0],
            low_ci=lambda x: bootstrap_ci(x)[1],
            high_ci=lambda x: bootstrap_ci(x)[2],
        )
        .reset_index()
    )

    synthetic_bootstrapped["id_center"] = synthetic_bootstrapped[
        "id_bin_pyopal"
    ].apply(lambda b: b.mid)

    # --- plot ---
    plt.figure(figsize=(5, 5))

    for _gc in sorted(synthetic_bootstrapped["gc"].unique()):
        sub = synthetic_bootstrapped[synthetic_bootstrapped["gc"] == _gc]

        plt.plot(sub["id_center"], sub["median_f1"], label=f"gc={_gc}", alpha=0.7)
        plt.fill_between(
            sub["id_center"], sub["low_ci"], sub["high_ci"], alpha=0.15
        )

    plt.xlabel("PyOpal identity")
    plt.xlim(0.3, 1)
    plt.ylabel("Synthetic-contacts F1")
    plt.title("Synthetic-contacts F1 vs Pyopal identity, bootstrapped")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All-identity gc summary
    """)
    return


@app.cell
def _(bootstrap_ci, f1_long, plt):
    # synthetic entries only
    all_identity_summary = (
        f1_long[f1_long["metric"] == "synthetic"]
        .groupby("gc")["f1"]
        .agg(
            median_f1=lambda x: bootstrap_ci(x)[0],
            low_ci=lambda x: bootstrap_ci(x)[1],
            high_ci=lambda x: bootstrap_ci(x)[2],
        )
        .reset_index()
    )

    # --- plot ---
    plt.figure(figsize=(11, 5))

    plt.errorbar(
        all_identity_summary["gc"],
        all_identity_summary["median_f1"],
        yerr=[
            all_identity_summary["median_f1"] - all_identity_summary["low_ci"],
            all_identity_summary["high_ci"] - all_identity_summary["median_f1"],
        ],
        fmt="o-",
        capsize=4,
    )

    plt.xlabel("gc")
    plt.xticks([0, 1, 2, 3, 4])
    plt.ylabel("Synthetic F1 (median, bootstrap CI)")
    plt.title("Global performance of gc on synthetic F1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gap size vs synthetic F1
    """)
    return


@app.cell
def _(full_results_df, vh):
    def compute_gap_runs(row):
        q = row["pyopal_query_aln"]
        t = row["pyopal_target_aln"]
        return vh.extract_synthetic_gap_runs(q, t)


    full_results_df["synthetic_gap_runs"] = full_results_df.apply(
        compute_gap_runs, axis=1
    )
    return


@app.cell
def _(bootstrap_ci, full_results_df, gc_values, np, pd):
    # 1) Build gap-events dataframe
    _records = []

    for idx, _row in full_results_df.iterrows():
        runs = _row["synthetic_gap_runs"]
        if not runs:
            continue

        py_id = _row["pyopal_identity"]

        for _gc in gc_values:
            f1_syn = _row.get(f"f1_syn_gc{_gc}", np.nan)
            if np.isnan(f1_syn):
                continue

            for g_len in runs:  # each gap event
                _records.append(
                    {
                        "query": _row["query"],
                        "target": _row["target"],
                        "gap_len": g_len,
                        "gc": _gc,
                        "f1_syn": f1_syn,
                        "pyopal_identity": py_id,
                    }
                )

    gap_events_df = pd.DataFrame(_records)

    # 2) Bin gap lengths
    gap_bins = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 200, np.inf]

    gap_labels = []
    for i in range(len(gap_bins) - 2):
        gap_labels.append(f"{gap_bins[i]}–{gap_bins[i + 1] - 1}")
    gap_labels.append(f">={gap_bins[-2]}")

    gap_events_df["gap_bin"] = pd.cut(
        gap_events_df["gap_len"],
        bins=gap_bins,
        labels=gap_labels,
        right=False,
    )

    # 3) Bootstrap summary per (gc, gap_bin)
    grouped = gap_events_df.groupby(["gc", "gap_bin"], observed=True)

    # Compute bootstrap CI for each group
    gap_ev_summary = (
        grouped["f1_syn"]
        .apply(
            lambda x: pd.Series(
                bootstrap_ci(x),
                index=["median_f1", "low_ci", "high_ci"],
            )
        )
        .unstack()
        .reset_index()
    )

    # Ensure columns are in correct order
    gap_ev_summary = gap_ev_summary[
        ["gc", "gap_bin", "median_f1", "low_ci", "high_ci"]
    ]
    return (gap_ev_summary,)


@app.cell
def _(gap_ev_summary, np, plt):
    plt.figure(figsize=(11, 5))

    for _gc in sorted(gap_ev_summary["gc"].unique()):
        _sub = gap_ev_summary[gap_ev_summary["gc"] == _gc]
        xs = np.arange(len(_sub["gap_bin"]))
        plt.plot(xs, _sub["median_f1"], marker="o", label=f"gc={_gc}")
        plt.fill_between(xs, _sub["low_ci"], _sub["high_ci"], alpha=0.15)

    plt.xticks(xs, _sub["gap_bin"], rotation=45, ha="right")
    plt.xlabel("Continuous gap length")
    plt.ylabel("Synthetic F1 (median)")
    plt.title("Synthetic F1 vs continuous gap length")
    plt.grid(alpha=0.3)
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cmap plotting and pymol visualizations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Select 10 samples per dense bins ###
    examples_df = sample_examples_by_dense_bin(
        full_results_df, n_per_bin=1, seed=42
    )
    print(len(examples_df), "examples selected")
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    for _gc in [0, 1, 2, 3, 4]:
        out_dir = f"contact_map_cmap_only_gc{_gc}"
        for _, row in examples_df.iterrows():
            plot_example_row_cmap_only(row, gc=_gc, out_dir=out_dir, show=False)
        print("Done gc", _gc)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    pymol_script_dir = "pymol_scripts_examples"

    for _, _row in examples_df.iterrows():
        write_pymol_script_for_row(
            _row,
            out_dir=pymol_script_dir,
            save_session=True,
        )
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    frames_dir = "pymol_movie_frames_examples"

    for _, _row in examples_df.iterrows():
        write_pymol_frames_script_for_row_af3(
            _row, out_dir=frames_dir, n_frames=120
        )
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    examples_df["diff"] = examples_df.f1_syn_gc3 - examples_df.f1_syn_gc2

    examples_df[["query", "diff"]]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Investigate poor F1 scores / TM-scores
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    poor_f1_high_ident = full_results_df.query(
        "f1_inh_gc2 < 0.6 and pyopal_identity > 0.7"
    )
    poor_tm_high_ident = full_results_df.query(
        "tm_score < 0.5 and pyopal_identity > 0.9"
    )
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### get contact map visualizations for poor F1 but high identity
    for _gc in [0, 1, 2, 3, 4]:
        _out_dir = f"poor_f1_high_ident_cmap_only_gc{_gc}"
        for _, _row in poor_f1_high_ident.iterrows():
            plot_example_row_cmap_only(_row, gc=_gc, out_dir=_out_dir, show=False)
        print("Done gc", _gc)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### get contact map visualization for poor TM-score but high identity
    for _gc in [2]:
        _out_dir = f"poor_tm_high_ident_cmap_only_gc{_gc}"
        for _, _row in poor_tm_high_ident.iterrows():
            plot_example_row_cmap_only(_row, gc=_gc, out_dir=_out_dir, show=False)
        print("Done gc", _gc)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### get spinning proteins for poor_f1_high_ident

    _frames_dir = "pymol_movie_poor_f1_high_ident"

    for _, _row in poor_f1_high_ident.iterrows():
        write_pymol_frames_script_for_row_af3(
            _row, out_dir=_frames_dir, n_frames=120
        )
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### get spinning proteins for poor_tm_high_ident

    _frames_dir = "pymol_movie_poor_tm_high_ident"

    for _, _row in poor_tm_high_ident.iterrows():
        write_pymol_frames_script_for_row_af3(
            _row, out_dir=_frames_dir, n_frames=120
        )
    """)
    return


if __name__ == "__main__":
    app.run()
