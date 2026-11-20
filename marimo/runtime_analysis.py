import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    return Path, mo, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # mdF vs eggNOG-mapper runtime benchmark

    Reads `results/runs.tsv` and `results/db_search.tsv` (made by
    `scripts/make_results.py`) and writes into `results/`:

    - `summary.tsv`: wall time per config, size and cache state (median, min, max over replicates);
    - `mdf_stages.tsv`: mdF stage times and GCN/CNN split, medians over replicates;
    - `db_memory.tsv`: peak memory of the MMseqs2 search of each database;
    - `plots/`: the three panels the handoff asks for, plus full AFDB vs the standard databases.
    """)
    return


@app.cell
def _(Path):
    results_dir = Path("/cephfs/team/structural_and_functional_genomics/users/filip.schymik/projects/metagenomic_deepfri/data/source/runtime_benchmark/results")
    plots_dir = Path("/cephfs/team/structural_and_functional_genomics/users/filip.schymik/projects/metagenomic_deepfri/plots")
    return plots_dir, results_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Runs
    """)
    return


@app.cell
def _(pl, results_dir):
    runs = pl.read_csv(results_dir / "runs.tsv", separator="\t", infer_schema_length=None)
    counted = runs.filter(
        (pl.col("exit_code") == 0) & (pl.col("invalid") == 0) & (pl.col("node") == "mab-n1")
    )
    grid = counted.filter(pl.col("tag") == "grid")
    afdb = counted.filter(pl.col("tag") == "afdb")
    assert grid.filter(pl.col("cache_ok") != 1).height == 0, "grid run without a clean cache state"
    assert afdb.filter(pl.col("cache_ok") != 1).height == 0, "AFDB run without a clean cache state"
    assert grid.height == 120, f"expected 120 grid run phases, got {grid.height}"
    return afdb, grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wall time per config, size and cache state
    """)
    return


@app.cell
def _(grid, pl, results_dir):
    summary = (
        grid.group_by("config", "n_proteins", "cache")
        .agg(
            replicates=pl.len(),
            median_s=pl.col("wall_s").median(),
            min_s=pl.col("wall_s").min(),
            max_s=pl.col("wall_s").max(),
        )
        .with_columns(median_h=(pl.col("median_s") / 3600).round(2))
        .sort("config", "n_proteins", "cache")
    )
    summary.write_csv(results_dir / "summary.tsv", separator="\t")
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mdF stages

    Stages from mdF's log markers (1 s resolution):

    - **search & align**: database loading plus the MMseqs2 search and alignment against every database;
    - **templates**: template structures, i.e. PDB downloads from RCSB and FoldComp reads;
    - **contact maps**;
    - **inference GCN / CNN**: the elapsed times of the GCN and CNN progress bars, summed over the four modes;
    - **other**: everything else (model loading between the bars, start-up and exit).
    """)
    return


@app.cell
def _(afdb, grid, pl):
    modes = ["bp", "cc", "ec", "mf"]
    stage_names = ["search & align", "templates", "contact maps", "inference GCN", "inference CNN", "other"]

    def _sum(names):
        return pl.sum_horizontal(pl.col(n).fill_null(0) for n in names)

    mdf_runs = pl.concat([grid, afdb], how="diagonal_relaxed").filter(
        pl.col("config").str.starts_with("mdf") & (pl.col("cache") == "cold")
    )
    search_cols = ["prepare_dbs_s"] + [c for c in mdf_runs.columns if c.startswith("search_align_")]
    mdf_stages_runs = mdf_runs.with_columns(
        _sum(search_cols).alias("search & align"),
        pl.col("templates_s").alias("templates"),
        pl.col("contact_maps_s").alias("contact maps"),
        _sum([f"infer_{m}_gcn_s" for m in modes]).alias("inference GCN"),
        _sum([f"infer_{m}_cnn_s" for m in modes]).alias("inference CNN"),
        _sum([f"infer_{m}_s" for m in modes]).alias("inference total"),
    ).with_columns(
        (
            pl.col("wall_s")
            - pl.sum_horizontal("search & align", "templates", "contact maps", "inference GCN", "inference CNN")
        ).alias("other")
    )
    return mdf_stages_runs, stage_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two usage scenarios of mdF:
    - quick, pdb100->afdb_v4_swissprot->esm_hq_clusters
    - full, pdb100_afdb_v4->esm_hq_clusters
    """)
    return


@app.cell
def _(mdf_stages_runs, pl, results_dir, stage_names):
    _keep = ["run_id", "wall_s", *stage_names, "inference total",
             "results_gcn_proteins", "results_cnn_proteins"]
    mdf_stages = (
        mdf_stages_runs.group_by("config", "dbset", "n_proteins")
        .agg(
            pl.len().alias("replicates"),
            pl.col(_keep).sort_by("wall_s").get(pl.len() // 2),
        )
        .rename({"results_gcn_proteins": "gcn_proteins", "results_cnn_proteins": "cnn_proteins"})
        .with_columns(
            pl.col("other").round(1),
            (pl.col("inference total") / pl.col("wall_s")).round(3).alias("inference_fraction"),
        )
        .sort("dbset", "config", "n_proteins", descending=[True, False, False])
    )
    mdf_stages.write_csv(results_dir / "mdf_stages.tsv", separator="\t")
    mdf_stages
    return (mdf_stages,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Peak memory per database

    From the MMseqs2 processes that searched each database (`results/db_search.tsv`).
    RSS includes the memory-mapped index, which is page cache the kernel can reclaim;
    anon is the memory MMseqs2 itself allocated.
    """)
    return


@app.cell
def _(pl, results_dir):
    db_search = pl.read_csv(results_dir / "db_search.tsv", separator="\t", infer_schema_length=None)
    db_memory = (
        db_search.filter(
            (pl.col("invalid") == 0)
            & pl.col("tag").is_in(["grid", "afdb"])
            & (pl.col("cache") == "cold")
        )
        .group_by("dbset", "database", "n_proteins")
        .agg(
            pl.len().alias("runs"),
            pl.col("search_align_s").median().alias("median_search_align_s"),
            pl.col("mmseqs_peak_rss_gib").max().alias("max_mmseqs_rss_gib"),
            pl.col("mmseqs_max_anon_gib").max().alias("max_mmseqs_anon_gib"),
        )
        .sort("dbset", "database", "n_proteins", descending=[True, False, False])
    )
    db_memory.write_csv(results_dir / "db_memory.tsv", separator="\t")
    db_memory
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(plt):
    config_style = {
        "mdf-cpu": ("#eb6834", "o", "mdF CPU"),
        "mdf-gpu": ("#1baf7a", "s", "mdF GPU"),
        "eggnog-fast": ("#86b6ef", "^", "eggNOG fast"),
        "eggnog-sensitive": ("#2a78d6", "D", "eggNOG sensitive"),
        "eggnog-ultra": ("#104281", "v", "eggNOG ultra-sensitive"),
    }
    stage_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    ink, ink_muted, grid_color, surface = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": ink_muted, "axes.labelcolor": ink,
        "xtick.color": ink_muted, "ytick.color": ink_muted, "axes.grid": True,
        "grid.color": grid_color, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.dpi": 200,
    })
    return config_style, ink, ink_muted, stage_colors, surface


@app.cell
def _(config_style, grid, pl, plots_dir, plt):
    from matplotlib.ticker import StrMethodFormatter

    _cold = grid.filter(pl.col("cache") == "cold")
    fig_total, _ax = plt.subplots(figsize=(5.6, 4.2), layout="constrained")
    for _config, (_color, _, _label) in config_style.items():
        _s = (
            _cold.filter(pl.col("config") == _config)
            .group_by("n_proteins")
            .agg(
                (pl.col("wall_s").median() / 3600).alias("med"),
                (pl.col("wall_s").min() / 3600).alias("lo"),
                (pl.col("wall_s").max() / 3600).alias("hi"),
            )
            .sort("n_proteins")
        )
        _med = _s["med"].to_numpy()
        _ax.errorbar(
            _s["n_proteins"].to_numpy(), _med,
            yerr=[_med - _s["lo"].to_numpy(), _s["hi"].to_numpy() - _med],
            color=_color, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6,
            elinewidth=1, capsize=2.5, capthick=1, zorder=2, label=_label,
        )
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel("proteins")
    _ax.set_ylabel("wall time (h)")
    _ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    _ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig_total.savefig(plots_dir / "total_time.png")
    fig_total.savefig(plots_dir / "total_time.pdf")
    fig_total
    return


@app.cell
def _(config_style, grid, ink_muted, pl, plots_dir, plt):
    sizes = [10, 100, 1000]
    caches = {"cold": "#2a78d6", "warm": "#eb6834"}
    fig_cw, _axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout="constrained")
    for _ax, _n in zip(_axes, sizes):
        for _i, _config in enumerate(config_style):
            for _j, (_cache, _color) in enumerate(caches.items()):
                _d = grid.filter((pl.col("config") == _config) & (pl.col("n_proteins") == _n)
                                 & (pl.col("cache") == _cache))
                _x = _i + (_j - 0.5) * 0.38
                _ax.bar(_x, _d["wall_s"].median(), width=0.36, color=_color,
                        edgecolor="white", linewidth=1,
                        label=_cache if (_i == 0 and _n == sizes[0]) else None)
                _ax.scatter([_x] * _d.height, _d["wall_s"], s=9, color=ink_muted, zorder=3)
        _ax.set_xticks(range(len(config_style)),
                       [v[2].replace(" ", "\n", 1) for v in config_style.values()], fontsize=7.5)
        _ax.set_title(f"{_n:,} proteins", loc="left", fontsize=9)
        _ax.grid(axis="x", visible=False)
    _axes[0].set_ylabel("wall time (s)")
    _axes[0].legend(frameon=False, fontsize=8,
                    title_fontsize=8, loc="upper left")
    fig_cw.savefig(plots_dir / "cold_vs_warm.png")
    fig_cw.savefig(plots_dir / "cold_vs_warm.pdf")
    fig_cw
    return


@app.cell
def _(ink, mdf_stages, pl, plots_dir, plt, stage_colors, stage_names, surface):
    panels = [("mdf-cpu", "swissprot_hq30", "mdF CPU"),
              ("mdf-gpu", "swissprot_hq30", "mdF GPU"),
              ("mdf-cpu", "afdb_hq30", "mdF CPU, full AFDB (1 replicate)")]
    fig_st, _axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, layout="constrained")
    for _ax, (_config, _dbset, _title) in zip(_axes, panels):
        _d = mdf_stages.filter((pl.col("config") == _config) & (pl.col("dbset") == _dbset)).sort("n_proteins")
        _x = list(range(_d.height))
        _bottom = [0.0] * _d.height
        for _stage, _color in zip(stage_names, stage_colors):
            _frac = (_d[_stage] / _d["wall_s"] * 100).to_list()
            _ax.bar(_x, _frac, bottom=_bottom, width=0.7, color=_color, edgecolor=surface,
                    linewidth=1, label=_stage)
            _bottom = [b + f for b, f in zip(_bottom, _frac)]
        for _xi, _w in zip(_x, _d["wall_s"]):
            _txt = f"{_w / 3600:.1f} h" if _w >= 3600 else f"{_w / 60:.0f} min"
            _ax.text(_xi, 101.5, _txt, ha="center", va="bottom", fontsize=7.5, color=ink)
        _ax.set_xticks(_x, [f"{n:,}" for n in _d["n_proteins"]], fontsize=8)
        _ax.set_xlabel("proteins")
        _ax.set_title(_title, loc="left", fontsize=9, pad=14)
        _ax.set_ylim(0, 108)
        _ax.grid(axis="x", visible=False)
    _axes[0].set_ylabel("share of wall time (%)")
    _handles, _labels = _axes[0].get_legend_handles_labels()
    fig_st.legend(_handles[::-1], _labels[::-1], loc="outside right center", frameon=False, fontsize=8)
    fig_st.savefig(plots_dir / "mdf_stages.png")
    fig_st.savefig(plots_dir / "mdf_stages.pdf")
    fig_st
    return


@app.cell
def _(mdf_stages, pl, plots_dir, plt, stage_colors, stage_names, surface):
    _sizes = [100, 1000, 10_000, 100_000]
    _sets = [("swissprot_hq30", "standard"), ("afdb_hq30", "full AFDB")]
    fig_afdb, _axes = plt.subplots(1, 4, figsize=(11, 3.6), layout="constrained")
    for _ax, _n in zip(_axes, _sizes):
        _scale, _unit = (3600, "h") if _n >= 10_000 else (60, "min")
        for _i, (_dbset, _name) in enumerate(_sets):
            _row = mdf_stages.filter((pl.col("config") == "mdf-cpu") & (pl.col("dbset") == _dbset)
                                     & (pl.col("n_proteins") == _n))
            _bottom = 0.0
            for _stage, _color in zip(stage_names, stage_colors):
                _v = _row[_stage][0] / _scale
                _ax.bar(_i, _v, bottom=_bottom, width=0.6, color=_color, edgecolor=surface,
                        linewidth=1, label=_stage if _n == _sizes[0] and _i == 0 else None)
                _bottom += _v
        _ax.set_xticks([0, 1], [s[1] for s in _sets], fontsize=8)
        _ax.set_ylabel(f"wall time ({_unit})")
        _ax.set_title(f"{_n:,} proteins", loc="left", fontsize=9)
        _ax.grid(axis="x", visible=False)
    _handles, _labels = _axes[0].get_legend_handles_labels()
    fig_afdb.legend(_handles[::-1], _labels[::-1], loc="outside right center", frameon=False, fontsize=8)
    fig_afdb.savefig(plots_dir / "afdb_vs_standard.png")
    fig_afdb.savefig(plots_dir / "afdb_vs_standard.pdf")
    fig_afdb
    return


if __name__ == "__main__":
    app.run()
