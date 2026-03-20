import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style
    from upsetplot import from_contents, plot
    import importlib
    import os
    import pandas as pd
    from goatools.obo_parser import GODag
    from pathlib import Path
    from scipy.stats import mannwhitneyu

    PLOT_DIR = "plots"
    RAW_DIR = os.path.join(PLOT_DIR, "raw_data")
    os.makedirs(RAW_DIR, exist_ok=True)

    # save and display plots in whitemode
    matplotlib.style.use("default")
    return (
        GODag,
        Path,
        PLOT_DIR,
        RAW_DIR,
        from_contents,
        importlib,
        mannwhitneyu,
        mo,
        np,
        os,
        pd,
        plot,
        plt,
    )


@app.cell
def _(importlib):
    import helpers.wang_similarity_helpers as wsh

    importlib.reload(wsh)
    return (wsh,)


@app.cell
def _(importlib):
    import helpers.goterm_propagator as prop

    importlib.reload(prop)
    return (prop,)


@app.cell
def _(importlib):
    import helpers.goterm_plots as gplot

    importlib.reload(gplot)
    return (gplot,)


@app.cell
def _():
    # ignore upset plot warnings
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
    return


@app.cell
def _(GODag, pd):
    # information content from CAFA 5
    # (https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data) IA.txt
    # ic_df = pd.read_csv("data/external/IA.txt", sep="\t", names=["go_term", "IC"])

    ic_df = pd.read_csv("data/external/IC_swissprot.csv")

    # GO DAF
    go_dag = GODag("data/external/go-basic-latest.obo")
    return go_dag, ic_df


@app.cell
def _(ic_df):
    print(ic_df)
    return


@app.cell
def _(go_dag, wsh):
    wang = wsh.WangSemantic(go_dag)
    return (wang,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Identity go_terms benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading
    """)
    return


@app.cell
def _(Path):
    BASE = Path(
        "/home/FilipS/2025/metagenomic_deepfri/data/source/reference_proteomes/mdf_results"
    )

    identity_bins = {
        "50": "identity_bin_0.00-0.50",
        "90": "identity_bin_0.00-0.90",
        "100": "identity_bin_0.00-1.01",
    }

    db_files = {
        "pdb": "pdb100_230517_alignment_scores.tsv",
        "uniprot": "afdb_uniprot_v4_alignment_scores.tsv",
        "esm": "highquality_clust30_alignment_scores.tsv",
    }

    results_file = "generate_contacts_2/results.tsv"
    return BASE, db_files, identity_bins, results_file


@app.cell
def _(BASE, db_files, ic_df, identity_bins, pd, results_file):
    def load_alignment_sets(bin_key):
        bin_dir = BASE / "pyopal_alignments" / identity_bins[bin_key]
        res_dir = BASE / "results" / identity_bins[bin_key]

        # Load PyOpal alignments
        alignments = {
            name: pd.read_csv(bin_dir / fname, sep="\t")
            for name, fname in db_files.items()
        }

        # Load GO results + merge IC table
        go = pd.read_csv(res_dir / results_file, sep="\t")
        go = (
            go.merge(
                ic_df, left_on="GO_term/EC_number", right_on="go_term", how="left"
            )
            .drop(columns=["go_term"])
            .rename(
                columns={"DeepFRI_mode": "aspect", "GO_term/EC_number": "go_term"}
            )
        )

        return alignments, go


    # Load everything
    pyopal_50, go_terms_50 = load_alignment_sets("50")
    pyopal_90, go_terms_90 = load_alignment_sets("90")
    pyopal_100, go_terms_100 = load_alignment_sets("100")
    return (
        go_terms_100,
        go_terms_50,
        go_terms_90,
        pyopal_100,
        pyopal_50,
        pyopal_90,
    )


@app.cell
def _(pyopal_100, pyopal_50, pyopal_90):
    def compute_coverage(align_dict, total_queries):
        """align_dict = {'pdb':df, 'uniprot':df, 'esm':df}"""

        # unique query hits for each DB
        hits = {db: set(df["query_name"]) for db, df in align_dict.items()}

        cov_pdb = len(hits["pdb"])
        cov_pdb_uniprot = len(hits["pdb"] | hits["uniprot"])
        cov_all = len(hits["pdb"] | hits["uniprot"] | hits["esm"])

        any_hit = cov_all

        return {
            "pdb_only": cov_pdb,
            "pdb_plus_uniprot": cov_pdb_uniprot,
            "pdb_plus_uniprot_plus_esm": cov_all,
            "any_hit": any_hit,
            "total_queries": total_queries,
        }


    cov_50 = compute_coverage(pyopal_50, total_queries=14895)
    cov_90 = compute_coverage(pyopal_90, total_queries=14895)
    cov_100 = compute_coverage(pyopal_100, total_queries=14895)
    cov_50, cov_90, cov_100
    return cov_100, cov_50, cov_90


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Composition plot
    """)
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, cov_100, cov_50, cov_90, pd, plt):
    def decompose_cov_percent(cov):
        total = cov["total_queries"]

        pdb = cov["pdb_only"]
        uni = cov["pdb_plus_uniprot"] - pdb
        esm = cov["pdb_plus_uniprot_plus_esm"] - (pdb + uni)
        no_hit = total - (pdb + uni + esm)

        # convert to percentages
        return (
            100 * pdb / total,
            100 * uni / total,
            100 * esm / total,
            100 * no_hit / total,
        )


    covs = [cov_50, cov_90, cov_100]
    _titles = ["Identity ≤ 0.50", "Identity ≤ 0.90", "Identity ≤ 1.00"]

    _fig, _axes = plt.subplots(1, 3, figsize=(5, 5), sharey=True)

    for _ax, cov, title in zip(_axes, covs, _titles):
        pdb, uni, esm, no_hit = decompose_cov_percent(cov)

        labels = [""]

        _ax.bar(labels, [pdb], label="PDB hits")
        _ax.bar(labels, [uni], bottom=[pdb], label="+UniProt hits")
        _ax.bar(labels, [esm], bottom=[pdb + uni], label="+ESM hits")
        _ax.bar(labels, [no_hit], bottom=[pdb + uni + esm], label="No hit")

        _ax.set_title(title)
        _ax.set_ylim(0, 100)

    # set y-axis label only once
    _axes[0].set_ylabel("Percent of queries (%)")


    # Figure-level title
    _fig.suptitle("Database hits distribution", fontsize=14, y=1.08)

    # Legend outside
    handles, labels = _axes[0].get_legend_handles_labels()
    _fig.legend(
        handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/composition_plot.svg", bbox_inches="tight")
    pd.DataFrame([cov_50, cov_90, cov_100], index=["id50", "id90", "id100"]).to_csv(
        f"{RAW_DIR}/composition_plot.csv"
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upset-diagram 50 vs 90 vs 100
    """)
    return


@app.cell
def _(from_contents, go_terms_100, go_terms_50, go_terms_90, plot, plt):
    def upset_by_aspect(aspect_label, **dfs):
        contents = {}

        for name, df in dfs.items():
            # filter to aspect
            if "aspect" in df.columns:
                subset = df[df["aspect"].str.lower() == aspect_label]
            elif "Aspect" in df.columns:
                subset = df[df["Aspect"].str.lower() == aspect_label]

            if "GO_term/EC" in df.columns:
                subset = subset.rename(columns={"GO_term/EC": "go_term"})

            # build set of (Protein, go_term) pairs
            pairs = set(
                subset[["Protein", "go_term"]].itertuples(index=False, name=None)
            )
            contents[name] = pairs

        series = from_contents(contents)

        fig = plt.figure(figsize=(7, 5))
        plot(
            series,
            subset_size="count",
            show_percentages=True,
            fig=fig,
            element_size=None,
            sort_categories_by="input",
            sort_by="cardinality",
        )
        # plt.suptitle(f"Annotation overlap for {aspect_label}")
        plt.show()


    for _aspect in ["bp", "mf", "cc"]:
        upset_by_aspect(
            aspect_label=_aspect,
            id50=go_terms_50,
            id90=go_terms_90,
            id100=go_terms_100,
        )
    return (upset_by_aspect,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refinement or noise
    """)
    return


@app.cell
def _(PLOT_DIR, go_terms_100, go_terms_50, go_terms_90, plt, wang, wsh):
    # define sources once: (key used in uniques/core, label on plot, dataframe)
    _sources = [
        ("go_terms50", "go_terms50", go_terms_50),
        ("go_terms90", "go_terms90", go_terms_90),
        ("go_terms100", "go_terms100", go_terms_100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        # 1) Per-protein GO sets for this aspect
        _G = {}
        for _key, _label, _df in _sources:
            _G[_key] = wsh.go_terms_per_protein(
                _df,
                aspect=_asp_code,
                go_col="go_term",
            )

        # 2) Compute core + unique sets (arbitrary number of sources)
        _core, _uniques = wsh.core_and_unique_per_protein(**_G)

        # 3) Similarity of unique terms to the core, per source
        _data = []
        _labels = []

        for _key, _label, _df in _sources:
            _stc = wsh.unique_to_core_similarity(_core, _uniques[_key], wang)
            _data.append(_stc.dropna())
            _labels.append(_label)

        # 4) Boxplot on this axis
        _bp = _ax.boxplot(_data, tick_labels=_labels, sym=".")

        # annotate medians
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
            _median_val = _y.mean()
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _ax.set_ylim(0.0, 1.0)
        _ax.set_title(f"{_asp_title}")

        if _asp_code == "bp":
            _ax.set_ylabel("Mean similarity of unique GO terms to core")

    _fig.suptitle(
        "Are method-specific GO terms refinements or novelty? (per GO aspect)",
        y=1.03,
    )
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/refinement_or_noise_identity.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median number of predictions per protein 50 vs 90 vs 100
    """)
    return


@app.cell
def _(PLOT_DIR, go_terms_100, go_terms_50, go_terms_90, plt):
    def counts_per_protein_aspect(df):
        """
        Returns a Series indexed by (Protein, aspect)
        with the number of unique GO terms for that (Protein, aspect).
        """
        return df.groupby(["Protein", "aspect"])["go_term"].nunique()


    counts_50 = counts_per_protein_aspect(go_terms_50)
    counts_90 = counts_per_protein_aspect(go_terms_90)
    counts_100 = counts_per_protein_aspect(go_terms_100)


    # helper to get a 1D Series: index = Protein, value = #GO for that aspect
    def get_aspect_counts(counts_series, aspect):
        return counts_series.xs(aspect, level="aspect")


    bp_50 = get_aspect_counts(counts_50, "bp")
    bp_90 = get_aspect_counts(counts_90, "bp")
    bp_100 = get_aspect_counts(counts_100, "bp")

    mf_50 = get_aspect_counts(counts_50, "mf")
    mf_90 = get_aspect_counts(counts_90, "mf")
    mf_100 = get_aspect_counts(counts_100, "mf")

    cc_50 = get_aspect_counts(counts_50, "cc")
    cc_90 = get_aspect_counts(counts_90, "cc")
    cc_100 = get_aspect_counts(counts_100, "cc")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    # Biological process
    axes[0].boxplot(
        [bp_50, bp_90, bp_100], tick_labels=["50", "90", "100"], sym="."
    )
    axes[0].set_title("BP")
    axes[0].set_ylabel("Number of GO annotations per protein")
    axes[0].set_ylim(0, 100)

    # Molecular function
    axes[1].boxplot(
        [mf_50, mf_90, mf_100], tick_labels=["50", "90", "100"], sym="."
    )
    axes[1].set_title("MF")
    axes[1].set_ylim(0, 30)

    # Cellular component
    axes[2].boxplot(
        [cc_50, cc_90, cc_100], tick_labels=["50", "90", "100"], sym="."
    )
    axes[2].set_title("CC")
    axes[2].set_ylim(0, 20)

    fig.suptitle(
        "Annotation number per protein by identity cutoff and GO aspect", y=1.03
    )
    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/annotations_per_protein_identity.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Median number of predictions vs max IC for > 0.2 scores
    """)
    return


@app.cell
def _(PLOT_DIR, go_terms_100, mannwhitneyu, np, plt):
    _score_cutoff = 0.2
    _bin_width = 3
    _min_n_for_test = 20
    _eps = 0.5


    def _p_to_stars(_p: float) -> str:
        if np.isnan(_p):
            return "n/a"
        if _p < 1e-4:
            return "****"
        if _p < 1e-3:
            return "***"
        if _p < 1e-2:
            return "**"
        if _p < 5e-2:
            return "*"
        return "ns"


    def _add_sig_bracket(_ax, _i, _j, _y, _h, _text):
        _ax.plot([_i, _i, _j, _j], [_y, _y + _h, _y + _h, _y], linewidth=1)
        _ax.text(
            (_i + _j) / 2, _y + _h, _text, ha="center", va="bottom", fontsize=9
        )


    def _bin_label(_b: int) -> str:
        return f"{int(_b)}–{int(_b + _bin_width)}"


    # filter terms, summarize per protein
    _df = go_terms_100.loc[go_terms_100["Score"] >= _score_cutoff].copy()
    _summ = (
        _df.groupby(["Protein", "aspect"])
        .agg(n_pred=("go_term", "nunique"), max_ic=("IC", "max"))
        .reset_index()
    )
    _summ["ic_bin"] = (np.floor(_summ["max_ic"] / _bin_width) * _bin_width).astype(
        "Int64"
    )

    _aspects = [("bp", "BP"), ("mf", "MF"), ("cc", "CC")]
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    _global_y_tops = []

    for _idx, (_ax, (_asp_code, _asp_title)) in enumerate(zip(_axes, _aspects)):
        _d = _summ.loc[
            _summ["aspect"].astype(str).str.lower() == _asp_code,
            ["n_pred", "ic_bin"],
        ].dropna()

        _bins = np.sort(_d["ic_bin"].unique())
        _data = [
            _d.loc[_d["ic_bin"] == _b, "n_pred"].astype(float).to_numpy()
            for _b in _bins
        ]
        _ns = [len(_arr) for _arr in _data]
        _tick_labels = [
            f"{_bin_label(int(_b))}\n" + f"n={_n}" for _b, _n in zip(_bins, _ns)
        ]

        _data_plot = [np.where(_arr <= 0, _eps, _arr) for _arr in _data]
        _bp = _ax.boxplot(
            _data_plot, tick_labels=_tick_labels, sym=".", widths=0.5
        )

        _ax.set_title(_asp_title)
        _ax.set_yscale("log")
        if _asp_code == "bp":
            _ax.set_ylabel("n predictions per protein")
        if _idx == 1:  # middle panel
            _ax.set_xlabel(f"Max IC bin")
        else:
            _ax.set_xlabel("")

        _all_y = np.concatenate(_data_plot)
        _ymin = max(_eps * 0.8, float(np.nanmin(_all_y)))
        _ymax = float(np.nanmax(_all_y))
        _ax.set_ylim(_ymin, _ymax * 2.5)

        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _median_val = float(np.exp(np.mean(np.log(_median_line.get_ydata()))))
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _pairs = [(i, i + 1) for i in range(1, len(_data_plot))]
        _panel_max = float(np.nanmax(_all_y))
        _y0 = _panel_max * 1.05
        _step = 1.25
        _h_factor = 1.10

        for _k, (_i, _j) in enumerate(_pairs):
            _a, _b = _data[_i - 1], _data[_j - 1]
            if (len(_a) < _min_n_for_test) or (len(_b) < _min_n_for_test):
                _stars = "n/a"
            else:
                _stars = _p_to_stars(
                    mannwhitneyu(_a, _b, alternative="two-sided").pvalue
                )

            _y = _y0 * (_step**_k)
            _add_sig_bracket(_ax, _i, _j, _y, _y * (_h_factor - 1), _stars)

        _required_top = _y0 * (_step ** max(1, len(_pairs))) * 1.3
        _global_y_tops.append(_required_top)
        _ax.tick_params(axis="x", labelsize=9)

    _global_top = max(_global_y_tops)

    for _ax in _axes:
        _ymin, _ = _ax.get_ylim()
        _ax.set_ylim(_ymin, _global_top)

    _fig.suptitle(
        f"Predictions per protein vs max IC | score ≥ {_score_cutoff}",
        y=1.01,
    )
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/predictions_vs_max_ic.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median IC per protein 50 vs 90 vs 100
    """)
    return


@app.cell
def _(PLOT_DIR, go_terms_100, go_terms_50, go_terms_90, plt):
    def ic_by_aspect(df, aspect):
        """
        Return a 1D Series of IC values for a given aspect
        (dropping NaN medians).
        """
        return (
            df.groupby(["Protein", "aspect"])["IC"]
            .median()
            .xs(aspect, level="aspect")
            .dropna()
        )


    # BP
    bp_50_ic = ic_by_aspect(go_terms_50, "bp")
    bp_90_ic = ic_by_aspect(go_terms_90, "bp")
    bp_100_ic = ic_by_aspect(go_terms_100, "bp")

    # MF
    mf_50_ic = ic_by_aspect(go_terms_50, "mf")
    mf_90_ic = ic_by_aspect(go_terms_90, "mf")
    mf_100_ic = ic_by_aspect(go_terms_100, "mf")

    # CC
    cc_50_ic = ic_by_aspect(go_terms_50, "cc")
    cc_90_ic = ic_by_aspect(go_terms_90, "cc")
    cc_100_ic = ic_by_aspect(go_terms_100, "cc")

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    # Biological process
    _axes[0].boxplot(
        [bp_50_ic, bp_90_ic, bp_100_ic], tick_labels=["50", "90", "100"], sym="."
    )
    _axes[0].set_title("BP")
    _axes[0].set_ylabel("IC")

    # Molecular function
    _axes[1].boxplot(
        [mf_50_ic, mf_90_ic, mf_100_ic], tick_labels=["50", "90", "100"], sym="."
    )
    _axes[1].set_title("MF")

    # Cellular component
    _axes[2].boxplot(
        [cc_50_ic, cc_90_ic, cc_100_ic], tick_labels=["50", "90", "100"], sym="."
    )
    _axes[2].set_title("CC")

    _fig.suptitle(
        "Per protein Information content",
        y=1.03,
    )
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/median_ic_per_protein_identity.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median score per protein 50 vs 90 vs 100
    """)
    return


@app.cell
def _(PLOT_DIR, go_terms_100, go_terms_50, go_terms_90, plt):
    def score_by_aspect(df, aspect):
        return (
            df.groupby(["Protein", "aspect"])["Score"]
            .median()
            .xs(aspect, level="aspect")
            .dropna()
        )


    # BP
    bp_50_score = score_by_aspect(go_terms_50, "bp")
    bp_90_score = score_by_aspect(go_terms_90, "bp")
    bp_100_score = score_by_aspect(go_terms_100, "bp")

    # MF
    mf_50_score = score_by_aspect(go_terms_50, "mf")
    mf_90_score = score_by_aspect(go_terms_90, "mf")
    mf_100_score = score_by_aspect(go_terms_100, "mf")

    # CC
    cc_50_score = score_by_aspect(go_terms_50, "cc")
    cc_90_score = score_by_aspect(go_terms_90, "cc")
    cc_100_score = score_by_aspect(go_terms_100, "cc")

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    # Biological Process
    _axes[0].boxplot(
        [bp_50_score, bp_90_score, bp_100_score],
        tick_labels=["50", "90", "100"],
        sym=".",
    )
    _axes[0].set_title("BP")
    _axes[0].set_ylabel("Score")

    # Molecular Function
    _axes[1].boxplot(
        [mf_50_score, mf_90_score, mf_100_score],
        tick_labels=["50", "90", "100"],
        sym=".",
    )
    _axes[1].set_title("MF")

    # Cellular Component
    _axes[2].boxplot(
        [cc_50_score, cc_90_score, cc_100_score],
        tick_labels=["50", "90", "100"],
        sym=".",
    )
    _axes[2].set_title("CC")

    _fig.suptitle("Per protein score", y=1.03)
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/median_score_per_protein_identity.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cohesion analysis
    """)
    return


@app.cell
def _(PLOT_DIR, go_terms_100, go_terms_50, go_terms_90, plt, wang, wsh):
    _sources = [
        ("go_terms_50", go_terms_50),
        ("go_terms_90", go_terms_90),
        ("go_terms_100", go_terms_100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    # compute cohesion per aspect and source
    _coh = {}  # _coh[asp_code][label] = Series

    for _asp_code, _asp_title in _aspects:
        _coh[_asp_code] = {}
        for _label, _df in _sources:
            _coh[_asp_code][_label] = wsh.per_protein_cohesion(
                _df, _asp_code, wang, go_col="go_term"
            )

    # shared figure with 3 subplots
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        _ylabel = (
            "Semantic cohesion (mean Wang similarity)"
            if _asp_code == "bp"
            else None
        )
        wsh.boxplot_cohesion(
            _ax,
            _coh[_asp_code],
            _title=_asp_title,
            _ylabel=_ylabel,
        )

    _fig.suptitle(
        "Per-protein GO-term cohesion across sources and GO aspects",
        y=1.03,
    )
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/cohesion_analysis.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # seq meta struct benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## data loading
    """)
    return


@app.cell
def _(go_terms_100, go_terms_50, go_terms_90, ic_df, pd):
    # sequence only deepFRI
    dFseq = pd.read_csv(
        "data/source/reference_proteomes/sequence_only_deepfri/merged_predictions.csv"
    )
    dFseq["Aspect"] = dFseq["Aspect"].str.lower()
    dFseq = pd.merge(
        dFseq,
        ic_df,
        left_on="GO_term/EC",
        right_on="go_term",
        how="left",
    ).drop(columns=["go_term"])


    # true positive structure deepFRI
    dFstruct = pd.read_csv(
        "data/source/reference_proteomes/true_structure_deepfri/merged_predictions.csv"
    )
    dFstruct["Aspect"] = dFstruct["Aspect"].str.lower()
    dFstruct = pd.merge(
        dFstruct,
        ic_df,
        left_on="GO_term/EC",
        right_on="go_term",
        how="left",
    ).drop(columns=["go_term"])

    # mdeepfri with forced identity below 50
    mdF50 = go_terms_50.rename(
        columns={"Annotation": "Name", "go_term": "GO_term/EC", "aspect": "Aspect"}
    )[["Protein", "GO_term/EC", "Score", "Name", "Aspect", "IC"]]

    # mdeepfri with forced identity below 90
    mdF90 = go_terms_90.rename(
        columns={"Annotation": "Name", "go_term": "GO_term/EC", "aspect": "Aspect"}
    )[["Protein", "GO_term/EC", "Score", "Name", "Aspect", "IC"]]

    # mdeepfri only with filtered out self-hits
    mdF100 = go_terms_100.rename(
        columns={"Annotation": "Name", "go_term": "GO_term/EC", "aspect": "Aspect"}
    )[["Protein", "GO_term/EC", "Score", "Name", "Aspect", "IC"]]
    return dFseq, dFstruct, mdF100, mdF50, mdF90


@app.cell
def _(mo):
    mo.md(r"""
    ## go-term propagation
    """)
    return


@app.cell
def _(mdF100):
    mdF100
    return


@app.cell
def _(dFstruct, ic_df, mdF100, prop):
    mdF100_propped = prop.propagate_go_annotations(
        mdF100,
        protein_col="Protein",
        term_col="GO_term/EC",
        score_col="Score",
        obo_path="data/external/go-basic-latest.obo",
        ic_df=ic_df,
        relations=("is_a"),
        exclude_roots=True,
        obsolete_mode="drop",  # or "map"
        obsolete_prefer="replaced_by",
        score_min=0.3,  # drop low-score annotations first
        round_decimals=3,  # round Score/IC/percent
        filter_predictable_terms=True,  # Enable filtering to only include dF predictable terms
        predictable_terms_path="data/external/deepfri_predictable_terms.csv",
    )

    dFstruct_propped = prop.propagate_go_annotations(
        dFstruct,
        protein_col="Protein",
        term_col="GO_term/EC",
        score_col="Score",
        obo_path="data/external/go-basic-latest.obo",
        ic_df=ic_df,
        relations=("is_a"),
        exclude_roots=True,
        obsolete_mode="drop",  # or "map"
        obsolete_prefer="replaced_by",
        score_min=0.3,  # drop low-score annotations first
        round_decimals=3,  # round Score/IC/percent
        filter_predictable_terms=True,  # Enable filtering to only include dF predictable terms
        predictable_terms_path="data/external/deepfri_predictable_terms.csv",
    )
    return dFstruct_propped, mdF100_propped


@app.cell
def _(mo):
    mo.md(r"""
    ## Concordance analysis
    """)
    return


@app.cell
def _(mdF100_propped):
    mdF100_propped
    return


@app.cell
def _(PLOT_DIR, dFstruct_propped, gplot, mdF100_propped, plt):
    per_prot, per_bin, stats = gplot.concordance_by_ic(
        mdF100_propped[0],
        dFstruct_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.3,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=9,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
        by_aspect=True,
        filter_predictable_terms=False,  # Filtering of predictable terms (REMOVES ALSO PROPAGATED TERMS)
        predictable_terms_path="data/external/deepfri_predictable_terms.csv",
    )

    _fig = gplot.plot_concordance_boxplot(
        per_prot,
        per_bin,
        # title="mdF concordance with EggNOG",
    )

    gplot.save_concordance_artifacts(
        _fig,
        per_prot,
        out_prefix="refprot_mdf_dfstr_concordance_by_ic",
        out_dir=PLOT_DIR,
    )

    plt.show()
    return per_bin, per_prot


@app.cell
def _(mo):
    mo.md(r"""
    ## Concordance violin plot
    """)
    return


@app.cell
def _(PLOT_DIR, gplot, per_bin, per_prot, plt):
    _fig = gplot.plot_concordance_violin(
        per_prot,
        per_bin,
        title="Method concordance by IC bin",
    )
    _fig.savefig(f"{PLOT_DIR}/refprot_concordance_violin.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(per_prot):
    per_prot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## coverage histogram
    """)
    return


@app.cell
def _(dFseq):
    dFseq
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, dFseq, dFstruct, mdF100, mdF90, pd, plt):
    _n_queries = 14895


    # filter on quality
    _dFseq_hq = dFseq[dFseq["Score"] >= 0.2]
    _dFstruct_hq = dFstruct[dFstruct["Score"] >= 0.3]
    _mdF90_hq = mdF90[mdF90["Score"] >= 0.3]
    _mdF100_hq = mdF100[mdF100["Score"] >= 0.3]

    # calculate prediction coverage
    _groups = {
        "dF_seq": len(_dFseq_hq["Protein"].unique()) / _n_queries * 100,
        "dF_struct": len(_dFstruct_hq["Protein"].unique()) / _n_queries * 100,
        "mdF_90": len(_mdF90_hq["Protein"].unique()) / _n_queries * 100,
        "mdF_100": len(_mdF100_hq["Protein"].unique()) / _n_queries * 100,
    }

    _groups_df = pd.DataFrame.from_dict(
        _groups, orient="index", columns=["proteins_with_prediction"]
    )

    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        _groups_df.index.astype(str), _groups_df["proteins_with_prediction"]
    )

    # Add text labels (rounded to 2 decimals)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Prediction source")
    plt.ylabel("% of proteins with at least one GO term predicted")
    plt.title("Prediction coverage")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/prediction_coverage.svg", bbox_inches="tight")
    _groups_df.to_csv(f"{RAW_DIR}/prediction_coverage.csv")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## upset diagram
    """)
    return


@app.cell
def _(mdF100):
    mdF100.query("Score >= 0.2")
    return


@app.cell
def _(dFstruct, mdF100, upset_by_aspect):
    # compare mdF only filtered by self-hits with perfect structures, only on MF
    for _aspect in ["mf"]:
        upset_by_aspect(
            aspect_label=_aspect,
            deepFRI_structure=dFstruct.query("Score >= 0.3"),
            metagenomic_deepFRI=mdF100.query("Score >= 0.3"),
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Venn-diagram mdF vs dF
    """)
    return


@app.cell
def _(plt):
    from matplotlib_venn import venn2


    def venn_with_percentages(
        set_a: set,
        set_b: set,
        label_a: str = "A",
        label_b: str = "B",
        colors: tuple[str, str] = ("#33b1ff", "#08bdba"),
        alpha: float = 0.6,
        offset: float = 0.15,
        overlap_color: str = "#e5f6ff",
        overlap_alpha: float = 0.85,
        save_path: str | None = None,  # optional save path
    ):
        a_only = len(set_a - set_b)
        b_only = len(set_b - set_a)
        both = len(set_a & set_b)
        total = a_only + b_only + both

        # Create figure explicitly
        _fig, _ax = plt.subplots(figsize=(6, 6))

        v = venn2(
            subsets=(a_only, b_only, both),
            set_labels=(label_a, label_b),
            set_colors=colors,
            alpha=alpha,
            ax=_ax,
        )

        def _fmt(n: int) -> str:
            return f"{n}\n({100 * n / total:.1f}%)" if total > 0 else "0\n(0.0%)"

        for region, value in zip(("10", "01", "11"), (a_only, b_only, both)):
            lbl = v.get_label_by_id(region)
            if lbl:
                lbl.set_text(_fmt(value))
                lbl.set_fontsize(16)

        for region, direction in [("10", -1), ("01", 1)]:
            lbl = v.get_label_by_id(region)
            if lbl:
                x, y = lbl.get_position()
                lbl.set_position((x + direction * offset, y))
                lbl.set_ha("right" if direction < 0 else "left")

        overlap = v.get_patch_by_id("11")
        if overlap:
            overlap.set_color(overlap_color)
            overlap.set_alpha(overlap_alpha)

        for patch in v.patches:
            if patch:
                patch.set_edgecolor("black")
                patch.set_linewidth(1.2)

        for label in v.set_labels:
            if label:
                label.set_fontsize(16)
                label.set_fontweight("bold")

        _fig.tight_layout()

        # --- Save if requested ---
        if save_path:
            _fig.savefig(f"{save_path}.svg", bbox_inches="tight")
            _fig.savefig(f"{save_path}.pdf", bbox_inches="tight", dpi=300)

        plt.show()
    return (venn_with_percentages,)


@app.cell
def _():
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, dFstruct, mdF100, mdF50, mdF90, pd, venn_with_percentages):
    mdf_variants = {
        "50": mdF50,
        "90": mdF90,
        "100": mdF100,
    }

    for suffix, mdf_df in mdf_variants.items():
        _mdf_hq = mdf_df.query("Score >= 0.3 & Aspect == 'mf'")
        _dfs_hq = dFstruct.query("Score >= 0.3 & Aspect == 'mf'")

        set_a = set(zip(_mdf_hq["Protein"], _mdf_hq["GO_term/EC"]))
        set_b = set(zip(_dfs_hq["Protein"], _dfs_hq["GO_term/EC"]))

        ### RAW DATA SAVING ###
        all_pairs = sorted(set_a | set_b)
        membership_df = pd.DataFrame(all_pairs, columns=["Protein", "GO_term/EC"])
        membership_df["pair"] = list(
            zip(membership_df["Protein"], membership_df["GO_term/EC"])
        )
        membership_df["in_mdf"] = membership_df["pair"].isin(set_a)
        membership_df["in_dfstruct"] = membership_df["pair"].isin(set_b)
        membership_df = membership_df.drop(columns="pair")
        membership_df.to_csv(
            f"{RAW_DIR}/venn_diagram_mdf{suffix}_dfstruct.csv",
            index=False,
        )

        venn_with_percentages(
            set_a,
            set_b,
            label_a=f"Metagenomic-deepFRI",
            label_b="       deepFRI-structure",
            offset=0.05,
            save_path=f"{PLOT_DIR}/venn_diagram_mdf{suffix}_dfstruct",
        )
    return (membership_df,)


@app.cell
def _(membership_df):
    membership_df
    return


@app.cell
def _(dFstruct, mdF100, mdF50, upset_by_aspect):
    # compare mdF filtered by self-hits AND with identity <90 with perfect structures, for all aspects
    for _aspect in ["mf"]:
        upset_by_aspect(
            aspect_label=_aspect,
            dFstruct=dFstruct,
            mdF50=mdF50,
            mdF100=mdF100,
        )
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF50, upset_by_aspect):
    # compare mdF filtered by self-hits WITH identity <90 AND WITH perfect structures AND WITH sequence only, for all aspects
    for _aspect in ["bp", "mf", "cc"]:
        upset_by_aspect(
            aspect_label=_aspect,
            dFseq=dFseq,
            dFstruct=dFstruct,
            mdF50=mdF50,
            mdF100=mdF100,
        )
    return


@app.cell
def _(mdF100):
    print(mdF100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refinement or noise?
    """)
    return


@app.cell
def _(PLOT_DIR, dFseq, dFstruct, mdF100, mdF90, plt, wang, wsh):
    # define sources once: (key used in uniques/core, label on plot, dataframe)
    _sources = [
        ("seq", "dFseq", dFseq),
        ("struct", "dFstruct", dFstruct),
        ("id90", "mdF90", mdF90),
        ("id100", "mdF100", mdF100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        # 1) Per-protein GO sets for this aspect
        _G = {}
        for _key, _label, _df in _sources:
            _G[_key] = wsh.go_terms_per_protein(
                _df,
                aspect=_asp_code,
                go_col="GO_term/EC",
            )

        # 2) Compute core + unique sets (arbitrary number of sources)
        _core, _uniques = wsh.core_and_unique_per_protein(**_G)

        # 3) Similarity of unique terms to the core, per source
        _data = []
        _labels = []

        for _key, _label, _df in _sources:
            _stc = wsh.unique_to_core_similarity(_core, _uniques[_key], wang)
            _data.append(_stc.dropna())
            _labels.append(_label)

        # 4) Boxplot on this axis
        _bp = _ax.boxplot(_data, tick_labels=_labels, sym=".")

        # annotate medians
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
            _median_val = _y.mean()
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _ax.set_ylim(0.0, 1.0)
        _ax.set_title(f"{_asp_title}")

        if _asp_code == "bp":
            _ax.set_ylabel("Mean similarity of unique GO terms to core")

    _fig.suptitle(
        "Are method-specific GO terms refinements or novelty?",
        y=1.03,
    )
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/refinement_or_noise_methods.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median number of predictions per protein
    """)
    return


@app.cell
def _(PLOT_DIR, dFseq, dFstruct, mdF100, mdF90, np, plt):
    def _get_aspect_counts(counts_series, aspect):
        aspect_lower = str(aspect).lower()
        aspect_index = (
            counts_series.index.get_level_values("Aspect").astype(str).str.lower()
        )
        return counts_series[aspect_index == aspect_lower].values


    _mdf100_hq = mdF100.query("Score >= 0.3")
    _mdf90_hq = mdF90.query("Score >= 0.3")
    _dfstr_hq = dFstruct.query("Score >= 0.3")
    _dfseq_hq = dFseq.query("Score >= 0.2")

    # 1) Counts per (Protein, Aspect)
    counts_dFseq = _dfseq_hq.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()
    counts_dFstruct = _dfstr_hq.groupby(["Protein", "Aspect"])[
        "GO_term/EC"
    ].nunique()
    counts_mdF90 = _mdf90_hq.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()
    counts_mdF100 = _mdf100_hq.groupby(["Protein", "Aspect"])[
        "GO_term/EC"
    ].nunique()

    # 2) Plotting config
    titles = {"bp": "BP", "mf": "MF", "cc": "CC"}
    y_limits = {"bp": 100, "mf": 30, "cc": 20}
    sources = [
        ("dFseq", counts_dFseq),
        ("dFstruct", counts_dFstruct),
        ("mdF90", counts_mdF90),
        ("mdF100", counts_mdF100),
    ]

    # 3) Plot
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    _rng = np.random.default_rng(42)


    def _jitter(vals, xpos, max_points=8000):
        vals = np.asarray(vals)
        if vals.size > max_points:
            vals = _rng.choice(vals, max_points, replace=False)
        x = xpos + _rng.normal(0, 0.04, size=vals.size)
        return x, vals


    for _ax, _asp in zip(_axes, ["bp", "mf", "cc"]):
        _data = [_get_aspect_counts(series, _asp) for _, series in sources]
        _labels = [label for label, _ in sources]

        for _i, _d in enumerate(_data, start=1):
            _xj, _yj = _jitter(np.asarray(_d), _i)
            _ax.scatter(_xj, _yj, s=8, alpha=0.15, zorder=1, linewidths=0)

        _bp = _ax.boxplot(
            _data,
            tick_labels=_labels,
            showmeans=True,
            meanline=True,
            showfliers=False,
            widths=0.55,
            patch_artist=True,
            zorder=2,
            boxprops=dict(linewidth=1.6),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            medianprops=dict(linewidth=2.2),
            meanprops=dict(linewidth=2.0),
        )
        for _box in _bp["boxes"]:
            _box.set_alpha(0.25)

        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _median_val = _median_line.get_ydata().mean()
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.0f}",
                ha="center",
                va="bottom",
                fontsize=13,
                zorder=3,
            )

        _ax.set_title(titles[_asp], fontsize=11)
        _ax.set_ylim(0, y_limits[_asp])
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(axis="x", length=0)
        _ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.25, zorder=0)

        if _asp == "bp":
            _ax.set_ylabel("Number of GO terms")

    _fig.suptitle("Annotation number per protein by source and GO aspect", y=1.03)
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/annotations_per_protein_methods.svg", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max IC per protein
    """)
    return


@app.cell
def _(PLOT_DIR, dFseq, dFstruct, mdF100, mdF50, mdF90, plt):
    ic_dFseq = dFseq.groupby(["Protein", "Aspect"])["IC"].max()
    ic_dFstruct = dFstruct.groupby(["Protein", "Aspect"])["IC"].max()
    ic_mdF50 = mdF50.groupby(["Protein", "Aspect"])["IC"].max()
    ic_mdF90 = mdF90.groupby(["Protein", "Aspect"])["IC"].max()
    ic_mdF100 = mdF100.groupby(["Protein", "Aspect"])["IC"].max()

    ic_dFseq_df = ic_dFseq.reset_index()
    ic_dFstruct_df = ic_dFstruct.reset_index()
    ic_mdF50_df = ic_mdF50.reset_index()
    ic_mdF90_df = ic_mdF90.reset_index()
    ic_mdF100_df = ic_mdF100.reset_index()


    def _ic_by_aspect(df, aspect, aspect_col=None, score_col="IC"):
        aspect_norm = str(aspect).lower()

        # detect aspect column if not given
        if aspect_col is None:
            for col in df.columns:
                if col.lower() == "aspect":
                    aspect_col = col
                    break

        mask = df[aspect_col].astype(str).str.lower() == aspect_norm
        return df.loc[mask, score_col].dropna()


    _sources = [
        ("seq", ic_dFseq_df),
        ("struct", ic_dFstruct_df),
        ("50", ic_mdF50_df),
        ("90", ic_mdF90_df),
        ("100", ic_mdF100_df),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(10, 5), sharey=False)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        # extract IC values for this aspect from all sources
        _data = [
            _ic_by_aspect(df, _asp_code, aspect_col="Aspect", score_col="IC")
            for _, df in _sources
        ]
        _labels = [label for label, _ in _sources]

        # boxplot (capture returned artists)
        _bp = _ax.boxplot(_data, tick_labels=_labels, sym=".")

        # --- annotate medians ---
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
            _median_val = _y.mean()  # y is the median line → two identical values

            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _ax.set_title(_asp_title)

        if _asp_code == "bp":
            _ax.set_ylabel("IC (max per protein)")

    # global figure title
    _fig.suptitle(
        "Per protein Max Information Content (IC) distributions",
        y=1.03,
    )
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/max_ic_per_protein.svg", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median score per protein
    """)
    return


@app.cell
def _(mdF100):
    _mdf100_hq = mdF100.query("Score >= 0.3")
    _mdf100_hq.groupby(["Protein", "Aspect"])["Score"].median()
    return


@app.cell
def _(PLOT_DIR, dFseq, dFstruct, mdF100, mdF90, np, plt):
    def _get_aspect_counts(counts_series, aspect):
        aspect_lower = str(aspect).lower()
        aspect_index = (
            counts_series.index.get_level_values("Aspect").astype(str).str.lower()
        )
        return counts_series[aspect_index == aspect_lower].values


    # no score filtering
    _mdf100_hq = mdF100.query("Score >= 0.0")
    _mdf90_hq = mdF90.query("Score >= 0.0")
    _dfstr_hq = dFstruct.query("Score >= 0.0")
    _dfseq_hq = dFseq.query("Score >= 0.0")

    # 1) Counts per (Protein, Aspect)
    _scores_dFseq = _dfseq_hq.groupby(["Protein", "Aspect"])["Score"].median()
    _scores_dFstruct = _dfstr_hq.groupby(["Protein", "Aspect"])["Score"].median()
    _scores_mdF90 = _mdf90_hq.groupby(["Protein", "Aspect"])["Score"].median()
    _scores_mdF100 = _mdf100_hq.groupby(["Protein", "Aspect"])["Score"].median()

    # 2) Plotting config
    _titles = {"bp": "BP", "mf": "MF", "cc": "CC"}
    _sources = [
        ("dFseq", _scores_dFseq),
        ("dFstruct", _scores_dFstruct),
        ("mdF90", _scores_mdF90),
        ("mdF100", _scores_mdF100),
    ]

    # 3) Plot
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    _rng = np.random.default_rng(42)


    def _jitter(vals, xpos, max_points=8000):
        vals = np.asarray(vals)
        if vals.size > max_points:
            vals = _rng.choice(vals, max_points, replace=False)
        x = xpos + _rng.normal(0, 0.04, size=vals.size)
        return x, vals


    for _ax, _asp in zip(_axes, ["bp", "mf", "cc"]):
        _data = [_get_aspect_counts(series, _asp) for _, series in _sources]
        _labels = [label for label, _ in _sources]

        for _i, _d in enumerate(_data, start=1):
            _xj, _yj = _jitter(np.asarray(_d), _i)
            _ax.scatter(_xj, _yj, s=8, alpha=0.15, zorder=1, linewidths=0)

        _bp = _ax.boxplot(
            _data,
            tick_labels=_labels,
            showmeans=True,
            meanline=True,
            showfliers=False,
            widths=0.55,
            patch_artist=True,
            zorder=2,
            boxprops=dict(linewidth=1.6),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            medianprops=dict(linewidth=2.2),
            meanprops=dict(linewidth=2.0),
        )
        for _box in _bp["boxes"]:
            _box.set_alpha(0.25)

        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _median_val = _median_line.get_ydata().mean()
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=13,
                zorder=3,
            )

        _ax.set_title(_titles[_asp], fontsize=11)
        _ax.set_ylim(0, 1)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(axis="x", length=0)
        _ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.25, zorder=0)

        if _asp == "bp":
            _ax.set_ylabel("Median score")

    _fig.suptitle("Median score per protein by source and GO aspect", y=1.03)
    plt.tight_layout()
    _fig.savefig(f"{PLOT_DIR}/median_score_per_protein_methods.svg", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
