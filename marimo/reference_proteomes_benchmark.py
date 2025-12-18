import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style
    from upsetplot import from_contents, UpSet, plot
    import importlib
    import plotly.express as px
    import plotly.io as pio
    import pandas as pd
    from goatools.obo_parser import GODag
    from pathlib import Path


    # save and display plots in whitemode
    matplotlib.style.use("default")
    pio.templates.default = "plotly"
    return GODag, Path, from_contents, importlib, mo, pd, plot, plt


@app.cell
def _(importlib):
    import wang_similarity_helpers as wsh

    importlib.reload(wsh)
    return (wsh,)


@app.cell
def _():
    # ignore upset plot warnings
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
    return


@app.cell
def _(GODag, pd):
    # information content from SwissProt
    ic_df = pd.read_csv(
        "/home/FilipS/2024/weave_warmup/generated_data/IC_swissprot.csv"
    )

    # GO DAF
    go_dag = GODag("/home/FilipS/2024/weave_warmup/generated_data/go-basic.obo")
    return go_dag, ic_df


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
        "/home/FilipS/software/mdeepfri_source/Metagenomic-DeepFRI/results/reference_proteomes"
    )

    identity_bins = {
        "50": "identity_bin_0.00-0.50",
        "90": "identity_bin_0.00-0.90",
        "100": "identity_bin_0.00-1.00",
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
def _(cov_100, cov_50, cov_90, plt):
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

        fig = plt.figure(figsize=(12, 5))
        plot(
            series,
            subset_size="count",
            show_percentages=True,
            fig=fig,
            element_size=None,
            sort_categories_by="input",
            sort_by="cardinality",
        )
        plt.suptitle(f"Annotation overlap for {aspect_label}")
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
def _(go_terms_100, go_terms_50, go_terms_90, plt, wang, wsh):
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median number of predictions per protein 50 vs 90 vs 100
    """)
    return


@app.cell
def _(go_terms_100, go_terms_50, go_terms_90, plt):
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median IC per protein 50 vs 90 vs 100
    """)
    return


@app.cell
def _(go_terms_100, go_terms_50, go_terms_90, plt):
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median score per protein 50 vs 90 vs 100
    """)
    return


@app.cell
def _(go_terms_100, go_terms_50, go_terms_90, plt):
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cohesion analysis
    """)
    return


@app.cell
def _(go_terms_100, go_terms_50, go_terms_90, plt, wang, wsh):
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
        "/home/FilipS/2025/metagenomic_deepfri/reference_proteomes/sequence_only_deepfri/merged_predictions.csv"
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
        "/home/FilipS/2025/metagenomic_deepfri/reference_proteomes/true_structure_deepfri/merged_predictions.csv"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## coverage histogram
    """)
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF90, pd, plt):
    _n_queries = 14895

    # calculate prediction coverage
    _groups = {
        "dF_seq": len(dFseq["Protein"].unique()) / _n_queries * 100,
        "dF_struct": len(dFstruct["Protein"].unique()) / _n_queries * 100,
        "mdF_90": len(mdF90["Protein"].unique()) / _n_queries * 100,
        "mdF_100": len(mdF100["Protein"].unique()) / _n_queries * 100,
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## upset diagram
    """)
    return


@app.cell
def _(dFstruct, mdF100, upset_by_aspect):
    # compare mdF only filtered by self-hits with perfect structures, only on MF
    for _aspect in ["mf"]:
        upset_by_aspect(
            aspect_label=_aspect,
            dFstruct=dFstruct,
            mdF100=mdF100,
        )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refinment or noise?
    """)
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF90, plt, wang, wsh):
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median number of predictions per protein
    """)
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF90, plt):
    counts_dFseq = dFseq.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()
    counts_dFstruct = dFstruct.groupby(["Protein", "Aspect"])[
        "GO_term/EC"
    ].nunique()
    counts_mdF90 = mdF90.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()
    counts_mdF100 = mdF100.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()


    def _get_aspect_counts(counts_series, aspect):
        aspect_lower = str(aspect).lower()
        aspect_index = (
            counts_series.index.get_level_values("Aspect").astype(str).str.lower()
        )
        return counts_series[aspect_index == aspect_lower].values


    # 1) Counts per (Protein, Aspect)
    counts_dFseq = dFseq.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()
    counts_dFstruct = dFstruct.groupby(["Protein", "Aspect"])[
        "GO_term/EC"
    ].nunique()
    counts_mdF90 = mdF90.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()
    counts_mdF100 = mdF100.groupby(["Protein", "Aspect"])["GO_term/EC"].nunique()

    # 2) Plotting config
    titles = {"bp": "BP", "mf": "MF", "cc": "CC"}
    y_limits = {"bp": 100, "mf": 30, "cc": 20}
    sources = [
        ("seq", counts_dFseq),
        ("struct", counts_dFstruct),
        ("90", counts_mdF90),
        ("100", counts_mdF100),
    ]

    # 3) Plot
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, asp in zip(_axes, ["bp", "mf", "cc"]):
        # extract per-source counts for this aspect
        data = [_get_aspect_counts(series, asp) for _, series in sources]
        _labels = [label for label, _ in sources]

        # draw boxplot and capture artists
        _bp = ax.boxplot(data, tick_labels=_labels, sym=".")

        # annotate medians
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
            _median_val = _y.mean()  # median value
            ax.text(
                _i,
                _median_val,
                f"{_median_val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(titles[asp])
        ax.set_ylim(0, y_limits[asp])

        if asp == "bp":
            ax.set_ylabel("Per protein median number of go-terms")

    _fig.suptitle(
        "Median annotation number per protein by source and GO aspect", y=1.03
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median IC per protein
    """)
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF50, mdF90, plt):
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median score per protein
    """)
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF90, plt):
    score_dFseq = dFseq.groupby(["Protein", "Aspect"])["Score"].median()
    score_dFstruct = dFstruct.groupby(["Protein", "Aspect"])["Score"].median()
    score_mdF90 = mdF90.groupby(["Protein", "Aspect"])["Score"].median()
    score_mdF100 = mdF100.groupby(["Protein", "Aspect"])["Score"].median()


    def _per_protein_median_by_aspect(s, aspect):
        """
        s: Series with MultiIndex (Protein, Aspect) and values = per-protein median Score
        aspect: "bp", "mf", or "cc"
        """
        aspect_norm = str(aspect).lower()
        # assume second level of index is Aspect
        aspects = s.index.get_level_values("Aspect").astype(str).str.lower()
        mask = aspects == aspect_norm
        return s[mask].dropna().values


    _sources = [
        ("seq", score_dFseq),
        ("struct", score_dFstruct),
        ("90", score_mdF90),
        ("100", score_mdF100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        # per-protein median scores for this aspect from all sources
        _data = [_per_protein_median_by_aspect(s, _asp_code) for _, s in _sources]
        _labels = [label for label, _ in _sources]

        # boxplot (capture returned artists)
        _bp = _ax.boxplot(_data, tick_labels=_labels, sym=".")

        # --- annotate medians ---
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
            _median_val = _y.mean()  # median line

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
            _ax.set_ylabel("Score (per-protein median)")

    _fig.suptitle(
        "Per protein median prediction score distribution",
        y=1.03,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cohesion analysis
    """)
    return


@app.cell
def _(dFseq, dFstruct, mdF100, mdF90, plt, wang, wsh):
    _sources = [
        ("seq", dFseq),
        ("struct", dFstruct),
        ("90", mdF90),
        ("100", mdF100),
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
                _df, _asp_code, wang, go_col="GO_term/EC"
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
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
