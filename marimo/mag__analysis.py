import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style
    import importlib
    from pathlib import Path
    from upsetplot import from_contents, UpSet, plot
    from goatools.obo_parser import GODag
    import os
    from scipy.stats import mannwhitneyu

    # save and display plots in whitemode
    matplotlib.style.use("default")
    return (
        GODag,
        Path,
        from_contents,
        importlib,
        mannwhitneyu,
        mo,
        np,
        pd,
        plot,
        plt,
    )


@app.cell
def _():
    # ignore upset plot warnings
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
    return


@app.cell
def _(importlib):
    import helpers.wang_similarity_helpers as wsh

    importlib.reload(wsh)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MAG selection from UHGP
    """)
    return


@app.cell
def _(pd):
    # load the metadata and choose high quality MAGs
    df = pd.read_csv("data/external/genomes-all_metadata.tsv", sep="\t")
    df_mags = df.query(
        "Genome_type=='MAG' & Completeness > 95 & Contamination < 1 & N_contigs < 100 & Genome==Species_rep"
    )

    # choose random genomes from different species
    species_ids = (
        df_mags["Species_rep"]
        .drop_duplicates()
        .sample(n=50, random_state=42)  # random species
    )

    selected_genomes = (
        df_mags[df_mags["Species_rep"].isin(species_ids)]
        .groupby("Species_rep")
        .apply(lambda x: x.sample(n=1))  # random genome for a selected species
        .reset_index(drop=True)
    )

    # get FTP accessions for eggNOG data and fasta
    _BASE = "https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v2.0.2/species_catalogue"


    def build_urls(sid: str):
        sid_prefix = sid[:-2]
        dir_url = f"{_BASE}/{sid_prefix}/{sid}/genome/"
        return {
            "Species_rep": sid,
            "faa_url": f"{dir_url}{sid}.faa",
            "eggnog_url": f"{dir_url}{sid}_eggNOG.tsv",
        }


    species_ids = selected_genomes["Species_rep"].unique()

    url_rows = [build_urls(sid) for sid in species_ids]

    urls_df = pd.DataFrame(url_rows)

    # write wget script to download those files
    with open("scripts/download_mgyg_50.sh", "w") as out:
        out.write("#!/usr/bin/env bash\n\n")
        for sid in species_ids:
            sid_prefix = sid[:-2]
            dir_url = f"{_BASE}/{sid_prefix}/{sid}/genome/"
            faa_url = f"{dir_url}{sid}.faa"
            eggnog_url = f"{dir_url}{sid}_eggNOG.tsv"
            out.write(
                f"wget -c '{faa_url}' -O ../data/source/uhgp_mags/uhgp_50_mags/{sid}.faa\n"
            )
            out.write(
                f"wget -c '{eggnog_url}' -O ../data/source/uhgp_mags/uhgp_50_mags/{sid}_eggNOG.tsv\n"
            )
    return df, selected_genomes


@app.cell
def _(selected_genomes):
    # helper to extract phylogeny
    def extract_rank(lineage, rank):
        for field in lineage.split(";"):
            if field.startswith(rank):
                return field.replace(rank, "")
        return None


    _df = selected_genomes.copy()
    _df["Phylum"] = selected_genomes["Lineage"].apply(
        lambda x: extract_rank(x, "p__")
    )
    _df["Class"] = selected_genomes["Lineage"].apply(
        lambda x: extract_rank(x, "c__")
    )
    _df["Species"] = selected_genomes["Lineage"].apply(
        lambda x: extract_rank(x, "s__")
    )

    _out = _df[
        [
            "Species_rep",
            "Genome_type",
            "Completeness",
            "Contamination",
            "N_contigs",
            "Phylum",
            "Class",
            "Species",
        ]
    ]
    _out

    # Completeness > 95 & Contamination < 1 & N_contigs < 100
    return (extract_rank,)


@app.cell
def _(df, extract_rank):
    _df = df.copy()
    _df["Class"] = df["Lineage"].apply(lambda x: extract_rank(x, "c__"))
    _df["Class"].unique()
    return


@app.cell
def _(df, extract_rank):
    # previous pseudo-mags
    _old_mags2 = df.query(
        "Genome in ["
        "'MGYG000000018', "
        "'MGYG000000151', "
        "'MGYG000001132', "
        "'MGYG000001373', "
        "'MGYG000001551', "
        "'MGYG000001803', "
        "'MGYG000002242', "
        "'MGYG000002893', "
        "'MGYG000003379', "
        "'MGYG000004694'"
        "]"
    )

    _old_mags2["Phylum"] = _old_mags2["Lineage"].apply(
        lambda x: extract_rank(x, "p__")
    )
    _old_mags2["Class"] = _old_mags2["Lineage"].apply(
        lambda x: extract_rank(x, "c__")
    )
    _old_mags2["Species"] = _old_mags2["Lineage"].apply(
        lambda x: extract_rank(x, "s__")
    )

    _out = _old_mags2[
        [
            "Species_rep",
            "Genome_type",
            "Completeness",
            "Contamination",
            "N_contigs",
            "Phylum",
            "Class",
            "Species",
        ]
    ]
    _out

    # Completeness > 95 & Contamination < 1 & N_contigs < 100
    return


@app.cell
def _(df):
    df.query("Genome == 'MGYG000000378'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Helpers
    """)
    return


@app.cell
def _(GODag, pd):
    # (https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data) IA.txt
    # ic_df = pd.read_csv("data/external/IA.txt", sep="\t", names=["go_term", "IC"])

    # information content from SwissProt
    ic_df = pd.read_csv("data/external/IC_swissprot.csv")

    obo = "data/external/go-basic-latest.obo"

    godag = GODag(
        obo,
        optional_attrs={"namespace", "is_obsolete", "replaced_by", "consider"},
        load_obsolete=True,
    )
    return godag, ic_df


@app.cell
def _(godag, pd):
    ### aspect retrieval helper ###
    def resolve_aspect(go_id: str):
        """Return aspect 'bp'/'mf'/'cc' for GO ID, resolving obsolete/alt_id where possible."""
        if not isinstance(go_id, str) or go_id.strip() == "" or go_id == "-":
            return pd.NA
        go_id = go_id.strip()

        # Resolve alt_id -> primary id if needed
        go_id = alt2id.get(go_id, go_id)

        term = godag.get(go_id)
        if term is None:
            return pd.NA

        # If namespace is present, use it (works for most obsolete terms too)
        ns = getattr(term, "namespace", None)
        if ns in ns2aspect:
            return ns2aspect[ns]

        # Otherwise, try replaced_by then consider (best-effort)
        for rep in getattr(term, "replaced_by", None) or []:
            rep = alt2id.get(rep, rep)
            t2 = godag.get(rep)
            ns = getattr(t2, "namespace", None) if t2 else None
            if ns in ns2aspect:
                return ns2aspect[ns]

        for cand in getattr(term, "consider", None) or []:
            cand = alt2id.get(cand, cand)
            t2 = godag.get(cand)
            ns = getattr(t2, "namespace", None) if t2 else None
            if ns in ns2aspect:
                return ns2aspect[ns]

        return pd.NA


    ns2aspect = {
        "biological_process": "bp",
        "molecular_function": "mf",
        "cellular_component": "cc",
    }

    alt2id = getattr(
        godag, "alt2id", {}
    )  # alt_id -> primary id mapping (if present)
    return (resolve_aspect,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MAG data analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dFseq and EggNOG loading and aspect fill
    """)
    return


@app.cell
def _(ic_df, pd, resolve_aspect):
    # sequence only deepFRI
    dFseq = pd.read_csv(
        "/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/sequence_only_deepfri/merged_predictions.csv"
    )
    dFseq = dFseq.rename(columns={"GO_term/EC": "go_term"})

    dFseq = pd.merge(
        dFseq,
        ic_df,
        left_on="go_term",
        right_on="go_term",
        how="left",
    )

    ### add missing aspect for dFseq ###
    _valid_terms_dfseq = (
        dFseq["go_term"].dropna().loc[lambda s: s.ne("-")].astype(str).unique()
    )

    go2aspect_dfseq = {t: resolve_aspect(t) for t in _valid_terms_dfseq}
    dFseq["aspect"] = dFseq["go_term"].map(go2aspect_dfseq)


    # EGGnog go-terms
    eggnog = pd.read_csv(
        "/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/all_eggNOG.tsv",
        sep="\t",
    )
    eggnog = eggnog.rename(columns={"#query": "Protein", "GOs": "go_term"})[
        ["Protein", "go_term"]
    ]

    eggnog = (
        eggnog.assign(go_term=eggnog["go_term"].str.split(","))
        .explode("go_term")
        .reset_index(drop=True)
    )

    eggnog = pd.merge(
        eggnog,
        ic_df,
        left_on="go_term",
        right_on="go_term",
        how="left",
    )

    ### add missing aspect for eggnog ###
    _valid_terms_egg = (
        eggnog["go_term"].dropna().loc[lambda s: s.ne("-")].astype(str).unique()
    )

    go2aspect_egg = {t: resolve_aspect(t) for t in _valid_terms_egg}
    eggnog["aspect"] = eggnog["go_term"].map(go2aspect_egg)

    # load deepgo predictions

    deep_go = pd.read_csv(
        "data/source/uhgp_mags/deep_go/uhgp_10_mags_all_aspects.tsv",
        sep="\t",
    )

    deep_go = pd.merge(
        deep_go,
        ic_df,
        left_on="go_term",
        right_on="go_term",
        how="left",
    ).rename(columns={"Aspect": "aspect"})
    return dFseq, deep_go, eggnog


@app.cell
def _(deep_go):
    # Investigate NaNs
    _summary = deep_go.agg(
        n_rows=("IC", "size"),
        n_nan_rows=("IC", lambda s: s.isna().sum()),
        n_unique_go=("go_term", "nunique"),
        n_unique_go_with_nan_ic=(
            "go_term",
            lambda s: s[deep_go.loc[s.index, "IC"].isna()].nunique(),
        ),
    )

    _summary
    return


@app.cell
def _():
    # calculate how many sequences we have
    _original_fasta = "data/source/uhgp_mags/uhgp_10_mag.faa"
    with open(_original_fasta) as _f:
        all_sequences = sum(1 for line in _f if line.startswith(">"))
    return (all_sequences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mdF data loading
    """)
    return


@app.cell
def _(Path, ic_df, pd):
    _BASE = Path(
        "/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/uhgp_10_mags_fix"
    )

    identity_bins = {
        "100": "identity_bin_0.00-1.01",
    }

    db_files = {
        "pdb": "pdb100_230517_alignment_scores.tsv",
        "uniprot": "afdb_uniprot_v4_alignment_scores.tsv",
        "esm": "highquality_clust30_alignment_scores.tsv",
    }

    results_file = "generate_contacts_2/results.tsv"


    def load_alignment_sets(bin_key):
        bin_dir = _BASE / "pyopal_alignments" / identity_bins[bin_key]
        res_dir = _BASE / "results" / identity_bins[bin_key]

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
    return (load_alignment_sets,)


@app.cell
def _(load_alignment_sets):
    pyopal_100, mdF100 = load_alignment_sets("100")
    return mdF100, pyopal_100


@app.cell
def _(all_sequences, pyopal_100):
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


    cov_100 = compute_coverage(pyopal_100, total_queries=all_sequences)
    cov_100
    return (cov_100,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Database composition
    """)
    return


@app.cell
def _(all_sequences, cov_100, plt):
    from matplotlib.ticker import FuncFormatter


    def decompose_cov(cov):
        total = cov["total_queries"]
        pdb = cov["pdb_only"]
        uni = cov["pdb_plus_uniprot"] - pdb
        esm = cov["pdb_plus_uniprot_plus_esm"] - (pdb + uni)
        no_hit = total - (pdb + uni + esm)
        return pdb, uni, esm, no_hit


    title = ""

    fig, ax = plt.subplots(figsize=(5, 5))

    pdb, uni, esm, no_hit = decompose_cov(cov_100)

    labels = [""]

    ax.bar(labels, [pdb], label="PDB hits")
    ax.bar(labels, [uni], bottom=[pdb], label="+UniProt hits")
    ax.bar(labels, [esm], bottom=[pdb + uni], label="+ESM hits")
    ax.bar(labels, [no_hit], bottom=[pdb + uni + esm], label="No hit")

    ax.set_title(title)
    ax.set_ylabel("Fraction of queries")
    ax.set_yticks(
        [
            0,
            0.25 * all_sequences,
            0.5 * all_sequences,
            0.75 * all_sequences,
            all_sequences,
        ]
    )

    # --- percent y-axis relative to all_sequences ---
    ax.set_ylim(0, all_sequences)

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{100 * y / all_sequences:.0f}%")
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{100 * y / all_sequences:.0f}% ({int(y):,})")
    )
    # Legend outside
    # Figure-level title
    fig.suptitle("Database hits distribution", fontsize=14, y=1.08)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0)
    )


    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, pd, plt, pyopal_100):
    def hit_coverage_curve(
        df: pd.DataFrame, thresholds: np.ndarray
    ) -> pd.DataFrame:
        """
        For each identity threshold t, compute the fraction of unique queries
        that have at least one hit with identity >= t.
        """
        # unique queries in this DB (denominator)
        n_queries = df["query_name"].nunique()

        # for speed: per-query best identity (max over hits)
        max_id_per_query = df.groupby("query_name")["identity"].max()

        out = []
        for t in thresholds:
            covered = (max_id_per_query >= t).sum()
            out.append(
                {
                    "identity_threshold": t,
                    "covered_queries": covered,
                    "coverage_frac": covered / n_queries,
                }
            )

        return pd.DataFrame(out)


    # choose thresholds (plot from high -> low identity)
    thresholds = np.round(np.linspace(1.0, 0.2, 81), 3)

    curves = {}
    for _name, _df in pyopal_100.items():
        curves[_name] = hit_coverage_curve(_df, thresholds)

    plt.figure(figsize=(7, 5))

    for name, curve in curves.items():
        plt.plot(curve["identity_threshold"], curve["coverage_frac"], label=name)

    plt.gca().invert_xaxis()  # identity decreases left->right? invert so "relaxing" goes to the right
    plt.xlabel("Identity threshold")
    plt.ylabel("Fraction of queries with a hit")
    plt.title("Relaxing identity cutoff increases coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## go-term coverage
    """)
    return


@app.cell
def _(all_sequences, dFseq, deep_go, eggnog, mdF100, pd, plt):
    def proteins_with_go(df):
        """
        Count unique proteins that have at least one valid GO term.
        A valid GO term is: not NaN and not '-'.
        """
        return df.loc[
            df["go_term"].notna() & (df["go_term"] != "-"), "Protein"
        ].nunique()


    _mdf100_hq = mdF100.query("Score >= 0.3")
    _dFseq_hq = dFseq.query("Score >= 0.2")
    _deep_go_hq = deep_go.query("Score >= 0.3")

    _groups = {
        "eggnog": proteins_with_go(eggnog) / all_sequences * 100,
        "sequence-dF": proteins_with_go(_dFseq_hq) / all_sequences * 100,
        "meta-dF": proteins_with_go(_mdf100_hq) / all_sequences * 100,
        "deep_go": proteins_with_go(_deep_go_hq) / all_sequences * 100,
    }

    _groups_df = pd.DataFrame.from_dict(
        _groups, orient="index", columns=["proteins_with_prediction"]
    )

    # --- customize colors here ---
    _colors = [
        "#ff9245",  # eggnog
        "#33b1ff",  # sequence-dF
        "#3ddbd9",  # meta-dF
        "#ff8389",  # deep_go
    ]

    plt.figure(figsize=(5, 5))
    bars = plt.bar(
        _groups_df.index.astype(str),
        _groups_df["proteins_with_prediction"],
        color=_colors,
        edgecolor="black",
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
    plt.ylabel("% proteins with ≥ 1 prediction")
    plt.title("Prediction coverage")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upset plots
    """)
    return


@app.cell
def _(dFseq, eggnog, from_contents, mdF100, plot, plt):
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

        fig = plt.figure(figsize=(12, 3))
        plot(
            series,
            subset_size="count",
            show_percentages=True,
            fig=fig,
            element_size=None,
            sort_categories_by="input",
        )
        plt.suptitle(f"Overlap of Protein–GO term annotations ({aspect_label})")
        plt.show()


    for _aspect in ["bp", "mf", "cc"]:
        upset_by_aspect(
            aspect_label=_aspect,
            eggnog=eggnog,
            dFseq=dFseq,
            mdF100=mdF100,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median number of predictions per protein
    """)
    return


@app.cell
def _(dFseq, eggnog, mdF100, plt):
    def _get_aspect_counts(counts_series, aspect):
        aspect_lower = str(aspect).lower()
        aspect_index = (
            counts_series.index.get_level_values("aspect").astype(str).str.lower()
        )
        return counts_series[aspect_index == aspect_lower].values


    # 1) Counts per (Protein, Aspect)
    counts_eggnog = eggnog.groupby(["Protein", "aspect"])["go_term"].nunique()
    counts_dFseq = dFseq.groupby(["Protein", "aspect"])["go_term"].nunique()
    counts_mdF100 = mdF100.groupby(["Protein", "aspect"])["go_term"].nunique()

    # 2) Plotting config
    titles = {"bp": "BP", "mf": "MF", "cc": "CC"}
    y_limits = {"bp": 100, "mf": 30, "cc": 20}
    sources = [
        ("eggnog", counts_eggnog),
        ("dFseq", counts_dFseq),
        ("mdF", counts_mdF100),
    ]

    # 3) Plot
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for _ax, asp in zip(_axes, ["bp", "mf", "cc"]):
        # extract per-source counts for this aspect
        data = [_get_aspect_counts(series, asp) for _, series in sources]
        _labels = [label for label, _ in sources]

        # draw boxplot and capture artists
        _bp = _ax.boxplot(data, tick_labels=_labels, sym=".")

        # annotate medians
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
            _median_val = _y.mean()  # median value
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _ax.set_title(titles[asp])
        _ax.set_ylim(0, y_limits[asp])

        if asp == "bp":
            _ax.set_ylabel("Per protein median number of go-terms")

    _fig.suptitle(
        "Median annotation number per protein by source and GO aspect", y=1.03
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### p-value helpers
    """)
    return


@app.cell
def _():
    def p_to_stars(p):
        if p < 1e-4:
            return "****"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "ns"


    import matplotlib.transforms as mtransforms


    def add_sig_bracket_axes(
        ax, x1, x2, y_ax, h_ax, stars, stars_offset_pts=1, lw=1.0
    ):
        """
        Draw bracket between x1 and x2 (data-x), at y_ax (axes-y fraction),
        with height h_ax (axes-y fraction). Stars are placed above.
        """
        trans = ax.get_xaxis_transform()  # x in data, y in axes fraction

        # bracket line (clip off so it can go above the axes)
        ax.plot(
            [x1, x1, x2, x2],
            [y_ax, y_ax + h_ax, y_ax + h_ax, y_ax],
            transform=trans,
            color="black",
            lw=lw,
            clip_on=False,
        )

        # stars text, offset in points so it won't collide with other text
        text_trans = trans + mtransforms.ScaledTranslation(
            0, stars_offset_pts / 72.0, ax.figure.dpi_scale_trans
        )
        ax.text(
            (x1 + x2) / 2,
            y_ax + h_ax,
            stars,
            transform=text_trans,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )


    def add_n_text_axes(ax, x1, x2, y_ax, h_ax, n_pairs, n_offset_pts=-2):
        trans = ax.get_xaxis_transform()
        text_trans = trans + mtransforms.ScaledTranslation(
            0, n_offset_pts / 72.0, ax.figure.dpi_scale_trans
        )
        ax.text(
            (x1 + x2) / 2,
            y_ax + h_ax,
            f"n={n_pairs}",
            transform=text_trans,
            ha="center",
            va="top",
            fontsize=6,
            clip_on=False,
        )
    return add_n_text_axes, add_sig_bracket_axes, p_to_stars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max IC per protein
    """)
    return


@app.cell
def _(
    add_sig_bracket_axes,
    dFseq,
    deep_go,
    eggnog,
    mannwhitneyu,
    mdF100,
    np,
    p_to_stars,
    plt,
):
    from scipy.stats import wilcoxon

    # --- max-IC-per-protein tables ---
    ic_eggnog = eggnog.groupby(["Protein", "aspect"])["IC"].max().reset_index()
    ic_dFseq_s20 = (
        dFseq.query("Score >= 0.2")
        .groupby(["Protein", "aspect"])["IC"]
        .max()
        .reset_index()
    )
    ic_mdF100_s30 = (
        mdF100.query("Score >= 0.3")
        .groupby(["Protein", "aspect"])["IC"]
        .max()
        .reset_index()
    )
    ic_deepgo_s30 = (
        deep_go.query("Score >= 0.3")
        .groupby(["Protein", "aspect"])["IC"]
        .max()
        .reset_index()
    )

    SOURCES = [
        ("eggnog", ic_eggnog),
        ("dF_hq", ic_dFseq_s20),
        ("mdF_hq", ic_mdF100_s30),
        ("deep_go", ic_deepgo_s30),
    ]
    ASPECTS = [("bp", "BP"), ("mf", "MF"), ("cc", "CC")]
    PAIRS = [(1, 3), (2, 3), (3, 4)]


    def ic_series(df, aspect_code):
        mask = df["aspect"].astype(str).str.lower().eq(str(aspect_code).lower())
        return df.loc[mask, "IC"].dropna()


    def proteins_in_aspect(df, aspect_code):
        mask = df["aspect"].astype(str).str.lower().eq(str(aspect_code).lower())
        return set(df.loc[mask, "Protein"].dropna().unique())


    def filter_to_proteins(df, proteins, aspect_code):
        mask = df["aspect"].astype(str).str.lower().eq(
            str(aspect_code).lower()
        ) & df["Protein"].isin(proteins)
        return df.loc[mask].copy()


    def paired_arrays(df_a, df_b, aspect_code):
        mask_a = (
            df_a["aspect"].astype(str).str.lower().eq(str(aspect_code).lower())
        )
        mask_b = (
            df_b["aspect"].astype(str).str.lower().eq(str(aspect_code).lower())
        )
        a = df_a.loc[mask_a, ["Protein", "IC"]]
        b = df_b.loc[mask_b, ["Protein", "IC"]]
        m = a.merge(b, on="Protein", how="inner", suffixes=("_a", "_b")).dropna()
        return m["IC_a"].to_numpy(), m["IC_b"].to_numpy(), len(m)


    def mwu_with_rbc(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            return np.nan, np.nan
        res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        U = res.statistic
        n1, n2 = len(x), len(y)
        rbc = 1.0 - (2.0 * U) / (n1 * n2)
        return res.pvalue, rbc


    def draw_violin(ax, data, labels):
        pos = np.arange(1, len(data) + 1)
        vp = ax.violinplot(
            [np.asarray(d) for d in data],
            positions=pos,
            widths=0.75,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor("#1192e8")
            body.set_edgecolor("black")
            body.set_alpha(0.50)
            body.set_linewidth(1.0)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=30, ha="center")
        # median annotations
        for i, d in enumerate(data, start=1):
            d = np.asarray(d)
            if d.size:
                med = float(np.nanmedian(d))
                ax.text(i, med, f"{med:.1f}", ha="center", va="bottom", fontsize=9)
        return pos


    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7, 7), sharey="row")
    labels = [name for name, _ in SOURCES]

    for col, (asp_code, asp_title) in enumerate(ASPECTS):
        # Row 0: Global (unpaired Mann–Whitney U)
        ax0 = axes[0, col]
        global_series = [ic_series(df, asp_code) for _, df in SOURCES]
        draw_violin(ax0, [s.to_numpy() for s in global_series], labels)
        ax0.set_title(asp_title)
        if col == 0:
            ax0.set_ylabel("IC (max per protein)")
        # Add n= labels
        ymin = ax0.get_ylim()[0]
        for i, n in enumerate([len(s) for s in global_series], start=1):
            ax0.text(
                i,
                ymin - 2.5,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=9,
                rotation=30,
            )

        # Statistical tests and brackets
        for k, (i, j) in enumerate(PAIRS):
            p, _ = mwu_with_rbc(
                global_series[i - 1].to_numpy(), global_series[j - 1].to_numpy()
            )
            stars = p_to_stars(p) if np.isfinite(p) else "n/a"
            y_ax = 0.75 + k * 0.08
            add_sig_bracket_axes(
                ax0, i, j, y_ax=y_ax, h_ax=0.03, stars=stars, stars_offset_pts=0
            )

        # Row 1: Common proteins (paired Wilcoxon)
        ax1 = axes[1, col]
        prot_sets = [proteins_in_aspect(df, asp_code) for _, df in SOURCES]
        common = set.intersection(*prot_sets) if prot_sets else set()
        common_dfs = [
            (name, filter_to_proteins(df, common, asp_code))
            for name, df in SOURCES
        ]
        common_series = [ic_series(df, asp_code) for _, df in common_dfs]
        draw_violin(ax1, [s.to_numpy() for s in common_series], labels)
        ax1.set_title(asp_title)
        if col == 0:
            ax1.set_ylabel("IC (max per protein)")
        # Add n= labels
        ymin = ax1.get_ylim()[0]
        for i, n in enumerate([len(s) for s in common_series], start=1):
            ax1.text(
                i,
                ymin - 2.5,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=9,
                rotation=30,
            )

        # Statistical tests and brackets
        for k, (i, j) in enumerate(PAIRS):
            _, df_i = common_dfs[i - 1]
            _, df_j = common_dfs[j - 1]
            a, b, n_pairs = paired_arrays(df_i, df_j, asp_code)
            if n_pairs == 0:
                stars = "n/a"
            else:
                diffs = a - b
                if np.allclose(diffs, 0):
                    p = 1.0
                else:
                    p = wilcoxon(
                        a, b, alternative="two-sided", zero_method="wilcox"
                    ).pvalue
                stars = p_to_stars(p)
            y_ax = 0.75 + k * 0.08
            add_sig_bracket_axes(
                ax1, i, j, y_ax=y_ax, h_ax=0.03, stars=stars, stars_offset_pts=0
            )

    fig.suptitle("Per-protein max IC comparison", fontsize=14, y=1.02)
    fig.text(
        0.25,
        0.99,
        "Global (unpaired Mann–Whitney U)",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.10,
        0.48,
        "Common proteins across all methods (paired Wilcoxon)",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.00,
        right=0.98,
        bottom=0.10,
        top=0.92,
        wspace=0.05,  # horizontal spacing between columns
        hspace=0.55,  # vertical spacing between rows
    )
    plt.savefig("plots/max_ic_distribution.svg", format="svg")
    plt.show()
    return (wilcoxon,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IC / score perturbations
    """)
    return


@app.cell
def _(dFseq, eggnog, mannwhitneyu, mdF100, np, pd, plt):
    def apply_score_filter(df, min_score=None, score_col="Score"):
        if min_score is None:
            return df
        return df.loc[df[score_col] >= min_score].copy()


    # --- cell-local config ---
    _conditions = [
        ("all", None),
        (">=0.2", 0.2),
        (">=0.3", 0.3),
    ]

    _methods = {
        "dFseq": dFseq,
        "mdF": mdF100,
    }

    _ic_tables = []

    # --- eggNOG: compute once, then replicate across conditions for consistent faceting ---
    _ic_eggnog_base = (
        eggnog.groupby(["Protein", "aspect"], as_index=False)["IC"]
        .max()
        .assign(method="eggnog")
    )

    for _label, _thr in _conditions:
        _ic_tables.append(_ic_eggnog_base.assign(condition=_label))

    # --- ML methods: filter -> groupby -> max(IC) ---
    for _method, _df in _methods.items():
        for _label, _thr in _conditions:
            _df_f = apply_score_filter(_df, _thr, score_col="Score")
            _ic = (
                _df_f.groupby(["Protein", "aspect"], as_index=False)["IC"]
                .max()
                .assign(method=_method, condition=_label)
            )
            _ic_tables.append(_ic)

    # --- big reusable output (no underscore) ---
    ic_all = pd.concat(_ic_tables, ignore_index=True)

    # Optional: normalize aspect strings defensively (keeps later plotting sane)
    ic_all["aspect"] = ic_all["aspect"].astype(str).str.lower()

    # --- cell-local config ---
    _methods = ["eggnog", "dFseq", "mdF"]
    _aspects = [("bp", "BP"), ("mf", "MF"), ("cc", "CC")]
    _conds = ["all", ">=0.2", ">=0.3"]
    _pairs = [(1, 2), (1, 3), (2, 3)]
    _pair_colors = {(1, 2): "#1f77b4", (1, 3): "#ff7f0e", (2, 3): "#2ca02c"}


    def _p_to_stars(p):
        return (
            "****"
            if p < 1e-4
            else "***"
            if p < 1e-3
            else "**"
            if p < 1e-2
            else "*"
            if p < 0.05
            else "ns"
        )


    def _series(_m, _a, _c):
        return ic_all.loc[
            (ic_all["method"] == _m)
            & (ic_all["aspect"].astype(str).str.lower() == _a)
            & (ic_all["condition"] == _c),
            "IC",
        ].dropna()


    def _bracket(ax, x1, x2, y, h, stars, color):
        ax.plot(
            [x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, c=color, clip_on=False
        )
        ax.text(
            (x1 + x2) / 2,
            y + h,
            stars,
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            clip_on=False,
        )


    # --- plot ---
    _fig, _axes = plt.subplots(
        len(_conds), len(_aspects), figsize=(11, 9), sharey=False
    )

    for _r, _cond in enumerate(_conds):
        for _c, (_asp, _title) in enumerate(_aspects):
            _ax = _axes[_r, _c]

            _data = [_series(_m, _asp, _cond).to_numpy() for _m in _methods]
            _bp = _ax.boxplot(_data, tick_labels=_methods, sym=".", widths=0.55)

            if _r == 0:
                _ax.set_title(_title)
            if _c == 0:
                _ax.set_ylabel(f"IC (max per protein)\n{_cond}")

            _mx = max(np.nanmax(d) if len(d) else np.nan for d in _data)
            _mn = min(np.nanmin(d) if len(d) else np.nan for d in _data)
            _rng = (_mx - _mn) if _mx > _mn else 1.0

            # bottom padding for n=
            _ax.set_ylim(_mn - 0.14 * _rng, _mx + 0.02 * _rng)

            # medians + n
            for _i, _med in enumerate(_bp["medians"], start=1):
                _v = float(np.mean(_med.get_ydata()))
                _ax.text(_i, _v, f"{_v:.1f}", ha="center", va="bottom", fontsize=8)
                _ax.text(
                    _i,
                    _mn - 0.075 * _rng,
                    f"n={len(_data[_i - 1])}",
                    ha="center",
                    va="top",
                    fontsize=7,
                )

            # significance brackets (more vertical room, non-overlapping)
            _yr = np.ptp(_ax.get_ylim()) or 1.0
            _y0 = _mx + 0.05 * _yr
            _step = 0.10 * _yr  # increased spacing
            _h = 0.02 * _yr

            for _k, (_i, _j) in enumerate(_pairs):
                _a, _b = _data[_i - 1], _data[_j - 1]
                _p = (
                    mannwhitneyu(_a, _b, alternative="two-sided").pvalue
                    if (len(_a) and len(_b))
                    else np.nan
                )
                _stars = _p_to_stars(_p) if np.isfinite(_p) else "na"
                _bracket(
                    _ax,
                    _i,
                    _j,
                    _y0 + _k * _step,
                    _h,
                    _stars,
                    _pair_colors[(_i, _j)],
                )

            _ax.set_ylim(top=_y0 + len(_pairs) * _step + 0.06 * _yr)

    _fig.suptitle(
        "Per-protein max IC by method, aspect, and score filtering", y=0.995
    )
    _fig.tight_layout()
    _fig
    return (ic_all,)


@app.cell
def _(ic_all):
    ic_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median score per protein
    """)
    return


@app.cell
def _(
    add_n_text_axes,
    add_sig_bracket_axes,
    dFseq,
    mdF100,
    np,
    p_to_stars,
    plt,
    wilcoxon,
):
    def plot_median_score_violins_paired(dFseq_df, mdF100_df):
        # per-protein median score tables (local)
        score_dfseq_local = (
            dFseq_df.groupby(["Protein", "aspect"])["Score"].median().reset_index()
        )
        score_mdf_local = (
            mdF100_df.groupby(["Protein", "aspect"])["Score"]
            .median()
            .reset_index()
        )

        sources_local = [
            ("dFseq", score_dfseq_local),
            ("mdF", score_mdf_local),
        ]
        aspects_local = [("bp", "BP"), ("mf", "MF"), ("cc", "CC")]

        def _scores_by_aspect_local(
            df_local,
            aspect_code_local,
            protein_col_local="Protein",
            aspect_col_local="aspect",
            score_col_local="Score",
        ):
            mask_local = (
                df_local[aspect_col_local].astype(str).str.lower()
                == str(aspect_code_local).lower()
            )
            return df_local.loc[
                mask_local, [protein_col_local, score_col_local]
            ].dropna()

        def _paired_scores_local(
            df_a_local,
            df_b_local,
            aspect_code_local,
            protein_col_local="Protein",
            aspect_col_local="aspect",
            score_col_local="Score",
        ):
            a_local = _scores_by_aspect_local(
                df_a_local,
                aspect_code_local,
                protein_col_local,
                aspect_col_local,
                score_col_local,
            ).rename(columns={score_col_local: "a"})
            b_local = _scores_by_aspect_local(
                df_b_local,
                aspect_code_local,
                protein_col_local,
                aspect_col_local,
                score_col_local,
            ).rename(columns={score_col_local: "b"})
            merged_local = a_local.merge(
                b_local, on=protein_col_local, how="inner"
            ).dropna()
            return (
                merged_local["a"].to_numpy(),
                merged_local["b"].to_numpy(),
                len(merged_local),
            )

        # bracket layout (LOCAL names)
        br_y0_local = 0.9
        br_step_local = 0.09
        br_h_local = 0.035

        fig_local, axes_local = plt.subplots(1, 3, figsize=(7, 5), sharey=False)

        for ax_local, (asp_code_local, asp_title_local) in zip(
            axes_local, aspects_local
        ):
            # violin data (unpaired display)
            data_local = []
            for _, df_local in sources_local:
                vals_local = _scores_by_aspect_local(df_local, asp_code_local)[
                    "Score"
                ].to_numpy()
                data_local.append(vals_local)

            labels_local = [lab for lab, _ in sources_local]
            pos_local = np.arange(1, len(data_local) + 1)

            vp_local = ax_local.violinplot(
                data_local,
                positions=pos_local,
                widths=0.75,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            for body_local in vp_local["bodies"]:
                body_local.set_facecolor("#1192e8")
                body_local.set_edgecolor("black")
                body_local.set_alpha(0.50)
                body_local.set_linewidth(1.0)

            vp_local["cmedians"].set_color("black")
            vp_local["cmedians"].set_linewidth(1.5)

            ax_local.set_xticks(pos_local)
            ax_local.set_xticklabels(labels_local, rotation=30)

            # median labels
            for idx_local, vals_local in enumerate(data_local, start=1):
                if len(vals_local):
                    med_local = float(np.nanmedian(vals_local))
                    ax_local.text(
                        idx_local,
                        med_local,
                        f"{med_local:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            ax_local.set_title(asp_title_local, pad=18)
            if asp_code_local == "bp":
                ax_local.set_ylabel("Score (median per protein)")

            # paired Wilcoxon + bracket (dFseq vs mdF)
            a_local, b_local, n_pairs_local = _paired_scores_local(
                sources_local[0][1], sources_local[1][1], asp_code_local
            )

            if n_pairs_local == 0:
                p_local = np.nan
                stars_local = "n/a"
            else:
                diffs_local = a_local - b_local
                if np.allclose(diffs_local, 0):
                    p_local = 1.0
                else:
                    p_local = wilcoxon(
                        a_local,
                        b_local,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                stars_local = p_to_stars(p_local)

            y_ax_local = br_y0_local
            add_sig_bracket_axes(
                ax_local,
                1,
                2,
                y_ax=y_ax_local,
                h_ax=br_h_local,
                stars=stars_local,
                stars_offset_pts=3,
            )
            add_n_text_axes(
                ax_local,
                1,
                2,
                y_ax=y_ax_local,
                h_ax=br_h_local,
                n_pairs=n_pairs_local,
                n_offset_pts=-2,
            )

        fig_local.suptitle("Per protein median score distributions", y=0.98)
        fig_local.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


    # run it
    plot_median_score_violins_paired(dFseq, mdF100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cohesion analysis
    """)
    return


@app.cell
def _(dFseq, eggnog, mdF100):
    _sources = [
        ("eggnog", eggnog),
        ("dFseq", dFseq),
        ("mdF", mdF100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]
    return


@app.cell
def _(eggnog):
    print(eggnog)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Concordance analysis
    """)
    return


@app.cell
def _(importlib):
    import helpers.goterm_plots as gplot

    importlib.reload(gplot)
    return (gplot,)


@app.cell
def _(importlib):
    import helpers.goterm_propagator as prop

    importlib.reload(prop)
    return (prop,)


@app.cell
def _(mo):
    mo.md(r"""
    ## go-term propagation
    """)
    return


@app.cell
def _(dFseq, deep_go, eggnog, ic_df, mdF100, prop):
    mdf_propped = prop.propagate_go_annotations(
        mdF100,
        protein_col="Protein",
        term_col="go_term",
        score_col="Score",
        obo_path="data/external/go-basic-latest.obo",
        ic_df=ic_df,
        relations=("is_a"),
        exclude_roots=True,
        obsolete_mode="drop",  # or "map"
        obsolete_prefer="replaced_by",
        score_min=0.3,  # drop low-score annotations first
        round_decimals=3,  # round Score/IC/percent
    )

    dFseq_propped = prop.propagate_go_annotations(
        dFseq,
        protein_col="Protein",
        term_col="go_term",
        score_col="Score",
        obo_path="data/external/go-basic-latest.obo",
        ic_df=ic_df,
        relations=("is_a"),
        exclude_roots=True,
        obsolete_mode="drop",  # or "map"
        obsolete_prefer="replaced_by",
        score_min=0.2,  # drop low-score annotations first
        round_decimals=3,  # round Score/IC/percent
    )

    eggnog_propped = prop.propagate_go_annotations(
        eggnog,
        protein_col="Protein",
        term_col="go_term",
        score_col=None,
        obo_path="data/external/go-basic-latest.obo",
        ic_df=ic_df,
        relations=("is_a"),
        exclude_roots=True,
        obsolete_mode="drop",  # or "map"
        obsolete_prefer="replaced_by",
        round_decimals=3,  # round Score/IC/percent
    )

    deepgo_propped = prop.propagate_go_annotations(
        deep_go,
        protein_col="Protein",
        term_col="go_term",
        score_col="Score",
        obo_path="data/external/go-basic-latest.obo",
        ic_df=ic_df,
        relations=("is_a"),
        exclude_roots=True,
        obsolete_mode="drop",  # or "map"
        obsolete_prefer="replaced_by",
        score_min=0.3,
        round_decimals=3,  # round Score/IC/percent
    )
    return dFseq_propped, deepgo_propped, eggnog_propped, mdf_propped


@app.cell
def _(mo):
    mo.md(r"""
    ### Concordance mdf vs df
    """)
    return


@app.cell
def _(dFseq_propped, gplot, mdf_propped, plt):
    _per_prot, _per_bin = gplot.concordance_by_ic(
        mdf_propped[0],
        dFseq_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.1,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=14.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        _per_prot,
        _per_bin,
        title="mdF concordance with df_seq",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Concordance mdf vs eggnog
    """)
    return


@app.cell
def _(eggnog_propped, gplot, mdf_propped, plt):
    per_prot, per_bin = gplot.concordance_by_ic(
        mdf_propped[0],
        eggnog_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.3,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=14.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        per_prot,
        per_bin,
        title="mdF concordance with EggNOG",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Concordance mdF vs deepGO
    """)
    return


@app.cell
def _(deepgo_propped, gplot, mdf_propped, plt):
    _per_prot, _per_bin = gplot.concordance_by_ic(
        mdf_propped[0],
        deepgo_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.3,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=14.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        _per_prot,
        _per_bin,
        title="mdF concordance with mdeepGO",
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Concordance deepGO vs eggNOG
    """)
    return


@app.cell
def _(deepgo_propped, eggnog_propped, gplot, plt):
    _per_prot, _per_bin = gplot.concordance_by_ic(
        eggnog_propped[0],
        deepgo_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.3,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=14.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        _per_prot,
        _per_bin,
        title="eggnog concordance with mdeepGO",
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Why IC deepFRIseq > mdF?
    """)
    return


@app.cell
def _(dFseq, eggnog, mdF100, pd):
    IC_swissprot = pd.read_csv("data/external/IC_swissprot.csv")

    dFseq_swiss = pd.merge(
        dFseq.rename(columns={"IC": "IC_cafa"}),
        IC_swissprot,
        left_on="go_term",
        right_on="go_term",
        how="left",
    ).rename(columns={"IC": "IC_swissprot"})

    mdF100_swiss = pd.merge(
        mdF100.rename(columns={"IC": "IC_cafa"}),
        IC_swissprot,
        left_on="go_term",
        right_on="go_term",
        how="left",
    ).rename(columns={"IC": "IC_swissprot"})

    eggnog_swiss = pd.merge(
        eggnog.rename(columns={"IC": "IC_cafa"}),
        IC_swissprot,
        left_on="go_term",
        right_on="go_term",
        how="left",
    ).rename(columns={"IC": "IC_swissprot"})
    return dFseq_swiss, eggnog_swiss, mdF100_swiss


@app.cell
def _(
    add_sig_bracket,
    dFseq_swiss,
    eggnog_swiss,
    mannwhitneyu,
    mdF100_swiss,
    np,
    p_to_stars,
    plt,
):
    ic_dFseq_s20_swiss_ic = (
        dFseq_swiss.query("Score >= 0.2")
        .groupby(["Protein", "aspect"])["IC_cafa"]
        .max()
        .reset_index()
    )

    ic_mdF100_s30_swiss_ic = (
        mdF100_swiss.query("Score >= 0.3")
        .groupby(["Protein", "aspect"])["IC_cafa"]
        .max()
        .reset_index()
    )

    ic_eggnog_swiss_swiss_ic = (
        eggnog_swiss.groupby(["Protein", "aspect"])["IC_cafa"].max().reset_index()
    )

    _sources = [
        ("eggnog", ic_eggnog_swiss_swiss_ic),
        ("dFseq_s20", ic_dFseq_s20_swiss_ic),
        ("mdF_s30", ic_mdF100_s30_swiss_ic),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]


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


    _fig, _axes = plt.subplots(1, 3, figsize=(9, 5), sharey=False)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        _data = [
            _ic_by_aspect(df, _asp_code, aspect_col="aspect", score_col="IC_cafa")
            for _, df in _sources
        ]
        _labels = [label for label, _ in _sources]

        _bp = _ax.boxplot(_data, tick_labels=_labels, sym=".", widths=0.5)

        # --- annotate medians ---
        for _i, _median_line in enumerate(_bp["medians"], start=1):
            _median_val = _median_line.get_ydata().mean()
            _ax.text(
                _i,
                _median_val,
                f"{_median_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _ax.set_title(_asp_title)
        if _asp_code == "bp":
            _ax.set_ylabel("IC (max per protein)")

        # --- significance brackets (pairwise) ---
        _pairs = [
            (1, 2),
            (1, 3),
            (2, 3),
        ]  # eggnog vs dFseq, eggnog vs mdF, dFseq vs mdF

        # baseline height just above the highest point shown in this panel
        _panel_max = max(
            [np.nanmax(d.values) if len(d) else np.nan for d in _data]
        )
        _y = _panel_max
        _y_range = np.ptp(_ax.get_ylim()) if np.ptp(_ax.get_ylim()) > 0 else 1.0
        _step = 0.06 * _y_range  # vertical spacing between brackets
        _h = 0.015 * _y_range  # bracket height

        for _k, (_i, _j) in enumerate(_pairs):
            _a = _data[_i - 1].values
            _b = _data[_j - 1].values

            # Mann–Whitney U (two-sided)
            _p = mannwhitneyu(_a, _b, alternative="two-sided").pvalue

            _stars = p_to_stars(_p)
            add_sig_bracket(_ax, _i, _j, _y + _k * _step, +_h, _stars)

        # make sure brackets fit
        _ax.set_ylim(top=_y + len(_pairs) * _step + 0.08 * _y_range)

    _fig.suptitle("Per protein Max CAFA IC distributions", y=1.00)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mdF100_swiss):
    mdF100_swiss[
        ["Protein", "go_term", "Score", "Annotation", "IC_swissprot", "IC_cafa"]
    ].query("Protein == 'MGYG000001551_00428'")
    return


@app.cell
def _(dFseq_swiss):
    dFseq_swiss[
        ["Protein", "go_term", "Score", "Name", "IC_swissprot", "IC_cafa"]
    ].query("Protein == 'MGYG000001551_00428'")
    return


@app.cell
def _(mdF100_swiss):
    print(mdF100_swiss)
    return


@app.cell
def _(dFseq_swiss):
    print(dFseq_swiss)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How many go-terms are NaNs in each set?
    """)
    return


@app.cell
def _(dFseq_swiss, mdF100_swiss, pd):
    def go_term_ic_nan_counts(df):
        g = df.groupby("go_term")[["IC_cafa", "IC_swissprot"]]

        has_ic = g.apply(lambda x: x.notna().any())
        total_go_terms = has_ic.shape[0]

        return pd.Series(
            {
                "total_go_terms": total_go_terms,
                "IC_cafa_nan": (~has_ic["IC_cafa"]).sum(),
                "IC_swissprot_nan": (~has_ic["IC_swissprot"]).sum(),
            }
        )


    _coverage = pd.concat(
        [
            go_term_ic_nan_counts(mdF100_swiss).rename("mdF100"),
            go_term_ic_nan_counts(dFseq_swiss).rename("dFseq"),
        ],
        axis=1,
    )

    _coverage
    return


@app.cell
def _(mdF100_swiss, np, plt):
    def go_term_ic_values(df):
        return df.groupby("go_term")[["IC_cafa", "IC_swissprot"]].first().dropna()


    _go_ic = go_term_ic_values(mdF100_swiss)

    plt.figure(figsize=(7, 4))

    bins = np.linspace(
        min(_go_ic.min()),
        max(_go_ic.max()),
        50,  # <-- number of bins
    )

    plt.hist(
        _go_ic["IC_cafa"],
        bins=bins,
        alpha=0.5,
        density=True,
        label="IC_cafa",
    )

    plt.hist(
        _go_ic["IC_swissprot"],
        bins=bins,
        alpha=0.6,
        density=True,
        label="IC_swissprot",
    )

    plt.xlabel("Information Content (IC)")
    plt.ylabel("Density")
    plt.title("GO-term IC distribution (mdF100)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Systematic shifts in IC values
    """)
    return


@app.cell
def _(mdF100_swiss, plt):
    def go_term_ic_table(df):
        return (
            df.groupby("go_term")[["IC_cafa", "IC_swissprot"]]
            .first()
            .dropna()  # keep only GO terms with both ICs
        )


    _go_ic = go_term_ic_table(mdF100_swiss)

    go_ic_ranked = _go_ic.assign(
        IC_cafa_rank=_go_ic["IC_cafa"].rank(ascending=True, pct=True).round(3),
        IC_swissprot_rank=_go_ic["IC_swissprot"]
        .rank(ascending=True, pct=True)
        .round(3),
    )


    plt.figure(figsize=(6, 6))

    plt.scatter(
        go_ic_ranked["IC_cafa_rank"],
        go_ic_ranked["IC_swissprot_rank"],
        s=5,
        alpha=0.3,
    )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("IC_cafa rank (low → high specificity)")
    plt.ylabel("IC_swissprot rank (low → high specificity)")
    plt.title("GO-term specificity rank comparison (mdF100)")

    plt.tight_layout()
    plt.show()

    # bonus, lets find a couple go-terms that drastically change ranks
    _drastic = go_ic_ranked[
        (go_ic_ranked["IC_cafa_rank"] < 0.2)
        & (go_ic_ranked["IC_swissprot_rank"] > 0.8)
    ]

    len(_drastic)
    _drastic.head(10)
    return (go_term_ic_table,)


@app.cell
def _(mo):
    mo.md(r"""
    Systematic shifts in IC values, IC_cafa 0 removed
    """)
    return


@app.cell
def _(go_term_ic_table, mdF100_swiss, plt):
    _go_ic = go_term_ic_table(mdF100_swiss.query("IC_cafa > 0"))

    go_ic_ranked_non_zero = _go_ic.assign(
        IC_cafa_rank=_go_ic["IC_cafa"].rank(ascending=True, pct=True).round(3),
        IC_swissprot_rank=_go_ic["IC_swissprot"]
        .rank(ascending=True, pct=True)
        .round(3),
    )

    plt.figure(figsize=(6, 6))

    plt.scatter(
        go_ic_ranked_non_zero["IC_cafa_rank"],
        go_ic_ranked_non_zero["IC_swissprot_rank"],
        s=5,
        alpha=0.3,
    )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("IC_cafa rank (low → high specificity)")
    plt.ylabel("IC_swissprot rank (low → high specificity)")
    plt.title("GO-term specificity rank comparison (mdF100) for IC_cafa > 0")

    plt.tight_layout()
    plt.show()

    # bonus, lets find a couple go-terms that drastically change ranks
    _drastic = go_ic_ranked_non_zero[
        (go_ic_ranked_non_zero["IC_cafa_rank"] < 0.2)
        & (go_ic_ranked_non_zero["IC_swissprot_rank"] > 0.8)
    ]

    len(_drastic)
    _drastic.head(10)
    return


@app.cell
def _(mdF100_swiss):
    mdF100_swiss.query("IC_cafa != 0")
    return


@app.cell
def _(ic_df):
    ic_df.query("IC > 0")["IC"].hist()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
