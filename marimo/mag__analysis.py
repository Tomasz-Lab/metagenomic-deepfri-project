import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style
    import importlib
    import plotly.express as px
    import plotly.io as pio
    from pathlib import Path
    from upsetplot import from_contents, UpSet, plot
    from goatools.obo_parser import GODag
    import os
    from scipy.stats import mannwhitneyu

    # save and display plots in whitemode
    matplotlib.style.use("default")
    pio.templates.default = "plotly"
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
    import wang_similarity_helpers as wsh

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
    df = pd.read_csv("genomes-all_metadata.tsv", sep="\t")
    df_mags = df.query(
        "Genome_type=='MAG' & Completeness > 95 & Contamination < 1 & N_contigs < 100"
    )

    # choose 10 genomes from different species
    species_ids = (
        df_mags["Species_rep"]
        .drop_duplicates()
        .sample(n=10, random_state=42)  # 10 random species
    )

    selected_genomes = (
        df_mags[df_mags["Species_rep"].isin(species_ids)]
        .groupby("Species_rep")
        .apply(lambda x: x.sample(n=1))  # random genome for a selected species
        .reset_index(drop=True)
    )

    # get FTP accessions for eggNOG data and fasta
    BASE = "https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v2.0.2/species_catalogue"


    def build_urls(sid: str):
        sid_prefix = sid[:-2]
        dir_url = f"{BASE}/{sid_prefix}/{sid}/genome/"
        return {
            "Species_rep": sid,
            "faa_url": f"{dir_url}{sid}.faa",
            "eggnog_url": f"{dir_url}{sid}_eggNOG.tsv",
        }


    species_ids = selected_genomes["Species_rep"].unique()

    url_rows = [build_urls(sid) for sid in species_ids]

    urls_df = pd.DataFrame(url_rows)

    # write wget script to download those files
    with open("download_mgyg_10.sh", "w") as out:
        out.write("#!/usr/bin/env bash\n\n")
        for sid in species_ids:
            sid_prefix = sid[:-2]
            dir_url = f"{BASE}/{sid_prefix}/{sid}/genome/"
            faa_url = f"{dir_url}{sid}.faa"
            eggnog_url = f"{dir_url}{sid}_eggNOG.tsv"
            out.write(f"wget -c '{faa_url}' -O {sid}.faa\n")
            out.write(f"wget -c '{eggnog_url}' -O {sid}_eggNOG.tsv\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Helpers
    """)
    return


@app.cell
def _(GODag, pd):
    # information content from SwissProt
    ic_df = pd.read_csv(
        "/home/FilipS/2024/weave_warmup/generated_data/IC_swissprot.csv"
    )

    obo = "/home/FilipS/2025/metagenomic_deepfri/go-basic-latest.obo"

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
        "/home/FilipS/2025/metagenomic_deepfri/uhgp_mags/dFsequence/merged_predictions.csv"
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
        "/home/FilipS/2025/metagenomic_deepfri/uhgp_mags/combined.tsv", sep="\t"
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
    return dFseq, eggnog


@app.cell
def _():
    all_sequences = 26577
    return (all_sequences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mdF data loading
    """)
    return


@app.cell
def _(Path, ic_df, pd):
    BASE = Path(
        "/home/FilipS/software/mdeepfri_source/Metagenomic-DeepFRI/results/uhgp_10_mags"
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
    return (load_alignment_sets,)


@app.cell
def _(load_alignment_sets):
    pyopal_100, mdF100 = load_alignment_sets("100")
    return mdF100, pyopal_100


@app.cell
def _(pyopal_100):
    pyopal_100
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## go-term coverage
    """)
    return


@app.cell
def _(all_sequences, dFseq, eggnog, mdF100, pd, plt):
    def proteins_with_go(df):
        """
        Count unique proteins that have at least one valid GO term.
        A valid GO term is: not NaN and not '-'.
        """
        return df.loc[
            df["go_term"].notna() & (df["go_term"] != "-"), "Protein"
        ].nunique()


    _groups = {
        "eggnog": proteins_with_go(eggnog) / all_sequences * 100,
        "dF_seq": proteins_with_go(dFseq) / all_sequences * 100,
        "mdF_100": proteins_with_go(mdF100) / all_sequences * 100,
    }

    _groups_df = pd.DataFrame.from_dict(
        _groups, orient="index", columns=["proteins_with_prediction"]
    )

    plt.figure(figsize=(5, 5))
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


    def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.2):
        # bracket
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw)
        # label
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=10)
    return add_sig_bracket, p_to_stars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max IC per protein
    """)
    return


@app.cell
def _(
    add_sig_bracket,
    dFseq,
    eggnog,
    mannwhitneyu,
    mdF100,
    np,
    p_to_stars,
    plt,
):
    ic_eggnog = eggnog.groupby(["Protein", "aspect"])["IC"].max().reset_index()
    ic_dFseq = dFseq.groupby(["Protein", "aspect"])["IC"].max().reset_index()
    ic_mdF100 = mdF100.groupby(["Protein", "aspect"])["IC"].max().reset_index()


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
        ("eggnog", ic_eggnog),
        ("dFseq", ic_dFseq),
        ("mdF", ic_mdF100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(7, 5), sharey=False)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        _data = [
            _ic_by_aspect(df, _asp_code, aspect_col="aspect", score_col="IC")
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
        pairs = [
            (1, 2),
            (1, 3),
            (2, 3),
        ]  # eggnog vs dFseq, eggnog vs mdF, dFseq vs mdF

        # baseline height just above the highest point shown in this panel
        panel_max = max([np.nanmax(d.values) if len(d) else np.nan for d in _data])
        y = panel_max
        y_range = np.ptp(_ax.get_ylim()) if np.ptp(_ax.get_ylim()) > 0 else 1.0
        step = 0.06 * y_range  # vertical spacing between brackets
        h = 0.015 * y_range  # bracket height

        for k, (i, j) in enumerate(pairs):
            a = _data[i - 1].values
            b = _data[j - 1].values

            # Mann–Whitney U (two-sided)
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue

            stars = p_to_stars(p)
            add_sig_bracket(_ax, i, j, y + k * step, h, stars)

        # make sure brackets fit
        _ax.set_ylim(top=y + len(pairs) * step + 0.08 * y_range)

    _fig.suptitle("Per protein Max Information Content (IC) distributions", y=1.00)
    plt.tight_layout()
    plt.show()
    return


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
        eggnog
        .groupby(["Protein", "aspect"], as_index=False)["IC"]
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
                _df_f
                .groupby(["Protein", "aspect"], as_index=False)["IC"]
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
        return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    def _series(_m, _a, _c):
        return ic_all.loc[
            (ic_all["method"] == _m)
            & (ic_all["aspect"].astype(str).str.lower() == _a)
            & (ic_all["condition"] == _c),
            "IC",
        ].dropna()

    def _bracket(ax, x1, x2, y, h, stars, color):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, c=color, clip_on=False)
        ax.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", fontsize=9, color="black", clip_on=False)

    # --- plot ---
    _fig, _axes = plt.subplots(len(_conds), len(_aspects), figsize=(11, 9), sharey=False)

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
                _ax.text(_i, _mn - 0.075 * _rng, f"n={len(_data[_i-1])}", ha="center", va="top", fontsize=7)

            # significance brackets (more vertical room, non-overlapping)
            _yr = np.ptp(_ax.get_ylim()) or 1.0
            _y0 = _mx + 0.05 * _yr
            _step = 0.10 * _yr   # increased spacing
            _h = 0.02 * _yr

            for _k, (_i, _j) in enumerate(_pairs):
                _a, _b = _data[_i - 1], _data[_j - 1]
                _p = mannwhitneyu(_a, _b, alternative="two-sided").pvalue if (len(_a) and len(_b)) else np.nan
                _stars = _p_to_stars(_p) if np.isfinite(_p) else "na"
                _bracket(_ax, _i, _j, _y0 + _k * _step, _h, _stars, _pair_colors[(_i, _j)])

            _ax.set_ylim(top=_y0 + len(_pairs) * _step + 0.06 * _yr)

    _fig.suptitle("Per-protein max IC by method, aspect, and score filtering", y=0.995)
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
def _(add_sig_bracket, dFseq, mannwhitneyu, mdF100, np, p_to_stars, plt):
    score_dFseq = (
        dFseq.groupby(["Protein", "aspect"])["Score"].median().reset_index()
    )
    scores_mdF100 = (
        mdF100.groupby(["Protein", "aspect"])["Score"].median().reset_index()
    )


    def _ic_by_aspect(df, aspect, aspect_col=None, score_col="Score"):
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
        ("dFseq", score_dFseq),
        ("mdF", scores_mdF100),
    ]

    _aspects = [
        ("bp", "BP"),
        ("mf", "MF"),
        ("cc", "CC"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(7, 5), sharey=False)

    for _ax, (_asp_code, _asp_title) in zip(_axes, _aspects):
        _data = [
            _ic_by_aspect(df, _asp_code, aspect_col="aspect", score_col="Score")
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
            _ax.set_ylabel("score (median per protein)")

        # --- significance brackets (pairwise) ---
        _pairs = [(1, 2)]

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
            add_sig_bracket(_ax, _i, _j, _y + _k * _step, _h, _stars)

        # make sure brackets fit
        _ax.set_ylim(top=_y + len(_pairs) * _step + 0.08 * _y_range)

    _fig.suptitle("Per protein median score distributions", y=1.00)
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
def _():
    return


if __name__ == "__main__":
    app.run()
