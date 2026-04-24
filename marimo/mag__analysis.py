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

    PLOT_DIR = "plots"
    RAW_DIR = "plots/raw_data"
    os.makedirs(RAW_DIR, exist_ok=True)
    return (
        GODag,
        PLOT_DIR,
        Path,
        RAW_DIR,
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
    selected_genomes
    return


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
    _df
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
    dFseq = pd.read_csv("data/source/uhgp_mags/uhgp_50_mags/deepfri_seq_preds.csv")
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


    # EGGnog go-terms from MGY
    eggnog = pd.read_csv(
        "data/source/uhgp_mags/uhgp_50_mags/eggnog_merged.tsv",
        sep="\t",
    ).rename(columns={"#query": "Protein", "GOs": "go_term"})

    eggnog_go = eggnog[["Protein", "go_term"]]

    eggnog_go = (
        eggnog_go.assign(go_term=eggnog_go["go_term"].str.split(","))
        .explode("go_term")
        .reset_index(drop=True)
    )

    eggnog_go = pd.merge(
        eggnog_go,
        ic_df,
        left_on="go_term",
        right_on="go_term",
        how="left",
    )

    ### add missing aspect for eggnog ###
    _valid_terms_egg = (
        eggnog_go["go_term"].dropna().loc[lambda s: s.ne("-")].astype(str).unique()
    )

    go2aspect_egg = {t: resolve_aspect(t) for t in _valid_terms_egg}
    eggnog_go["aspect"] = eggnog_go["go_term"].map(go2aspect_egg)

    # load deepgo predictions

    deep_go = pd.read_csv(
        "data/source/uhgp_mags/uhgp_50_mags/deep_go_meta/uhgp_50_mags_preds_merged.tsv",
        sep="\t",
    )

    deep_go = pd.merge(
        deep_go,
        ic_df,
        left_on="go_term",
        right_on="go_term",
        how="left",
    )
    return dFseq, deep_go, eggnog, eggnog_go


@app.cell
def _(deep_go):
    deep_go["Protein"].nunique()
    return


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
    _original_fasta = "data/source/uhgp_mags/uhgp_50_mags/uhgp_50_mags.faa"
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
    _BASE = Path("data/source/uhgp_mags/uhgp_50_mags/mdeepfri_results")

    identity_bins = {
        "100": "identity_bin_0.00-1.01",
    }

    db_files = {
        "pdb100": "pdb100_230517_alignment_scores.tsv",
        "afdb_v4": "afdb_uniprot_v4_alignment_scores.tsv",
        "esm_highquality_clust30": "highquality_clust30_alignment_scores.tsv",
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
def _(pyopal_100):
    pyopal_100["pdb100"].query("query_name=='MGYG000000513_01635'")
    return


@app.cell
def _(mdF100):
    mdF100.query("Identity < 0.3")
    return


@app.cell
def _(all_sequences, pyopal_100):
    def compute_coverage(align_dict, total_queries):
        """align_dict = {'pdb':df, 'uniprot':df, 'esm':df}"""

        # unique query hits for each DB
        hits = {db: set(df["query_name"]) for db, df in align_dict.items()}

        cov_pdb = len(hits["pdb100"])
        cov_pdb_uniprot = len(hits["pdb100"] | hits["afdb_v4"])
        cov_all = len(
            hits["pdb100"] | hits["afdb_v4"] | hits["esm_highquality_clust30"]
        )

        any_hit = cov_all

        return {
            "pdb100_only": cov_pdb,
            "pdb_plus_afdb": cov_pdb_uniprot,
            "pdb_plus_afdb_plus_esm": cov_all,
            "any_hit": any_hit,
            "total_queries": total_queries,
        }


    cov_100 = compute_coverage(pyopal_100, total_queries=all_sequences)
    cov_100
    return (cov_100,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Structure database composition
    """)
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, all_sequences, cov_100, pd, plt):
    from matplotlib.ticker import FuncFormatter


    def decompose_cov(cov):
        total = cov["total_queries"]
        pdb = cov["pdb100_only"]
        uni = cov["pdb_plus_afdb"] - pdb
        esm = cov["pdb_plus_afdb_plus_esm"] - (pdb + uni)
        no_hit = total - (pdb + uni + esm)
        return pdb, uni, esm, no_hit


    title = ""

    _fig, ax = plt.subplots(figsize=(5, 5))

    pdb, uni, esm, no_hit = decompose_cov(cov_100)

    _labels = [""]

    ax.bar(_labels, [pdb], label="PDB100 hits", color="#1192e8")
    ax.bar(_labels, [uni], bottom=[pdb], label="+AFDBv4 hits", color="#ff832b")
    ax.bar(
        _labels,
        [esm],
        bottom=[pdb + uni],
        label="+ESM_clust hits",
        color="#198038",
    )
    ax.bar(
        _labels,
        [no_hit],
        bottom=[pdb + uni + esm],
        label="No hit",
        color="#da1e28",
    )

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
    _fig.suptitle("Database hits distribution", fontsize=14, y=1.08)
    handles, _labels = ax.get_legend_handles_labels()
    _fig.legend(
        handles, _labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0)
    )


    plt.tight_layout()

    _fig.savefig(f"{PLOT_DIR}/structure_db_composition.svg", bbox_inches="tight")
    pd.DataFrame([cov_100]).to_csv(
        f"{RAW_DIR}/structure_db_composition.csv", index=False
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identity coverage curve
    """)
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, np, pd, plt, pyopal_100):
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

    _colors = {
        "pdb100": "#1192e8",
        "afdb_v4": "#ff832b",
        "esm_highquality_clust30": "#198038",
    }

    plt.figure(figsize=(7, 5))

    for name, curve in curves.items():
        plt.plot(
            curve["identity_threshold"],
            curve["coverage_frac"],
            label=name,
            color=_colors.get(name, "gray"),
        )

    plt.gca().invert_xaxis()
    plt.xlabel("Identity threshold")
    plt.ylabel("Fraction of queries with a hit")
    plt.title("Relaxing identity cutoff increases coverage")
    plt.legend()
    plt.tight_layout()

    _fig = plt.gcf()
    _fig.savefig(f"{PLOT_DIR}/identity_coverage_curve.svg", bbox_inches="tight")
    pd.concat(
        [df.assign(db_name=name) for name, df in curves.items()], ignore_index=True
    ).to_csv(f"{RAW_DIR}/identity_coverage_curve.csv", index=False)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction coverage
    """)
    return


@app.cell
def _(eggnog):
    eggnog.head()
    return


@app.cell
def _(pd):
    # EggNOG filtration
    import re

    EVIDENCE_COLS = [
        "COG_category",
        "Description",
        "Preferred_name",
        "GOs",
        "EC",
        "KEGG_ko",
        "KEGG_Pathway",
        "KEGG_Module",
        "KEGG_Reaction",
        "KEGG_rclass",
        "BRITE",
        "KEGG_TC",
        "CAZy",
        "BiGG_Reaction",
        "PFAMs",
    ]

    PFAM_BAD_PAT = re.compile(
        r"^(?:DUF\d*|unknown|uncharacteri[sz]ed|hypothetical)$", re.IGNORECASE
    )

    # Description strings that should NOT count as meaningful evidence (tune as needed)
    DESC_BAD_PAT = re.compile(
        r"(?:^|\b)(protein of unknown function|hypothetical protein|uncharacteri[sz]ed protein|"
        r"domain of unknown function|unknown function)(?:\b|$)",
        re.IGNORECASE,
    )


    def filter_meaningful_eggnog_rows(df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()
        _cols = [c for c in EVIDENCE_COLS if c in _df.columns]
        if not _cols:
            return _df.iloc[0:0].copy()

        def _present(s: pd.Series) -> pd.Series:
            s = s.astype("string")
            return ~(s.isna() | (s.str.strip().isin(["", "-"])))

        def _not_bad_desc(s: pd.Series) -> pd.Series:
            s = s.astype("string").fillna("").str.strip()
            # count as meaningful only if present and NOT matching bad patterns
            return ~s.isin(["", "-"]) & ~s.str.contains(DESC_BAD_PAT)

        # Base: any evidence present anywhere
        _any_evidence = pd.concat([_present(_df[c]) for c in _cols], axis=1).any(
            axis=1
        )

        # COG status
        if "COG_category" in _df.columns:
            _cog = _df["COG_category"].astype("string").fillna("").str.strip()
            _cog_present = ~_cog.isin(["", "-"])
            _cog_is_S = _cog.eq("S")
            _cog_good = _cog_present & ~_cog_is_S
        else:
            _cog_present = pd.Series(False, index=_df.index)
            _cog_is_S = pd.Series(False, index=_df.index)
            _cog_good = pd.Series(False, index=_df.index)

        # PFAM status: bad only if ALL tokens are bad
        if "PFAMs" in _df.columns:
            _pf = _df["PFAMs"].astype("string").fillna("").str.strip()
            _pf_present = ~_pf.isin(["", "-"])
            _tokens = _pf.str.replace(";", ",", regex=False).str.split(",")
            _pfam_is_bad = _tokens.apply(
                lambda toks: (
                    len([t for t in toks if t.strip() and t.strip() != "-"]) > 0
                    and all(
                        PFAM_BAD_PAT.match(t.strip())
                        for t in toks
                        if t.strip() and t.strip() != "-"
                    )
                )
            )
            _pfam_good = _pf_present & ~_pfam_is_bad
        else:
            _pf_present = pd.Series(False, index=_df.index)
            _pfam_is_bad = pd.Series(False, index=_df.index)
            _pfam_good = pd.Series(False, index=_df.index)

        # Description: meaningful only if not boilerplate unknown-like
        if "Description" in _df.columns:
            _desc_present = _present(_df["Description"])
            _desc_good = _not_bad_desc(_df["Description"])
            _desc_bad = _desc_present & ~_desc_good
        else:
            _desc_present = pd.Series(False, index=_df.index)
            _desc_good = pd.Series(False, index=_df.index)
            _desc_bad = pd.Series(False, index=_df.index)

        # "Other evidence" = all evidence columns except (COG_category, PFAMs, Description),
        # plus "good Description" counts as other evidence.
        _other_cols = [
            c for c in _cols if c not in {"COG_category", "PFAMs", "Description"}
        ]
        _has_other = (
            pd.concat([_present(_df[c]) for c in _other_cols], axis=1).any(axis=1)
            if _other_cols
            else pd.Series(False, index=_df.index)
        )
        _has_other_or_good_desc = _has_other | _desc_good

        # Drop if nothing meaningful exists beyond:
        # - COG=S
        # - bad PFAM
        # - bad Description
        # and also no good COG / good PFAM / good Description / other columns
        _only_bad_sources = (
            ~_has_other_or_good_desc
            & (
                (_cog_present & _cog_is_S)
                | (_pf_present & _pfam_is_bad)
                | _desc_bad
            )
            & ~_cog_good
            & ~_pfam_good
            & ~_desc_good
        )

        _keep = _any_evidence & ~_only_bad_sources
        return _df.loc[_keep].copy()


    def count_meaningful_eggnog_rows(df: pd.DataFrame) -> int:
        return int(filter_meaningful_eggnog_rows(df).shape[0])
    return (count_meaningful_eggnog_rows,)


@app.cell
def _(
    PLOT_DIR,
    RAW_DIR,
    all_sequences,
    count_meaningful_eggnog_rows,
    dFseq,
    deep_go,
    eggnog,
    eggnog_go,
    mdF100,
    pd,
    plt,
):
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
        "EggNOG-go": proteins_with_go(eggnog_go) / all_sequences * 100,
        "EggNOG-any": count_meaningful_eggnog_rows(eggnog) / all_sequences * 100,
        "deepFRI-seq": proteins_with_go(_dFseq_hq) / all_sequences * 100,
        "mdeepFRI": proteins_with_go(_mdf100_hq) / all_sequences * 100,
        "DeepGOMeta": proteins_with_go(_deep_go_hq) / all_sequences * 100,
    }

    _groups_df = pd.DataFrame.from_dict(
        _groups, orient="index", columns=["proteins_with_prediction"]
    )

    # --- customize colors here ---
    _colors = [
        "#ff9245",  # EggNOG-go
        "#ff8389",  # EggNOG-any
        "#33b1ff",  # deepFRI-sequence
        "#3ddbd9",  # Metagenonic-deepFRI
        "#6fdc8c",  # deep_go
    ]

    _fig, _ax = plt.subplots(figsize=(4, 5))

    bars = _ax.bar(
        _groups_df.index.astype(str),
        _groups_df["proteins_with_prediction"],
        color=_colors,
        edgecolor="black",
    )

    # Add text labels
    for bar in bars:
        height = bar.get_height()
        _ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    _ax.set_xlabel("Prediction source", fontsize=14)
    _ax.tick_params(axis="x", labelsize=14, rotation=90)
    _ax.tick_params(axis="y", labelsize=14, rotation=0)

    _ax.set_ylabel("% proteins with ≥ 1 prediction", fontsize=14)

    _fig.tight_layout()

    # Save
    _fig.savefig(f"{PLOT_DIR}/prediction_coverage.svg", bbox_inches="tight")
    _fig.savefig(
        f"{PLOT_DIR}/prediction_coverage.pdf", bbox_inches="tight", dpi=300
    )
    _groups_df.to_csv(f"{RAW_DIR}/prediction_coverage.csv")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GO-term overlap (UpSet)
    """)
    return


@app.cell
def _(PLOT_DIR, dFseq, eggnog_go, from_contents, mdF100, plot, plt):
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
        fig.savefig(
            f"{PLOT_DIR}/goterm_overlap_upset_{aspect_label}.svg",
            bbox_inches="tight",
        )
        plt.show()


    for _aspect in ["bp", "mf", "cc"]:
        upset_by_aspect(
            aspect_label=_aspect,
            eggnog=eggnog_go,
            dFseq=dFseq,
            mdF100=mdF100,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Annotations per protein
    """)
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, dFseq, eggnog_go, mdF100, pd, plt):
    def _get_aspect_counts(counts_series, aspect):
        aspect_lower = str(aspect).lower()
        aspect_index = (
            counts_series.index.get_level_values("aspect").astype(str).str.lower()
        )
        return counts_series[aspect_index == aspect_lower].values


    # 1) Counts per (Protein, Aspect)
    counts_eggnog = eggnog_go.groupby(["Protein", "aspect"])["go_term"].nunique()
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

    _fig.savefig(f"{PLOT_DIR}/annotations_per_protein.svg", bbox_inches="tight")
    pd.concat(
        [
            counts_eggnog.reset_index().assign(method="eggnog"),
            counts_dFseq.reset_index().assign(method="dFseq"),
            counts_mdF100.reset_index().assign(method="mdF"),
        ],
        ignore_index=True,
    ).to_csv(f"{RAW_DIR}/annotations_per_protein.csv", index=False)

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
    return add_sig_bracket_axes, p_to_stars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max IC per protein
    """)
    return


@app.cell
def _(
    PLOT_DIR,
    RAW_DIR,
    add_sig_bracket_axes,
    dFseq,
    deep_go,
    eggnog_go,
    mannwhitneyu,
    mdF100,
    np,
    p_to_stars,
    pd,
    plt,
):
    # --- max-IC-per-protein tables ---
    ic_eggnog = eggnog_go.groupby(["Protein", "aspect"])["IC"].max().reset_index()
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
        ("EggNOG-go", ic_eggnog),
        ("deepFRI-seq", ic_dFseq_s20),
        ("mdeepFRI", ic_mdF100_s30),
        ("DeepGOMeta", ic_deepgo_s30),
    ]

    METHOD_COLORS = {
        "EggNOG-go": "#ff9245",
        "deepFRI-seq": "#33b1ff",
        "mdeepFRI": "#3ddbd9",
        "DeepGOMeta": "#6fdc8c",
    }

    ASPECTS = [("bp", "BP"), ("mf", "MF"), ("cc", "CC")]
    PAIRS = [(1, 3), (2, 3), (3, 4)]  # 1-indexed positions

    labels = [name for name, _ in SOURCES]
    colors = [METHOD_COLORS[name] for name in labels]


    ##### SAVE RAW DATA FOR JOURNAL #####
    _records = []

    for _method_name, _df in SOURCES:
        _tmp = _df.copy()
        _tmp = _tmp[["Protein", "aspect", "IC"]].copy()
        _tmp["method"] = _method_name
        _records.append(_tmp)

    source_data = pd.concat(_records, ignore_index=True)

    # Standardize aspect case
    source_data["aspect"] = source_data["aspect"].str.lower()

    # Save as CSV
    source_data.to_csv(f"{RAW_DIR}/max_ic_distribution.csv", index=False)
    #####  #####  #####  #####  #####  #####


    def ic_series(df, aspect_code):
        _mask = df["aspect"].astype(str).str.lower().eq(str(aspect_code).lower())
        return df.loc[_mask, "IC"].dropna()


    def mwu_with_rbc(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            return np.nan, np.nan
        _res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        _U = _res.statistic
        _n1, _n2 = len(x), len(y)
        _rbc = 1.0 - (2.0 * _U) / (_n1 * _n2)
        return _res.pvalue, _rbc


    def draw_violin(_ax, _data, _labels, _colors):
        _positions = np.arange(1, len(_data) + 1)

        _vp = _ax.violinplot(
            [np.asarray(d) for d in _data],
            positions=_positions,
            widths=0.75,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        # Color each violin separately
        for _body, _color in zip(_vp["bodies"], _colors):
            _body.set_facecolor(_color)
            _body.set_edgecolor("black")
            _body.set_alpha(1.0)
            _body.set_linewidth(1.0)

        _vp["cmedians"].set_color("black")
        _vp["cmedians"].set_linewidth(1.5)

        _ax.set_xticks(_positions)
        _ax.set_xticklabels(_labels, rotation=30, ha="center", fontsize=12)
        _ax.tick_params(axis="y", labelsize=14)

        # median annotations
        for _i, _d in enumerate(_data, start=1):
            _d = np.asarray(_d)
            if _d.size:
                _med = float(np.nanmedian(_d))
                _ax.text(
                    _i, _med, f"{_med:.1f}", ha="center", va="bottom", fontsize=10
                )

        return _positions


    # ---- Figure: single row (global / unpaired MWU) ----
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    labels = [name for name, _ in SOURCES]

    for col, (asp_code, _asp_title_unused) in enumerate(ASPECTS):
        _ax = axes[col]

        _global_series = [ic_series(df, asp_code) for _, df in SOURCES]
        _data = [s.to_numpy() for s in _global_series]
        _positions = draw_violin(_ax, _data, labels, colors)

        _ax.text(
            0.02,
            0.98,
            _asp_title_unused,  # e.g. "BP", "MF", "CC"
            transform=_ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        if col == 0:
            _ax.set_ylabel("IC (max per protein)", fontsize=14)

        # Grey counts in parentheses under each bin
        _n_labels = [len(s) for s in _global_series]
        for _pos, _n in zip(_positions, _n_labels):
            _ax.text(
                _pos,
                -0.1,
                f"({_n})",
                ha="center",
                va="top",
                fontsize=10,
                color="0.5",
                rotation=30,
                transform=_ax.get_xaxis_transform(),
            )

        # Statistical tests + brackets (MWU)
        for k, (i, j) in enumerate(PAIRS):
            _p, _ = mwu_with_rbc(
                _global_series[i - 1].to_numpy(), _global_series[j - 1].to_numpy()
            )
            _stars = p_to_stars(_p) if np.isfinite(_p) else "n/a"
            _y_ax = 0.75 + k * 0.08
            add_sig_bracket_axes(
                _ax, i, j, y_ax=_y_ax, h_ax=0.03, stars=_stars, stars_offset_pts=0
            )

    _ymin, _ymax = _ax.get_ylim()
    _ax.set_ylim(_ymin, _ymax * 1.10)  # headroom

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=1, wspace=0.05)

    plt.savefig(f"{PLOT_DIR}/max_ic_distribution.svg", format="svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max IC by score threshold
    """)
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, dFseq, eggnog_go, mannwhitneyu, mdF100, np, pd, plt):
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
        eggnog_go.groupby(["Protein", "aspect"], as_index=False)["IC"]
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

    _fig.savefig(f"{PLOT_DIR}/max_ic_by_score_threshold.svg", bbox_inches="tight")
    ic_all.to_csv(f"{RAW_DIR}/max_ic_by_score_threshold.csv", index=False)

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
    PLOT_DIR,
    RAW_DIR,
    add_sig_bracket_axes,
    dFseq,
    mannwhitneyu,
    mdF100,
    np,
    p_to_stars,
    pd,
    plt,
):
    def plot_median_score_violins_mwu(dFseq_df, mdF100_df):
        # per-protein median score tables
        _score_dfseq = (
            dFseq_df.groupby(["Protein", "aspect"])["Score"].median().reset_index()
        )
        _score_mdf = (
            mdF100_df.groupby(["Protein", "aspect"])["Score"]
            .median()
            .reset_index()
        )

        _sources = [
            ("deepFRI-seq", _score_dfseq),
            ("mdeepFRI", _score_mdf),
        ]
        _aspects = [("bp", "BP"), ("mf", "MF"), ("cc", "CC")]

        _method_colors = {
            "deepFRI-seq": "#33b1ff",
            "mdeepFRI": "#3ddbd9",
        }

        ##### SAVE RAW DATA FOR JOURNAL #####
        _records = []

        for _method_name, _df in _sources:
            _tmp = _df.copy()
            _tmp = _tmp[["Protein", "aspect", "Score"]].copy()
            _tmp["method"] = _method_name
            _tmp = _tmp.rename(columns={"Score": "median_score"})
            _records.append(_tmp)

        source_data_scores = pd.concat(_records, ignore_index=True)

        # standardize aspect formatting
        source_data_scores["aspect"] = source_data_scores["aspect"].str.lower()

        # Save file
        source_data_scores.to_csv(
            f"{RAW_DIR}/median_score_distribution.csv",
            index=False,
        )

        #### #### #### #### #### #### #### #### ####

        def _series(df, _asp):
            _mask = df["aspect"].astype(str).str.lower().eq(str(_asp).lower())
            return df.loc[_mask, "Score"].dropna()

        def _mwu(x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if len(x) == 0 or len(y) == 0:
                return np.nan
            return mannwhitneyu(
                x, y, alternative="two-sided", method="auto"
            ).pvalue

        def _draw_violin(_ax, _data, _labels, _colors):
            _positions = np.arange(1, len(_data) + 1)
            _vp = _ax.violinplot(
                [np.asarray(d) for d in _data],
                positions=_positions,
                widths=0.75,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for _body, _c in zip(_vp["bodies"], _colors):
                _body.set_facecolor(_c)
                _body.set_edgecolor("black")
                _body.set_alpha(0.9)
                _body.set_linewidth(1.0)

            _vp["cmedians"].set_color("black")
            _vp["cmedians"].set_linewidth(1.5)

            _ax.set_xticks(_positions)
            _ax.set_xticklabels(_labels, rotation=30, ha="center", fontsize=13)

            # median labels
            for _i, _vals in enumerate(_data, start=1):
                _vals = np.asarray(_vals)
                if _vals.size:
                    _med = float(np.nanmedian(_vals))
                    _ax.text(
                        _i,
                        _med,
                        f"{_med:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            return _positions

        fig, axes = plt.subplots(1, 3, figsize=(5, 5), sharey=True)

        _labels = [n for n, _ in _sources]
        _colors = [_method_colors[n] for n in _labels]

        for col, (_asp_code, _asp_label) in enumerate(_aspects):
            _ax = axes[col]

            _series_list = [_series(df, _asp_code) for _, df in _sources]
            _data = [s.to_numpy() for s in _series_list]
            _positions = _draw_violin(_ax, _data, _labels, _colors)

            # In-panel ontology label (no subplot title)
            _ax.text(
                0.02,
                0.98,
                _asp_label,
                transform=_ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            if col == 0:
                _ax.set_ylabel("Score (median per protein)", fontsize=14)

            # Grey counts in parentheses under each bin (no "n=")
            _n_labels = [len(s) for s in _series_list]
            for _pos, _n in zip(_positions, _n_labels):
                _ax.text(
                    _pos,
                    -0.12,
                    f"({_n})",
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="0.5",
                    rotation=30,
                    transform=_ax.get_xaxis_transform(),
                )

            # Mann–Whitney U + bracket (dFseq vs mdF)
            _p = _mwu(_data[0], _data[1])
            _stars = p_to_stars(_p) if np.isfinite(_p) else "n/a"

            # add top headroom for bracket/asterisks (works even with varying ranges)
            _ymin, _ymax = _ax.get_ylim()
            _ax.set_ylim(_ymin, _ymax * 1.04)

            add_sig_bracket_axes(
                _ax,
                1,
                2,
                y_ax=0.85,
                h_ax=0.035,
                stars=_stars,
                stars_offset_pts=3,
            )

        # layout: extra bottom space for counts
        fig.subplots_adjust(
            left=0.08, right=0.99, bottom=0.25, top=0.98, wspace=0.10
        )
        plt.savefig(
            f"{PLOT_DIR}/median_score_distribution.svg", format="svg", dpi=300
        )
        plt.show()


    # run it
    plot_median_score_violins_mwu(dFseq, mdF100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cohesion analysis
    """)
    return


@app.cell
def _(dFseq, eggnog_go, mdF100):
    _sources = [
        ("eggnog", eggnog_go),
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
def _(eggnog_go):
    print(eggnog_go)
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
def _(dFseq, deep_go, eggnog_go, ic_df, mdF100, prop):
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
        filter_predictable_terms=True,  # Enable filtering to only include dF predictable terms
        predictable_terms_path="data/external/deepfri_predictable_terms.csv",
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
        eggnog_go,
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
        filter_predictable_terms=True,  # Enable filtering
        predictable_terms_path="data/external/deepfri_predictable_terms.csv",
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
        filter_predictable_terms=True,  # Enable filtering
        predictable_terms_path="data/external/deepfri_predictable_terms.csv",
    )
    return dFseq_propped, deepgo_propped, eggnog_propped, mdf_propped


@app.cell
def _(mo):
    mo.md(r"""
    ### Concordance mdF vs deepFRI-seq
    """)
    return


@app.cell
def _(PLOT_DIR, dFseq_propped, gplot, mdf_propped, plt):
    _per_prot, _per_bin, _stats = gplot.concordance_by_ic(
        mdf_propped[0],
        dFseq_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",  # obsolete terms are removed
        score_min=0.2,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=14.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        _per_prot,
        _per_bin,
        title="mdF concordance with deepFRI-seq",
    )

    gplot.save_concordance_artifacts(
        _fig,
        _per_prot,
        out_prefix="concordance_mdf_vs_dfseq",
        out_dir=PLOT_DIR,
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
def _(PLOT_DIR, eggnog_propped, gplot, mdf_propped, plt):
    per_prot, per_bin, stats = gplot.concordance_by_ic(
        mdf_propped[0],
        eggnog_propped[0],
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
        out_prefix="concordance_mdf_vs_eggnog",
        out_dir=PLOT_DIR,
    )

    plt.show()
    return per_bin, per_prot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compact concordance: faceted violins (mdf vs eggnog)
    """)
    return


@app.cell
def _(PLOT_DIR, gplot, per_bin, per_prot, plt):
    _fig_faceted = gplot.plot_concordance_faceted(
        per_prot,
        per_bin,
        figsize=(5, 3),
        aspects=["mf"],
        x_bin_labels_as_interval=True,
        x_bin_interval_labels_zero_origin=True,
    )

    gplot.save_concordance_artifacts(
        _fig_faceted,
        per_prot,
        out_prefix="concordance_mdf_vs_eggnog_faceted_MF",
        out_dir=PLOT_DIR,
    )

    plt.show()
    return


@app.cell
def _(PLOT_DIR, gplot, per_bin, per_prot, plt):
    _fig_faceted = gplot.plot_concordance_faceted(
        per_prot,
        per_bin,
        figsize=(5, 3),
        aspects=["bp"],
        x_bin_labels_as_interval=True,
        x_bin_interval_labels_zero_origin=True,
    )

    gplot.save_concordance_artifacts(
        _fig_faceted,
        per_prot,
        out_prefix="concordance_mdf_vs_eggnog_faceted_BP",
        out_dir=PLOT_DIR,
    )

    plt.show()
    return


@app.cell
def _(PLOT_DIR, gplot, per_bin, per_prot, plt):
    _fig_faceted = gplot.plot_concordance_faceted(
        per_prot,
        per_bin,
        figsize=(5, 3),
        aspects=["cc"],
        x_bin_labels_as_interval=True,
        x_bin_interval_labels_zero_origin=True,
    )

    gplot.save_concordance_artifacts(
        _fig_faceted,
        per_prot,
        out_prefix="concordance_mdf_vs_eggnog_faceted_CC",
        out_dir=PLOT_DIR,
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concordance heatmap (mdF vs eggNOG)
    """)
    return


@app.cell
def _(PLOT_DIR, gplot, per_bin, per_prot, plt):
    _fig_heatmap = gplot.plot_concordance_heatmap(
        per_prot,
        per_bin,
    )
    _fig_heatmap.savefig(
        f"{PLOT_DIR}/concordance_mdf_vs_eggnog_heatmap.svg", bbox_inches="tight"
    )
    plt.show()
    return


@app.cell
def _(per_prot):
    per_prot.query("Protein=='MGYG000000354_00030'")
    return


@app.cell
def _(mdf_propped):
    mdf_propped[0].query("Protein=='MGYG000000354_00030' & IC <= 6 & IC >5")
    return


@app.cell
def _(eggnog_propped):
    eggnog_propped[0].query("Protein=='MGYG000000354_00030' & IC <= 6 & IC >5")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Propagated terms fraction by IC
    """)
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, mdf_propped, np, plt):
    # Number of bins (change if needed)
    _n_bins = 20

    _df = mdf_propped[0][np.isfinite(mdf_propped[0]["IC"])].copy()

    _df["_IC_int_bin"] = np.floor(_df["IC"]).astype(int)

    # Aggregate statistics
    _bin_stats = _df.groupby("_IC_int_bin", observed=True).agg(
        total_terms=("Propagated", "count"),
        propagated_terms=("Propagated", lambda x: (x == True).sum()),
    )

    _bin_stats["_fraction_propagated"] = (
        _bin_stats["propagated_terms"] / _bin_stats["total_terms"]
    )

    _bin_stats = _bin_stats.reset_index()

    _bin_stats

    _figsize = (8, 4)

    _fig, _ax = plt.subplots(figsize=_figsize)

    _ax.bar(
        _bin_stats["_IC_int_bin"],
        _bin_stats["_fraction_propagated"],
    )

    _xticks = _bin_stats["_IC_int_bin"]

    _labels = [f"{int(ic)}" for ic in _xticks]

    _ax.set_xticks(_xticks)
    _ax.set_xticklabels(_labels)

    # Second annotation row
    for ic, n in zip(_bin_stats["_IC_int_bin"], _bin_stats["total_terms"]):
        _ax.text(
            ic,
            -0.07,  # vertical position below axis
            f"(n={int(n)})",
            ha="center",
            va="top",
            transform=_ax.get_xaxis_transform(),
            rotation=30,
            fontsize=9,
        )

    _ax.set_xlabel("IC (integer bin)", labelpad=33)
    _ax.set_ylabel("Fraction Propagated")
    _ax.set_title("Fraction of Propagated Terms per IC Level")

    plt.tight_layout()

    _fig.savefig(f"{PLOT_DIR}/propagated_fraction_by_ic.svg", bbox_inches="tight")
    _bin_stats.to_csv(f"{RAW_DIR}/propagated_fraction_by_ic.csv", index=False)

    plt.show()
    return


@app.cell
def _(mdf_propped, np):
    # calculate fraction of max IC terms that are propagated

    _df = mdf_propped[0][np.isfinite(mdf_propped[0]["IC"])].copy()

    _max = _df.loc[_df.groupby("Protein")["IC"].idxmax()]

    _max["Propagated"].mean()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Concordance mdF vs deepGO
    """)
    return


@app.cell
def _(PLOT_DIR, deepgo_propped, gplot, mdf_propped, plt):
    _per_prot, _per_bin, _stats = gplot.concordance_by_ic(
        mdf_propped[0],
        deepgo_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.3,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=9.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        _per_prot,
        _per_bin,
        title="mdF concordance with DeepGOMeta",
    )

    gplot.save_concordance_artifacts(
        _fig,
        _per_prot,
        out_prefix="concordance_mdf_vs_deepgo",
        out_dir=PLOT_DIR,
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
def _(PLOT_DIR, deepgo_propped, eggnog_propped, gplot, plt):
    _per_prot, _per_bin, _stats = gplot.concordance_by_ic(
        eggnog_propped[0],
        deepgo_propped[0],
        protein_col="Protein",
        term_col="GO_term",
        ic_col="IC",
        metric="jaccard",  # 0..1 overlap
        score_col="Score",
        score_min=0.3,  # or e.g. 0.2
        ic_min=1.0,
        ic_max=9.0,  # bins 1..13
        bin_width=1.0,
        drop_propagated=None,  # True for originals, False for propagated only and None for both
        require_both=True,  # force calculation only if both methods have > 0 go-terms for that IC
    )

    _fig = gplot.plot_concordance_boxplot(
        _per_prot,
        _per_bin,
        title="eggNOG concordance with DeepGOMeta",
    )

    gplot.save_concordance_artifacts(
        _fig,
        _per_prot,
        out_prefix="concordance_deepgo_vs_eggnog",
        out_dir=PLOT_DIR,
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
