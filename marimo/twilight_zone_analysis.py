import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Contact maps from twilight-zone templates

    Contact map median F1 is 0.90 for templates at 20-30%
    PyOpal identity. This notebook checks what that number means: how stable is the
    identity definition on the x-axis, which contacts the F1 counts, how templates
    cover the query, and which queries fail.

    Inputs are written by scripts in `scripts/`:

    - `identity_definitions.py`: MMseqs2 identity (as reported and exact) and PyOpal identity and coverage variants
    - `contact_counts_by_separation.py`: true, predicted and correct contacts per sequence-separation band
    - `structure_features.py`: pLDDT, secondary structure and contact order of queries and templates
    - `ted_assignment.py`: TED domains and CATH labels of queries and templates
    """)
    return


@app.cell
def _():
    import importlib
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.style
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    # save and display plots in whitemode
    matplotlib.style.use("default")

    PLOT_DIR = Path("plots")
    RAW_DIR = PLOT_DIR / "raw_data"
    return (
        LinearSegmentedColormap,
        LogNorm,
        PLOT_DIR,
        RAW_DIR,
        importlib,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(importlib):
    import helpers.twilight_helpers as th

    importlib.reload(th)
    return (th,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data preparation
    """)
    return


@app.cell
def _(np, pd, th):
    ### pair-level table: identities, TM-score and contact scores at gc = 2 ###
    _keys = ["query", "target", "idbin"]
    _identity = pd.read_csv(
        "data/generated/gc_benchmark_identity_definitions.csv"
    )
    _tm = pd.read_csv(
        "data/generated/gc_benchmark_with_tm.csv",
        usecols=_keys + ["tm_score", "evalue"],
    )
    _counts = pd.read_csv("data/generated/gc_benchmark_contact_counts.csv")
    df = (
        _identity.merge(_tm, on=_keys)
        .merge(_counts.query("gc == 2"), on=_keys)
        .reset_index(drop=True)
    )

    # column -> (label, contact definition); min_sep counts the upper triangle only
    contact_definitions = {
        "f1_full": ("full map", {}),
        "f1_sep1": ("|i-j| ≥ 1", {"min_sep": 1}),
        "f1_sep3": ("|i-j| ≥ 3", {"min_sep": 3}),
        "f1_sep6": ("|i-j| ≥ 6", {"min_sep": 6}),
        "f1_sep12": ("|i-j| ≥ 12", {"min_sep": 12}),
        "f1_sep24": ("|i-j| ≥ 24", {"min_sep": 24}),
    }
    for _col, (_, _kwargs) in contact_definitions.items():
        df[_col] = th.contact_scores(df, **_kwargs)["f1"]
    _nonlocal = th.contact_scores(df, min_sep=6)
    df["precision_sep6"] = _nonlocal["precision"]
    df["recall_sep6"] = _nonlocal["recall"]

    # template-free predictor: diagonal and |i-j| <= 2 only
    _band = {"0", "1-2"}
    df["baseline_full"] = th.contact_scores(df, predicted_bands=_band)["f1"]
    df["baseline_sep1"] = th.contact_scores(
        df, min_sep=1, predicted_bands=_band
    )["f1"]
    df["random_sep6"] = th.contact_density(df, 6)
    df["n_true_sep6"] = sum(df[f"n_true_{b}"] for b in th.BANDS[3:])

    # Fig. 1C bins: PyOpal identity in 0.1 steps, right-closed
    _edges = np.round(np.arange(0.2, 1.01, 0.1), 1)
    bin_labels = [f"{a:.1f}-{b:.1f}" for a, b in zip(_edges[:-1], _edges[1:])]
    df["pyopal_bin"] = pd.cut(
        df["pyopal_identity"], _edges, labels=bin_labels, include_lowest=True
    )
    df["aligned_length"] = df["pyopal_matches"] + df["pyopal_mismatches"]
    df["hssp_distance"] = 100 * df["pyopal_identity"] - th.hssp_threshold(
        df["aligned_length"]
    )
    return bin_labels, contact_definitions, df


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Identity and coverage definitions

    Three different measures of identity

    | name | computed by | alignment | definition |
    |---|---|---|---|
    | MMseqs2 `fident` (identity bins of the run, `idbin`) | `mmseqs search` inside mdF | local | **estimated from the alignment score.** mdF searches with `--alignment-mode 2` and without backtrace (`-a 0`), so MMseqs2 never counts identical residues; it converts the score per aligned column into an identity. |
    | MMseqs2 exact | `mmseqs align -a --alignment-mode 3` on the same pairs | the same local alignment | identical residues / alignment columns (gaps included) |
    | PyOpal identity (x-axis of Fig. 1B/C) | `FullResult.identity()` after Needleman-Wunsch | global | matches / (matches + mismatches); **gap columns are excluded** |

    Coverage also differs:

    - **MMseqs2 `qcov`/`tcov`**: the span of the local alignment on the query/template. The benchmark kept only templates with qcov and tcov ≥ 0.9 and E ≤ 1e-3.
    - **PyOpal `coverage("query")`** (`pyopal_coverage`): the query span between the first and last residue aligned to a template residue, divided by query length. Internal insertions count as covered; only terminal overhangs reduce it.
    - **Aligned fraction**: query residues aligned to a template residue / query length. The rest are insertions, which get synthetic contacts (`--generate-contacts`).
    """)
    return


@app.cell
def _(bin_labels, df, pd):
    ### identity and coverage variants per PyOpal identity bin (medians) ###
    _columns = {
        "fident": "MMseqs2 fident",
        "mm_exact_fident": "MMseqs2 exact",
        "pyopal_identity": "PyOpal",
        "pyopal_id_columns": "PyOpal incl. gaps",
        "pyopal_coverage": "PyOpal coverage",
        "pyopal_cov_aligned": "aligned fraction",
        "pyopal_synthetic_in_span": "insertions in span",
        "qcov": "MMseqs2 qcov",
        "tcov": "MMseqs2 tcov",
        "aligned_length": "aligned residues",
    }
    _grouped = df.groupby("pyopal_bin", observed=True)
    identity_table = (
        _grouped[list(_columns)]
        .median()
        .rename(columns=_columns)
        .assign(pairs=_grouped.size())
        .reindex(bin_labels)
        .round(3)
    )
    _check = pd.Series(
        {
            "max |rerun estimate - reported fident|": (
                df["mm_est_fident"] - df["fident"]
            )
            .abs()
            .max(),
            "PyOpal 0.2-0.3: share with exact MMseqs2 identity <= 0.3": (
                df.loc[df["pyopal_bin"] == "0.2-0.3", "mm_exact_fident"] <= 0.3
            ).mean(),
            "pairs with exact identity 1.0: median reported fident": df.loc[
                df["mm_exact_fident"] >= 0.999, "fident"
            ].median(),
        }
    ).round(3)
    identity_table, _check
    return


@app.cell
def _(LinearSegmentedColormap, LogNorm, PLOT_DIR, RAW_DIR, df, plt, th):
    ### reported and PyOpal identity against exact MMseqs2 identity ###
    _out_prefix = "identity_definitions"
    _cmap = LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
    )
    _panels = [
        ("fident", "MMseqs2 reported identity\n(score-based estimate)"),
        ("pyopal_identity", "PyOpal identity\n(global, gaps excluded)"),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(9, 4.2), sharex=True, sharey=True
    )
    for _ax, (_col, _label) in zip(_axes, _panels):
        _hb = _ax.hexbin(
            df["mm_exact_fident"],
            df[_col],
            gridsize=60,
            extent=(0.1, 1, 0.1, 1),
            cmap=_cmap,
            norm=LogNorm(),
            mincnt=1,
            linewidths=0,
        )
        _ax.plot([0.1, 1], [0.1, 1], color=th.INK_SECONDARY, linewidth=0.8)
        _ax.set_xlabel("MMseqs2 exact identity (local)")
        _ax.set_ylabel(_label)
        _ax.set_aspect("equal")
        th.style_axis(_ax)
    _fig.colorbar(_hb, ax=_axes, shrink=0.8, label="pairs")
    th.save_figure(_fig, PLOT_DIR, _out_prefix)

    df[
        [
            "query",
            "target",
            "idbin",
            "fident",
            "mm_exact_fident",
            "pyopal_identity",
        ]
    ].to_csv(RAW_DIR / f"{_out_prefix}.csv", index=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. How often a twilight-zone template is the only option

    Bins come from the full search of the 50,000 AFDB queries (Protein Diversity Dataset). The second table places
    each benchmark pair against the length-dependent identity threshold of
    Rost (1999)(HSSP identity-3D similarity curve), using PyOpal identity over aligned residues: a positive
    distance means the pair lies above the curve, where structural similarity is
    expected.

    Low-identity pairs are long because of the search filters, not by chance.
    Every template needed E ≤ 1e-3 against 214M AFDB sequences (about 45-50
    bits), and alignments at 20-30% identity earn only ~0.4 bits per aligned
    column (~2 at > 90%). With query and template coverage ≥ 0.9 the alignment
    spans almost the whole protein, so only long queries can yield a
    significant low-identity template; for the same query, low- and
    high-identity templates have the same aligned length. The HSSP curve
    encodes the same length-identity trade-off, so passing the E-value filter
    at full length places most pairs above it. The comparison shows that the
    filters admit only safe-zone templates; it is not independent evidence of
    structural similarity. Short proteins (< 100 aa) rarely receive a
    twilight-zone template.
    """)
    return


@app.cell
def _(bin_labels, df, np, pd):
    ### best hit and any hit per identity bin (MMseqs2 reported identity) ###
    _search = pd.read_csv(
        "data/source/landscape/mmseqs2_search/afdb_uniprot_v4_results.tsv",
        sep="\t",
        usecols=["query", "target", "fident"],
    ).query("query != target")
    _n_queries = 50_000
    _edges = np.round(np.arange(0, 1.01, 0.1), 1)
    _labels = [f"{a:.1f}-{b:.1f}" for a, b in zip(_edges[:-1], _edges[1:])]
    _bin = pd.cut(_search["fident"], _edges, labels=_labels, right=False)
    _best = pd.cut(
        _search.groupby("query")["fident"].max(),
        _edges,
        labels=_labels,
        right=False,
    )
    hit_availability = pd.DataFrame(
        {
            "queries with any hit (%)": _search.groupby(_bin, observed=False)[
                "query"
            ].nunique()
            / _n_queries
            * 100,
            "queries with best hit here (%)": _best.value_counts().reindex(
                _labels
            )
            / _n_queries
            * 100,
        }
    ).round(2)
    hit_availability.index = hit_availability.index.astype(str)
    hit_availability.loc["no hit"] = [
        np.nan,
        (_n_queries - _search["query"].nunique()) / _n_queries * 100,
    ]

    _grouped = df.groupby("pyopal_bin", observed=True)
    hssp_table = (
        pd.DataFrame(
            {
                "median aligned residues": _grouped["aligned_length"].median(),
                "median distance to HSSP curve (pp)": _grouped[
                    "hssp_distance"
                ].median(),
                "pairs above HSSP curve (%)": _grouped["hssp_distance"].apply(
                    lambda d: (d > 0).mean() * 100
                ),
            }
        )
        .reindex(bin_labels)
        .round(1)
    )

    ### why low-identity templates are long: E-value and coverage filters ###
    _significance = (
        pd.DataFrame(
            {
                "bits per aligned column (median)": (df["bits"] / df["alnlen"])
                .groupby(df["pyopal_bin"], observed=True)
                .median(),
                "aligned residue pairs (5th percentile)": _grouped[
                    "aligned_length"
                ].quantile(0.05),
                "pairs with E > 1e-10 (%)": _grouped["evalue"].apply(
                    lambda e: (e > 1e-10).mean() * 100
                ),
            }
        )
        .reindex(bin_labels)
        .round(2)
    )

    _lengths = {}
    with open("data/source/landscape/landscape_afdb_50k.fasta") as _f:
        for _line in _f:
            if _line.startswith(">"):
                _name = _line[1:].split()[0]
                _lengths[_name] = 0
            else:
                _lengths[_name] += len(_line.strip())
    _queries = pd.DataFrame({"length": pd.Series(_lengths)})
    _queries["any_hit"] = _queries.index.isin(_search["query"])
    _queries["low_template"] = _queries.index.isin(
        df.loc[df["pyopal_identity"] <= 0.4, "query"]
    )
    _by_length = _queries.groupby(
        pd.cut(_queries["length"], [0, 60, 100, 150, 250, 400, np.inf]),
        observed=True,
    )
    length_selection = pd.DataFrame(
        {
            "queries": _by_length.size(),
            "with any hit (%)": _by_length["any_hit"].mean() * 100,
            "with a template at identity <= 0.4, of those with a hit (%)": (
                _by_length["low_template"].sum()
                / _by_length["any_hit"].sum()
                * 100
            ),
        }
    ).round(1)

    # same query, low- vs high-identity template
    _low = df.query("pyopal_identity <= 0.4").groupby("query")[
        "aligned_length"
    ]
    _high = df.query("pyopal_identity > 0.6").groupby("query")[
        "aligned_length"
    ]
    _ratio = (_low.median() / _high.median()).dropna()
    _within_query = pd.Series(
        {
            "queries with templates at <= 0.4 and > 0.6": len(_ratio),
            "median aligned-length ratio (low / high)": round(
                _ratio.median(), 3
            ),
        }
    )
    (
        hit_availability,
        hssp_table,
        _significance,
        length_selection,
        _within_query,
    )
    return


@app.cell
def _(LogNorm, PLOT_DIR, RAW_DIR, bin_labels, df, np, plt, th):
    ### benchmark pairs against the HSSP curve ###
    _out_prefix = "twilight_hssp_distance"
    _max_length = 700
    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # identity vs aligned residue pairs, with the curve
    _shown = df["aligned_length"] <= _max_length
    _h = _ax1.hist2d(
        df.loc[_shown, "aligned_length"],
        100 * df.loc[_shown, "pyopal_identity"],
        bins=[
            np.linspace(0, _max_length, 71),
            np.linspace(0, 100, 101),
        ],  # y-bins from 0
        cmap="BuPu",
        norm=LogNorm(vmin=1),
    )
    _fig.colorbar(_h[3], ax=_ax1, label="Count (log scale)")
    _length = np.arange(12, _max_length + 1)
    _ax1.plot(
        _length,
        th.hssp_threshold(_length),
        color="black",
        linewidth=2,
        label="HSSP curve (Rost 1999)",
    )

    _ax1.set_ylim(0, 100)
    _ax1.set_xlabel("Aligned residue pairs (L)", fontsize=14)
    _ax1.set_ylabel("Pyopal identity (%)", fontsize=14)
    _ax1.tick_params(labelsize=14)
    _ax1.legend(loc="upper right", fontsize=11)
    _ax1.grid(alpha=0.3)

    # distance to the curve per identity bin
    _positions = np.array([float(b.split("-")[0]) + 0.05 for b in bin_labels])
    _grouped = df.groupby("pyopal_bin", observed=True)
    th.binned_boxplot(
        _ax2,
        [
            _grouped.get_group(b)["hssp_distance"].to_numpy()
            for b in bin_labels
        ],
        _positions,
        bin_labels,
        "#8a3ffc",
    )
    _ax2.axhline(0, color="black", linewidth=1)
    _ax2.set_xlim(0.2, 1.0)
    _ax2.set_ylabel("Distance to HSSP curve (pp)", fontsize=14)
    _ax2.set_xlabel("Pyopal identity bins", fontsize=14, labelpad=30)

    for _ax, _letter in zip((_ax1, _ax2), "AB"):
        _ax.text(
            -0.12,
            1.05,
            _letter,
            transform=_ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    th.save_figure(_fig, PLOT_DIR, _out_prefix)

    df[
        [
            "query",
            "target",
            "idbin",
            "pyopal_identity",
            "aligned_length",
            "hssp_distance",
        ]
    ].to_csv(RAW_DIR / f"{_out_prefix}.csv", index=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. F1 by contact range

    The preprint F1 counts the full L × L cmap: the diagonal plus both mirrored
    triangles. Mirroring doubles every off-diagonal count, so it leaves precision
    and recall unchanged; the diagonal and the contacts between residues one or
    two apart are what inflate the score. At 6 Å Cα they are almost always in
    contact, whatever the template. The columns below count the upper triangle
    only, from |i-j| ≥ 1 (no mirror, no diagonal) to long-range contacts (CASP:
    short ≥ 6, medium ≥ 12, long ≥ 24).

    Two reference predictors use no template at all:

    - **chain baseline**: calls every pair with |i-j| ≤ 2 a contact and nothing else;
    - **random**: the contact density at |i-j| ≥ 6, which is the expected F1 of random guesses.

    Pairs without any true contact in a range are left out of that column.
    """)
    return


@app.cell
def _(bin_labels, contact_definitions, df, pd, th):
    ### share of true contacts per separation band (upper triangle + diagonal) ###
    _true = df[[f"n_true_{b}" for b in th.BANDS]].sum()
    contact_share = (_true / _true.sum() * 100).round(1)
    contact_share.index = [f"|i-j| = {b}" for b in th.BANDS]

    ### median [IQR] F1 per PyOpal identity bin ###
    def _summary(values):
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        return f"{median:.2f} [{q1:.2f}-{q3:.2f}]"

    _columns = {col: label for col, (label, _) in contact_definitions.items()}
    _columns.update(
        {
            "baseline_full": "chain baseline, full map",
            "baseline_sep1": "chain baseline, |i-j| ≥ 1",
            "random_sep6": "random, |i-j| ≥ 6",
            "precision_sep6": "precision |i-j| ≥ 6",
            "recall_sep6": "recall |i-j| ≥ 6",
        }
    )
    _grouped = df.groupby("pyopal_bin", observed=True)
    f1_table = (
        _grouped[list(_columns)]
        .agg(_summary)
        .rename(columns=_columns)
        .assign(pairs=_grouped.size())
        .reindex(bin_labels)
    )
    _missing = pd.Series(
        {
            label: f"{df[col].isna().mean() * 100:.1f}%"
            for col, (label, _) in contact_definitions.items()
        },
        name="pairs without true contacts",
    )
    contact_share, f1_table, _missing
    return


@app.cell
def _(PLOT_DIR, RAW_DIR, bin_labels, contact_definitions, df, np, plt, th):
    ### F1 per identity bin, one Fig. 1C-style panel per contact definition ###
    _out_prefix = "twilight_f1_by_contact_range"
    _panels = {
        "f1_full": "baseline_full",
        "f1_sep1": "baseline_sep1",
        "f1_sep6": None,
    }
    _positions = np.array([float(b.split("-")[0]) + 0.05 for b in bin_labels])
    _grouped = df.groupby("pyopal_bin", observed=True)

    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for _ax, (_col, _baseline) in zip(_axes, _panels.items()):
        th.binned_boxplot(
            _ax,
            [_grouped.get_group(b)[_col].to_numpy() for b in bin_labels],
            _positions,
            bin_labels,
            "#1192e8",
        )
        if _baseline:
            # template-free chain baseline: median per bin
            _ax.hlines(
                _grouped[_baseline].median().reindex(bin_labels),
                _positions - 0.04,
                _positions + 0.04,
                color="#eb6834",
                linewidth=2,
                zorder=3,
                label="chain baseline (median)",
            )
            _ax.legend(loc="lower left", fontsize=11)
        _ax.set_xlim(0.2, 1.0)
        _ax.set_title(contact_definitions[_col][0], fontsize=14)
        _ax.set_xlabel("Pyopal identity bins", fontsize=14, labelpad=30)
    _axes[0].set_ylabel("F1-score", fontsize=14)
    _axes[0].set_ylim(0, 1.02)
    th.save_figure(_fig, PLOT_DIR, _out_prefix)

    df[
        ["query", "target", "idbin", "pyopal_identity", "pyopal_bin"]
        + list(_panels)
        + ["baseline_full", "baseline_sep1"]
    ].to_csv(RAW_DIR / f"{_out_prefix}.csv", index=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Failures at low identity

    Low-identity pairs are those at PyOpal identity ≤ 0.4. A pair fails when
    it recovers fewer than half of the non-local contacts (F1 at |i-j| ≥ 6
    below 0.5). Pairs with fewer than 10 true non-local contacts are left out,
    because their F1 rests on a handful of contacts.

    AUROC is the probability that a failing pair has a higher feature value
    than a successful one (0.5 = no difference).

    Fold class comes from TED (Lau et al., Science 2024; Zenodo
    10.5281/zenodo.13908086). TED chopped every AFDB v4 model into domains
    (consensus of Merizo, Chainsaw and UniDoc) and labelled them with a CATH
    superfamily (level H) or fold (level T) from Foldseek and Foldclass
    searches against CATH. A query takes the class of its largest
    CATH-labelled domain. Queries whose TED domains carry no CATH label, and
    queries without any TED domain, form their own groups.
    """)
    return


@app.cell
def _(pd, th):
    ### structure features and CATH classes of queries ###
    _features = pd.read_csv("data/generated/landscape_structure_features.csv")
    query_features = (
        _features.query("kind == 'query'").drop(columns="kind").set_index("id")
    )
    template_features = (
        _features.query("kind == 'target'")
        .drop(
            columns=[
                "kind",
                "relative_contact_order",
                "long_range_contacts_per_res",
            ]
        )
        .set_index("id")
    )

    # TED domains of queries and templates; CATH names from cath-names.txt
    _domains = pd.read_csv("data/generated/landscape_ted_domains.csv")
    _names = th.read_cath_names("/mnt/storage_10T/software/ted/cath-names.txt")
    query_cath = th.protein_cath(_domains, _names)
    return query_cath, query_features, template_features


@app.cell
def _(df, query_cath, query_features, template_features):
    ### low-identity pairs with query/template features ###
    low_identity = (
        df.query("pyopal_identity <= 0.4 and n_true_sep6 >= 10")
        .join(query_features.add_suffix("_query"), on="query")
        .join(template_features.add_suffix("_template"), on="target")
        .join(query_cath, on="query")
    )
    low_identity["cath_class"] = low_identity["cath_class"].fillna(
        "no TED domain"
    )
    low_identity["ted_domains"] = low_identity["ted_domains"].fillna(0)
    low_identity["ted_fraction"] = (
        low_identity["ted_residues"].fillna(0) / low_identity["length_query"]
    )
    low_identity["failure"] = low_identity["f1_sep6"] < 0.5
    return (low_identity,)


@app.cell
def _(low_identity, pd, th):
    ### which features separate failures from successes ###
    _features = {
        "tm_score": "TM-score template vs query",
        "pyopal_identity": "PyOpal identity",
        "mm_exact_fident": "MMseqs2 exact identity",
        "pyopal_cov_aligned": "aligned fraction of query",
        "pyopal_synthetic_in_span": "insertions in span",
        "mm_exact_bits": "MMseqs2 bit score",
        "length_query": "query length",
        "mean_plddt_query": "query mean pLDDT",
        "frac_plddt_lt70_query": "query pLDDT < 70 (fraction)",
        "mean_plddt_template": "template mean pLDDT",
        "frac_plddt_lt70_template": "template pLDDT < 70 (fraction)",
        "frac_helix_query": "query helix fraction",
        "frac_strand_query": "query strand fraction",
        "relative_contact_order_query": "query relative contact order",
        "long_range_contacts_per_res_query": "query long-range contacts per residue",
        "ted_domains": "query TED domains",
        "ted_fraction": "query residues in TED domains (fraction)",
    }
    _grouped = low_identity.groupby("failure")
    failure_features = (
        pd.DataFrame(
            {
                "success (median)": _grouped[list(_features)]
                .median()
                .loc[False],
                "failure (median)": _grouped[list(_features)]
                .median()
                .loc[True],
                "AUROC": [
                    th.rank_auc(low_identity[c], low_identity["failure"])
                    for c in _features
                ],
            }
        )
        .rename(index=_features)
        .round(3)
    )
    _summary = pd.Series(
        {
            "low-identity pairs": len(low_identity),
            "failures (F1 |i-j| >= 6 < 0.5)": int(
                low_identity["failure"].sum()
            ),
            "failure rate": round(low_identity["failure"].mean(), 3),
            "TM-score < 0.5": int((low_identity["tm_score"] < 0.5).sum()),
            "failures with TM-score < 0.5": int(
                (
                    low_identity["failure"] & (low_identity["tm_score"] < 0.5)
                ).sum()
            ),
        }
    )
    failure_features.sort_values("AUROC"), _summary
    return


@app.cell
def _(df, low_identity, pd, query_cath):
    ### non-local F1 and failure rate per CATH class and architecture ###
    _high = (
        df.query("pyopal_identity > 0.6 and n_true_sep6 >= 10")
        .join(query_cath, on="query")
        .assign(cath_class=lambda d: d["cath_class"].fillna("no TED domain"))
    )

    def _by(low, column, min_pairs):
        _low = low.groupby(column)
        table = pd.DataFrame(
            {
                "pairs (identity <= 0.4)": _low.size(),
                "median F1 |i-j| >= 6": _low["f1_sep6"].median(),
                "failure rate": _low["failure"].mean(),
                "median F1 |i-j| >= 6 (identity > 0.6)": _high.groupby(column)[
                    "f1_sep6"
                ].median(),
            }
        )
        return (
            table[table.iloc[:, 0] >= min_pairs]
            .sort_values("pairs (identity <= 0.4)", ascending=False)
            .round(3)
        )

    cath_class_table = _by(low_identity, "cath_class", 1)
    _single_domain = _by(
        low_identity.query("ted_domains == 1"), "cath_class", 1
    )
    _architectures = _by(low_identity, "cath_architecture", 50)
    _architectures.insert(
        0,
        "name",
        _architectures.index.map(
            dict(
                zip(
                    query_cath["cath_architecture"],
                    query_cath["cath_architecture_name"],
                )
            )
        ),
    )
    cath_class_table, _single_domain, _architectures
    return (cath_class_table,)


@app.cell
def _(
    PLOT_DIR,
    RAW_DIR,
    cath_class_table,
    df,
    low_identity,
    np,
    pd,
    plt,
    query_cath,
    th,
):
    ### non-local F1 per CATH class, low vs high identity ###
    _out_prefix = "twilight_f1_by_cath_class"
    _unlabelled = ["no CATH label", "no TED domain"]
    _classes = cath_class_table.index[cath_class_table.iloc[:, 0] >= 20]
    _classes = [c for c in _classes if c not in _unlabelled] + [
        c for c in _unlabelled if c in _classes
    ]
    _high = (
        df.query("pyopal_identity > 0.6 and n_true_sep6 >= 10")
        .join(query_cath, on="query")
        .assign(cath_class=lambda d: d["cath_class"].fillna("no TED domain"))
    )
    _groups = {
        "PyOpal identity ≤ 0.4": low_identity,
        "PyOpal identity > 0.6": _high,
    }
    _positions = np.arange(len(_classes))
    _labels = ["\n".join(c.rsplit(" ", 1)) for c in _classes]

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for _ax, (_title, _data) in zip(_axes, _groups.items()):
        th.binned_boxplot(
            _ax,
            [
                _data.loc[_data["cath_class"] == c, "f1_sep6"].to_numpy()
                for c in _classes
            ],
            _positions,
            _labels,
            "#1192e8",
            width=0.5,
            rotation=0,
            counts_y=-0.17,
        )
        _ax.set_xlim(-0.6, len(_classes) - 0.4)
        _ax.set_title(_title, fontsize=14)
        _ax.set_xlabel("CATH class of the query", fontsize=14, labelpad=30)
    _axes[0].set_ylabel("F1-score, |i-j| ≥ 6", fontsize=14)
    _axes[0].set_ylim(0, 1.02)
    th.save_figure(_fig, PLOT_DIR, _out_prefix)

    pd.concat([d.assign(identity_group=g) for g, d in _groups.items()])[
        [
            "query",
            "target",
            "pyopal_identity",
            "identity_group",
            "cath_class",
            "f1_sep6",
        ]
    ].to_csv(RAW_DIR / f"{_out_prefix}.csv", index=False)
    _fig
    return


if __name__ == "__main__":
    app.run()
