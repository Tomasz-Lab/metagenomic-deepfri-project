import numpy as np
import pandas as pd

# separation bands of scripts/contact_counts_by_separation.py (upper triangle)
BANDS = ["0", "1-2", "3-5", "6-11", "12-23", "24+"]
BAND_START = [0, 1, 3, 6, 12, 24]

# ink and hairline colors of style_axis
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _bands_from(min_sep):
    return [b for b, start in zip(BANDS, BAND_START) if start >= min_sep]


def _totals(counts, kind, min_sep, predicted_bands=None):
    """Sum counts over the bands of a contact definition.

    min_sep=None is the published definition: the full L x L map, i.e. the
    diagonal plus both (mirrored) triangles.
    predicted_bands overrides the predicted map with a band predictor that
    calls every position in these bands a contact.
    """
    bands = BANDS if min_sep is None else _bands_from(min_sep)
    if predicted_bands is not None and kind in ("tp", "pred"):
        source = "true" if kind == "tp" else "pos"
        values = [
            counts[f"n_{source}_{b}"] * (b in predicted_bands) for b in bands
        ]
    else:
        values = [counts[f"n_{kind}_{b}"] for b in bands]
    weights = [1 if (min_sep is not None or b == "0") else 2 for b in bands]
    return sum(w * v for w, v in zip(weights, values))


def contact_scores(counts, min_sep=None, predicted_bands=None):
    """Precision, recall and F1 per pair for one contact definition.

    Pairs without a true contact in the definition get NaN.
    """
    tp, pred, true = (
        _totals(counts, k, min_sep, predicted_bands)
        for k in ("tp", "pred", "true")
    )
    precision = tp / pred.where(pred > 0)
    recall = tp / true.where(true > 0)
    f1 = (2 * precision * recall / (precision + recall)).fillna(0)
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1.where(true > 0),
        }
    )


def contact_density(counts, min_sep):
    """Fraction of residue pairs in contact: F1 of a random predictor."""
    bands = _bands_from(min_sep)
    true = sum(counts[f"n_true_{b}"] for b in bands)
    positions = sum(counts[f"n_pos_{b}"] for b in bands)
    return true / positions


def hssp_threshold(aligned_length):
    """HSSP-curve identity threshold (%) for L aligned residue pairs.

    Rost (1999), Protein Eng. 12:85-94; HVAL = PIDE - threshold, where PIDE
    is 100 x identical residues / L and L counts aligned pairs, not gaps.
    """
    length = np.asarray(aligned_length, dtype=float)
    curve = 480 * length ** (-0.32 * (1 + np.exp(-length / 1000)))
    return np.select([length <= 11, length > 450], [100.0, 19.5], curve)


def read_cath_names(path):
    """CATH node (e.g. '3' or '3.40') -> name, from cath-names.txt."""
    names = pd.read_csv(
        path,
        sep=r"\s{4}",
        comment="#",
        header=None,
        names=["node", "domain", "name"],
        dtype=str,
        engine="python",
    )
    return dict(zip(names["node"], names["name"].str.lstrip(":")))


def protein_cath(domains, names):
    """Protein-level CATH summary from TED domains.

    A protein takes the class and architecture of its largest CATH-labelled
    domain; proteins whose domains carry no CATH label get "no CATH label".
    """
    labelled = domains.dropna(subset=["cath_label"]).copy()
    parts = labelled["cath_label"].str.split(".")
    labelled["cath_class"] = parts.str[0].map(names)
    labelled["cath_architecture"] = parts.str[:2].str.join(".")
    largest = (
        labelled.sort_values("nres_domain", ascending=False)
        .drop_duplicates("model")
        .set_index("model")
    )
    grouped = domains.groupby("model")
    summary = pd.DataFrame(
        {
            "ted_domains": grouped.size(),
            "ted_residues": grouped["nres_domain"].sum(),
            "cath_classes": labelled.groupby("model")["cath_class"].nunique(),
        }
    )
    summary["cath_class"] = largest["cath_class"]
    summary["cath_class"] = summary["cath_class"].fillna("no CATH label")
    summary["cath_architecture"] = largest["cath_architecture"]
    summary["cath_architecture_name"] = summary["cath_architecture"].map(names)
    return summary


def rank_auc(values, labels):
    """Probability that a positive scores higher than a negative (ties 1/2)."""
    values, labels = pd.Series(values), pd.Series(labels).astype(bool)
    keep = values.notna()
    values, labels = values[keep], labels[keep]
    n_pos, n_neg = labels.sum(), (~labels).sum()
    ranks = values.rank()
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def save_figure(fig, plot_dir, name):
    for fmt in ("svg", "pdf", "png"):
        fig.savefig(plot_dir / f"{name}.{fmt}", bbox_inches="tight", dpi=300)


def binned_boxplot(
    ax,
    values_by_bin,
    positions,
    labels,
    color,
    width=0.075,
    rotation=30,
    counts_y=-0.12,
    seed=0,
):
    """Box plot per bin in the style of Fig. 1B/C.

    Grey jittered points behind a semi-transparent box without fliers, with
    the number of values per bin under each tick (as in gc_data_analysis).
    The defaults fit identity bins of width 0.1; counts_y is the height of
    the counts in axes coordinates.
    """
    rng = np.random.default_rng(seed)
    values_by_bin = [np.asarray(v)[~np.isnan(v)] for v in values_by_bin]
    for pos, values in zip(positions, values_by_bin):
        ax.scatter(
            pos + rng.normal(0, width / 7.5, size=len(values)),
            values,
            s=10,
            alpha=0.3,
            color="gray",
            edgecolors="none",
            zorder=1,
        )
    ax.boxplot(
        values_by_bin,
        positions=positions,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor=color, color="black", alpha=0.50),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        showfliers=False,
        zorder=2,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, fontsize=14)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", labelsize=14)
    for pos, values in zip(positions, values_by_bin):
        ax.text(
            pos,
            counts_y,
            f"({len(values)})",
            ha="center",
            va="top",
            rotation=rotation,
            fontsize=10,
            color="0.5",
            transform=ax.get_xaxis_transform(),
        )
    ax.grid(alpha=0.3)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK)
