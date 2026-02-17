from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Literal, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Metric = Literal["jaccard", "overlap_min", "overlap_A", "overlap_B"]

@dataclass(frozen=True)
class BoxplotTheme:
    boxplot: dict
    x_tick_labelsize: int = 8


DEFAULT_THEME = BoxplotTheme(
    boxplot=dict(
        widths=0.55,
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
    ),
    x_tick_labelsize=8,
)

def _prepare_method_df(
    df: pd.DataFrame,
    protein_col: str,
    term_col: str,
    ic_col: str,
    score_col: Optional[str] = None,
    score_min: Optional[float] = None,
) -> pd.DataFrame:
    cols = [protein_col, term_col, ic_col]
    if score_col is not None and score_col in df.columns:
        cols.append(score_col)

    x = df[cols].copy()
    x = x.dropna(subset=[protein_col, term_col, ic_col])

    if score_min is not None and score_col is not None and score_col in x.columns:
        x[score_col] = pd.to_numeric(x[score_col], errors="coerce")
        x = x.dropna(subset=[score_col])
        x = x[x[score_col] >= float(score_min)]

    x[protein_col] = x[protein_col].astype(str)
    x[term_col] = x[term_col].astype(str)
    x[ic_col] = pd.to_numeric(x[ic_col], errors="coerce")
    x = x.dropna(subset=[ic_col])

    # Keep unique protein-term (multiple rows can exist; we only care presence)
    x = x.drop_duplicates(subset=[protein_col, term_col, ic_col])
    return x

def _assign_ic_bins(
    ic: pd.Series,
    ic_min: float,
    ic_max: float,
    bin_width: float,
) -> pd.Series:
    # bins like [1,2), [2,3), ...; labels 1..13
    edges = np.arange(ic_min, ic_max + bin_width, bin_width)
    labels = [f"{int(e)}" for e in edges[:-1]]  # label by left edge integer
    return pd.cut(ic, bins=edges, right=False, labels=labels, include_lowest=True)

def concordance_by_ic(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    protein_col: str = "Protein",
    term_col: str = "GO_term",
    ic_col: str = "IC",
    score_col: Optional[str] = None,
    score_min: Optional[float] = None,
    ic_min: float = 1.0,
    ic_max: float = 14.0,     # gives bins 1..13 for width 1.0
    bin_width: float = 1.0,
    metric: Metric = "jaccard",
    drop_propagated: Optional[bool] = None,  # True = only originals; False = only propagated; None = both
    propagated_col: str = "Propagated",
    require_both: bool = False,  # If True, only include proteins with annotations in both datasets (nA > 0 AND nB > 0)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-protein concordance stratified by IC bins.

    Returns:
      per_protein: columns [Protein, ic_bin, nA, nB, n_intersection, n_union, concordance]
      per_bin_summary: columns [ic_bin, N_proteins, median, mean]

    Args:
      require_both: If True, only include proteins that have annotations in both
                    datasets for a given IC bin (nA > 0 AND nB > 0). If False,
                    include proteins with annotations in either dataset (nA > 0 OR nB > 0).

    metric options:
      - "jaccard": |A∩B| / |A∪B|
      - "overlap_min": |A∩B| / min(|A|,|B|)  (if both nonzero)
      - "overlap_A": |A∩B| / |A|
      - "overlap_B": |A∩B| / |B|
    """
    A = df_a.copy()
    B = df_b.copy()

    if drop_propagated is not None and propagated_col in A.columns and propagated_col in B.columns:
        A = A[A[propagated_col] == bool(drop_propagated)]
        B = B[B[propagated_col] == bool(drop_propagated)]

    A = _prepare_method_df(A, protein_col, term_col, ic_col, score_col=score_col, score_min=score_min)
    B = _prepare_method_df(B, protein_col, term_col, ic_col, score_col=score_col, score_min=score_min)

    A["ic_bin"] = _assign_ic_bins(A[ic_col], ic_min=ic_min, ic_max=ic_max, bin_width=bin_width)
    B["ic_bin"] = _assign_ic_bins(B[ic_col], ic_min=ic_min, ic_max=ic_max, bin_width=bin_width)
    A = A.dropna(subset=["ic_bin"])
    B = B.dropna(subset=["ic_bin"])

    # Presence tables (protein, ic_bin, term)
    A3 = A[[protein_col, "ic_bin", term_col]].drop_duplicates()
    B3 = B[[protein_col, "ic_bin", term_col]].drop_duplicates()

    # sizes per protein/bin
    nA = A3.groupby([protein_col, "ic_bin"], as_index=False).size().rename(columns={"size": "nA"})
    nB = B3.groupby([protein_col, "ic_bin"], as_index=False).size().rename(columns={"size": "nB"})

    # intersection per protein/bin
    inter = A3.merge(B3, on=[protein_col, "ic_bin", term_col], how="inner")
    nI = inter.groupby([protein_col, "ic_bin"], as_index=False).size().rename(columns={"size": "n_intersection"})

    # combine and compute union
    per = nA.merge(nB, on=[protein_col, "ic_bin"], how="outer").merge(
        nI, on=[protein_col, "ic_bin"], how="left"
    )
    per["nA"] = per["nA"].fillna(0).astype(int)
    per["nB"] = per["nB"].fillna(0).astype(int)
    per["n_intersection"] = per["n_intersection"].fillna(0).astype(int)
    per["n_union"] = (per["nA"] + per["nB"] - per["n_intersection"]).astype(int)

    # concordance
    if metric == "jaccard":
        denom = per["n_union"].replace(0, np.nan)
        per["concordance"] = (per["n_intersection"] / denom).fillna(0.0)
    elif metric == "overlap_min":
        denom = np.minimum(per["nA"], per["nB"]).replace(0, np.nan)
        per["concordance"] = (per["n_intersection"] / denom).fillna(0.0)
    elif metric == "overlap_A":
        denom = per["nA"].replace(0, np.nan)
        per["concordance"] = (per["n_intersection"] / denom).fillna(0.0)
    elif metric == "overlap_B":
        denom = per["nB"].replace(0, np.nan)
        per["concordance"] = (per["n_intersection"] / denom).fillna(0.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # summary per bin
    # Filter based on require_both parameter
    if require_both:
        # Only include proteins with annotations in both datasets (nA > 0 AND nB > 0)
        per_with_data = per[(per["nA"] > 0) & (per["nB"] > 0)]
    else:
        # Include proteins with annotations in either dataset (nA > 0 OR nB > 0)
        per_with_data = per[(per["nA"] > 0) | (per["nB"] > 0)]
    
    summ = (
        per_with_data.groupby("ic_bin", as_index=False)
        .agg(
            N_proteins=(protein_col, "nunique"),
            median=("concordance", "median"),
            mean=("concordance", "mean"),
        )
        .sort_values("ic_bin")
        .reset_index(drop=True)
    )

    # Return filtered per_with_data so boxplot matches summary statistics
    return per_with_data, summ

def plot_concordance_boxplot(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    theme: BoxplotTheme = DEFAULT_THEME,
    figsize: Tuple[float, float] = (6, 4),
    title: Optional[str] = None,
    ylabel: str = "Concordance (0–1)",
) -> plt.Figure:
    """
    Boxplot of per-protein concordance per IC bin, with N per bin shown as 'N=...'.
    """
    # Ensure bins in order
    bins = per_bin_summary["ic_bin"].astype(str).tolist()

    data: List[np.ndarray] = []
    for b in bins:
        vals = per_protein.loc[per_protein["ic_bin"].astype(str) == b, "concordance"].to_numpy()
        data.append(vals)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    bp = ax.boxplot(data, labels=bins, **theme.boxplot)

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=20)  # Add padding to prevent overlap

    ax.tick_params(axis="x", labelsize=theme.x_tick_labelsize)
    ax.set_ylim(0, 1)

    # Add N annotations below x-axis labels
    # Use transform to position relative to axes (0 = bottom of axes, negative = below)
    for i, (_, row) in enumerate(per_bin_summary.iterrows(), start=1):
        ax.text(
            i, -0.08, f"N={int(row['N_proteins'])}", 
            ha="center", va="top", 
            fontsize=8, rotation=30,
            transform=ax.get_xaxis_transform()  # Use x-axis transform for positioning
        )

    # Adjust layout to accommodate annotations below x-axis
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave bottom margin for N annotations
    return fig


def _is_heavily_concentrated_near_one(
    values: np.ndarray,
    threshold: float = 0.8,
    concentration_threshold: float = 0.5,
) -> bool:
    """
    Detect if a distribution is heavily concentrated near 1.0.
    
    Args:
        values: Array of concordance values (0-1)
        threshold: Consider values >= threshold as "near 1.0"
        concentration_threshold: Fraction of values that must be >= threshold
    
    Returns:
        True if distribution is heavily concentrated near 1.0
    """
    if len(values) == 0:
        return False
    near_one_fraction = np.sum(values >= threshold) / len(values)
    return near_one_fraction >= concentration_threshold


def plot_concordance_violin(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (8, 4),
    title: Optional[str] = None,
    ylabel: str = "Concordance (0–1)",
    show_medians: bool = True,
) -> plt.Figure:
    """
    Violin plot of per-protein concordance per IC bin.
    Better than boxplots for distributions heavily concentrated near 1.0,
    as it shows the full distribution shape including density.
    """
    bins = per_bin_summary["ic_bin"].astype(str).tolist()
    
    data: List[np.ndarray] = []
    for b in bins:
        vals = per_protein.loc[per_protein["ic_bin"].astype(str) == b, "concordance"].to_numpy()
        data.append(vals)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    positions = np.arange(1, len(bins) + 1)
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showmedians=show_medians,
        showextrema=True,
    )
    
    # Style the violins
    for pc in parts["bodies"]:
        pc.set_facecolor("#1192e8")
        pc.set_edgecolor("black")
        pc.set_alpha(0.50)
        pc.set_linewidth(1.0)
    
    # Style median line
    if show_medians:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(bins)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10)
    
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(0, 1)
    
    # Add N annotations below x-axis labels
    for i, (_, row) in enumerate(per_bin_summary.iterrows(), start=1):
        ax.text(
            i, -0.08, f"n={int(row['N_proteins'])}",
            ha="center", va="top",
            fontsize=9, rotation=30,
            transform=ax.get_xaxis_transform()
        )
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.99])
    return fig


def plot_concordance_ecdf(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
    xlabel: str = "Concordance (0–1)",
    ylabel: str = "Cumulative Probability",
    alpha: float = 0.7,
) -> plt.Figure:
    """
    Empirical Cumulative Distribution Function (ECDF) plot.
    Excellent for visualizing distributions heavily concentrated near 1.0,
    as it clearly shows the fraction of values at each concordance level.
    """
    bins = per_bin_summary["ic_bin"].astype(str).tolist()
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    for b in bins:
        vals = per_protein.loc[per_protein["ic_bin"].astype(str) == b, "concordance"].to_numpy()
        if len(vals) > 0:
            # Sort values for ECDF
            sorted_vals = np.sort(vals)
            # ECDF: y = (1 to n) / n
            y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, y, label=f"Bin {b}", alpha=alpha, linewidth=1.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    
    # Add N annotations in legend or as text
    n_text = ", ".join([f"Bin {b}: N={int(row['N_proteins'])}" 
                        for b, (_, row) in zip(bins, per_bin_summary.iterrows())])
    ax.text(0.02, 0.98, n_text, transform=ax.transAxes,
            fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    fig.tight_layout()
    return fig


def plot_concordance_histogram(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: str = "Concordance (0–1)",
    ylabel: str = "Density",
    bins: int = 30,
    alpha: float = 0.6,
    density: bool = True,
) -> plt.Figure:
    """
    Histogram/density plot showing distribution shape.
    Useful for seeing concentration near 1.0 and distribution details.
    """
    bins_list = per_bin_summary["ic_bin"].astype(str).tolist()
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    for b in bins_list:
        vals = per_protein.loc[per_protein["ic_bin"].astype(str) == b, "concordance"].to_numpy()
        if len(vals) > 0:
            n_proteins = int(per_bin_summary[per_bin_summary["ic_bin"].astype(str) == b]["N_proteins"].iloc[0])
            ax.hist(
                vals,
                bins=bins,
                alpha=alpha,
                label=f"Bin {b} (N={n_proteins})",
                density=density,
                edgecolor="black",
                linewidth=0.5,
            )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=20)
    
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    
    fig.tight_layout()
    return fig


def plot_concordance_split_axis(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    ylabel: str = "Concordance (0–1)",
    split_threshold: float = 0.7,
    lower_height_ratio: float = 0.4,
    theme: BoxplotTheme = DEFAULT_THEME,
) -> plt.Figure:
    """
    Split-axis plot: zoomed view of lower values (0 to split_threshold) 
    and full view showing concentration at 1.0.
    Excellent for distributions heavily concentrated near 1.0.
    """
    bins = per_bin_summary["ic_bin"].astype(str).tolist()
    
    data: List[np.ndarray] = []
    for b in bins:
        vals = per_protein.loc[per_protein["ic_bin"].astype(str) == b, "concordance"].to_numpy()
        data.append(vals)
    
    fig = plt.figure(figsize=figsize)
    
    # Create split axes
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[1 - lower_height_ratio, lower_height_ratio], hspace=0.05)
    ax_upper = fig.add_subplot(gs[0])
    ax_lower = fig.add_subplot(gs[1])
    
    # Upper plot: full range (0-1), focus on values >= split_threshold
    bp_upper = ax_upper.boxplot(data, labels=bins, **theme.boxplot)
    ax_upper.set_ylim(split_threshold, 1.0)
    ax_upper.set_ylabel(ylabel, fontsize=10)
    ax_upper.tick_params(axis="x", labelsize=theme.x_tick_labelsize, labelbottom=False)
    ax_upper.spines["bottom"].set_visible(False)
    ax_upper.tick_params(axis="x", bottom=False)
    
    # Lower plot: zoomed view (0 to split_threshold)
    bp_lower = ax_lower.boxplot(data, labels=bins, **theme.boxplot)
    ax_lower.set_ylim(0, split_threshold)
    ax_lower.set_xlabel("IC Bin", fontsize=10)
    ax_lower.tick_params(axis="x", labelsize=theme.x_tick_labelsize)
    ax_lower.spines["top"].set_visible(False)
    ax_lower.tick_params(axis="x", top=False)
    
    # Add break marks
    d = 0.015  # size of diagonal lines
    kwargs = dict(transform=ax_upper.transAxes, color="k", clip_on=False, linewidth=1)
    ax_upper.plot((-d, +d), (-d, +d), **kwargs)
    ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax_lower.transAxes)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)
    
    # Add N annotations below x-axis labels on lower plot
    for i, (_, row) in enumerate(per_bin_summary.iterrows(), start=1):
        ax_lower.text(
            i, -0.15, f"N={int(row['N_proteins'])}",
            ha="center", va="top",
            fontsize=8, rotation=30,
            transform=ax_lower.get_xaxis_transform()
        )
    
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return fig


def plot_concordance_auto(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    concentration_threshold: float = 0.5,
    figsize: Tuple[float, float] = (6, 4),
    title: Optional[str] = None,
    ylabel: str = "Concordance (0–1)",
    theme: BoxplotTheme = DEFAULT_THEME,
) -> plt.Figure:
    """
    Automatically choose visualization based on distribution characteristics.
    Uses violin plots if distributions are heavily concentrated near 1.0,
    otherwise uses boxplots.
    """
    bins = per_bin_summary["ic_bin"].astype(str).tolist()
    
    # Check if any bin is heavily concentrated near 1.0
    use_violin = False
    for b in bins:
        vals = per_protein.loc[per_protein["ic_bin"].astype(str) == b, "concordance"].to_numpy()
        if _is_heavily_concentrated_near_one(vals, concentration_threshold=concentration_threshold):
            use_violin = True
            break
    
    if use_violin:
        return plot_concordance_violin(
            per_protein, per_bin_summary,
            figsize=figsize, title=title, ylabel=ylabel
        )
    else:
        return plot_concordance_boxplot(
            per_protein, per_bin_summary,
            theme=theme, figsize=figsize, title=title, ylabel=ylabel
        )
