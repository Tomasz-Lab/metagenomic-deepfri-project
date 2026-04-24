from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Literal, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

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
    """
    Assign IC values to bins. Creates bins up to ic_max (inclusive), and aggregates
    values >= ic_max + bin_width into an overflow bin.
    
    Example: ic_min=1, ic_max=10, bin_width=1.0
      - Creates bins: 1, 2, 3, ..., 10 (representing [1,2), [2,3), ..., [10,11))
      - Overflow bin: 11+ (for values >= 11)
    """
    # Create regular bins up to ic_max (inclusive)
    # If ic_max=10, bin_width=1.0: bins are [1,2), [2,3), ..., [10,11)
    # So edges should be: [1, 2, 3, ..., 10, 11]
    regular_edges = np.arange(ic_min, ic_max + bin_width + bin_width, bin_width)
    
    # Create labels for regular bins (using left edge of each bin)
    regular_labels = [f"{int(e)}" for e in regular_edges[:-1]]
    
    # Create overflow bin label
    overflow_threshold = ic_max + bin_width
    overflow_label = f"{int(overflow_threshold)}+"
    
    # Initialize result series with NaN (will be filled with regular bins or overflow)
    result = pd.Series(index=ic.index, dtype=object)
    
    # Assign regular bins using pd.cut
    # This handles the [a, b) intervals correctly
    regular_binned = pd.cut(
        ic, 
        bins=regular_edges, 
        right=False, 
        labels=regular_labels, 
        include_lowest=True
    )
    
    # Assign regular bins where they are not NaN (i.e., within regular range)
    result.loc[regular_binned.notna()] = regular_binned.loc[regular_binned.notna()]
    
    # Assign overflow bin only for values >= overflow_threshold
    overflow_mask = ic >= overflow_threshold
    result.loc[overflow_mask] = overflow_label
    
    return result

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
    by_aspect: bool = False,  # If True, calculate concordance separately for each ontology (bp, mf, cc)
    aspect_col: str = "Aspect",  # Column name for aspect/ontology
    filter_predictable_terms: bool = False,  # If True, only include GO terms from predictable_terms_path
    predictable_terms_path: Optional[str] = None,  # Path to CSV file with predictable GO terms (default: /home/FilipS/2025/metagenomic_deepfri/data/external/deepfri_predictable_terms.csv)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-protein concordance stratified by IC bins.

    Returns:
      per_protein: columns [Protein, ic_bin, nA, nB, n_intersection, n_union, concordance]
      per_bin_summary: columns [ic_bin, N_proteins, median, mean]
      statistics: columns [metric, value] with various statistics about the comparison

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

    # Check if aspect column exists and by_aspect is enabled
    has_aspect = by_aspect and aspect_col in A.columns and aspect_col in B.columns
    if by_aspect and not has_aspect:
        raise ValueError(f"by_aspect=True but aspect column '{aspect_col}' not found in both dataframes")

    # Preserve aspect column before preparation if needed
    if has_aspect:
        A_aspect = A[[protein_col, term_col, aspect_col]].copy()
        B_aspect = B[[protein_col, term_col, aspect_col]].copy()
        # Normalize aspect values to lowercase
        A_aspect[aspect_col] = A_aspect[aspect_col].astype(str).str.lower()
        B_aspect[aspect_col] = B_aspect[aspect_col].astype(str).str.lower()
        A_aspect = A_aspect.dropna(subset=[aspect_col])
        B_aspect = B_aspect.dropna(subset=[aspect_col])

    A = _prepare_method_df(A, protein_col, term_col, ic_col, score_col=score_col, score_min=score_min)
    B = _prepare_method_df(B, protein_col, term_col, ic_col, score_col=score_col, score_min=score_min)

    # Merge aspect column back if needed
    if has_aspect:
        A = A.merge(A_aspect, on=[protein_col, term_col], how="left")
        B = B.merge(B_aspect, on=[protein_col, term_col], how="left")
        A = A.dropna(subset=[aspect_col])
        B = B.dropna(subset=[aspect_col])

    # Filter to predictable terms if requested
    if filter_predictable_terms:
        # Determine path to predictable terms CSV
        if predictable_terms_path is None:
            predictable_terms_path = "/home/FilipS/2025/metagenomic_deepfri/data/external/deepfri_predictable_terms.csv"
        predictable_terms_path = Path(predictable_terms_path)
        
        if not predictable_terms_path.exists():
            raise FileNotFoundError(f"Predictable terms file not found: {predictable_terms_path}")
        
        # Load predictable terms
        predictable_df = pd.read_csv(predictable_terms_path)
        if "go_term" not in predictable_df.columns:
            raise ValueError(f"CSV file must have a 'go_term' column. Found columns: {list(predictable_df.columns)}")
        
        # Get set of predictable GO terms (normalize to strings)
        predictable_terms = set(predictable_df["go_term"].astype(str).str.strip())
        
        # Filter dataframes to only include predictable terms
        A = A[A[term_col].astype(str).isin(predictable_terms)]
        B = B[B[term_col].astype(str).isin(predictable_terms)]

    A["ic_bin"] = _assign_ic_bins(A[ic_col], ic_min=ic_min, ic_max=ic_max, bin_width=bin_width)
    B["ic_bin"] = _assign_ic_bins(B[ic_col], ic_min=ic_min, ic_max=ic_max, bin_width=bin_width)
    A = A.dropna(subset=["ic_bin"])
    B = B.dropna(subset=["ic_bin"])

    # Presence tables (protein, ic_bin, term, [aspect])
    group_cols = [protein_col, "ic_bin", term_col]
    if has_aspect:
        group_cols.append(aspect_col)
    
    A3 = A[group_cols].drop_duplicates()
    B3 = B[group_cols].drop_duplicates()

    # sizes per protein/bin/[aspect]
    groupby_cols = [protein_col, "ic_bin"]
    if has_aspect:
        groupby_cols.append(aspect_col)
    
    nA = A3.groupby(groupby_cols, as_index=False).size().rename(columns={"size": "nA"})
    nB = B3.groupby(groupby_cols, as_index=False).size().rename(columns={"size": "nB"})

    # intersection per protein/bin/[aspect]
    merge_cols = [protein_col, "ic_bin", term_col]
    if has_aspect:
        merge_cols.append(aspect_col)
    
    inter = A3.merge(B3, on=merge_cols, how="inner")
    nI = inter.groupby(groupby_cols, as_index=False).size().rename(columns={"size": "n_intersection"})

    # combine and compute union
    merge_on = [protein_col, "ic_bin"]
    if has_aspect:
        merge_on.append(aspect_col)
    
    per = nA.merge(nB, on=merge_on, how="outer").merge(
        nI, on=merge_on, how="left"
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

    # summary per bin [and aspect]
    # Filter based on require_both parameter
    if require_both:
        # Only include proteins with annotations in both datasets (nA > 0 AND nB > 0)
        per_with_data = per[(per["nA"] > 0) & (per["nB"] > 0)]
    else:
        # Include proteins with annotations in either dataset (nA > 0 OR nB > 0)
        per_with_data = per[(per["nA"] > 0) | (per["nB"] > 0)]
    
    summ_groupby = ["ic_bin"]
    if has_aspect:
        summ_groupby.append(aspect_col)
    
    summ = (
        per_with_data.groupby(summ_groupby, as_index=False)
        .agg(
            N_proteins=(protein_col, "nunique"),
            median=("concordance", "median"),
            mean=("concordance", "mean"),
        )
    )
    
    # Sort bins properly: numeric bins first, then overflow bins
    def bin_sort_key(bin_label: str) -> Tuple[int, int]:
        """Return (sort_order, numeric_value) for proper sorting"""
        if bin_label.endswith("+"):
            try:
                num = int(bin_label[:-1])
                return (1, num)  # Overflow bins come after regular bins
            except ValueError:
                return (1, 999999)  # Non-numeric overflow at very end
        else:
            try:
                num = int(bin_label)
                return (0, num)  # Regular bins come first
            except ValueError:
                return (0, 999999)  # Non-numeric regular bins at end of regular section
    
    # Create temporary sort key column
    summ["_sort_key"] = summ["ic_bin"].astype(str).map(bin_sort_key)
    
    # Sort by aspect (if present) then by bin
    sort_cols = ["_sort_key"]
    if has_aspect:
        # Sort aspects in order: bp, mf, cc
        aspect_order = {"bp": 0, "mf": 1, "cc": 2}
        summ["_aspect_sort"] = summ[aspect_col].map(lambda x: aspect_order.get(str(x).lower(), 99))
        sort_cols = ["_aspect_sort"] + sort_cols
    
    summ = summ.sort_values(sort_cols).drop(columns=[c for c in ["_sort_key", "_aspect_sort"] if c in summ.columns]).reset_index(drop=True)

    # ---- Calculate additional statistics ----
    stats_list = []
    
    # 1. Total unique proteins (at least one IC bin)
    n_unique_proteins = int(per_with_data[protein_col].nunique())
    stats_list.append(("total_unique_proteins", n_unique_proteins))
    
    # 2. Median number of GO-terms per IC bin for each method
    # Group by ic_bin and calculate median nA and nB
    median_stats_groupby = ["ic_bin"]
    if has_aspect:
        median_stats_groupby.append(aspect_col)
    
    median_per_bin = per_with_data.groupby(median_stats_groupby, as_index=False).agg(
        median_nA=("nA", "median"),
        median_nB=("nB", "median"),
    )
    
    # Add to stats: overall median across all bins (median of medians)
    overall_median_nA = float(median_per_bin["median_nA"].median())
    overall_median_nB = float(median_per_bin["median_nB"].median())
    stats_list.append(("median_go_terms_per_bin_method_A", overall_median_nA))
    stats_list.append(("median_go_terms_per_bin_method_B", overall_median_nB))
    
    # Also add per-bin medians (one row per bin)
    for _, row in median_per_bin.iterrows():
        bin_label = str(row["ic_bin"])
        if has_aspect:
            aspect_label = str(row[aspect_col])
            stats_list.append((f"median_go_terms_bin_{bin_label}_aspect_{aspect_label}_method_A", float(row["median_nA"])))
            stats_list.append((f"median_go_terms_bin_{bin_label}_aspect_{aspect_label}_method_B", float(row["median_nB"])))
        else:
            stats_list.append((f"median_go_terms_bin_{bin_label}_method_A", float(row["median_nA"])))
            stats_list.append((f"median_go_terms_bin_{bin_label}_method_B", float(row["median_nB"])))
    
    # 3. Complete containment: one method entirely contained in the other
    # This happens when n_intersection == min(nA, nB) and min(nA, nB) > 0
    min_counts = np.minimum(per_with_data["nA"], per_with_data["nB"])
    complete_containment = (per_with_data["n_intersection"] == min_counts) & (min_counts > 0)
    n_complete_containment = int(complete_containment.sum())
    total_comparisons = len(per_with_data)
    stats_list.append(("complete_containment_count", n_complete_containment))
    stats_list.append(("complete_containment_percent", (100.0 * n_complete_containment / total_comparisons) if total_comparisons > 0 else 0.0))
    
    # 4. Complete disagreement: no common GO-terms
    # This happens when n_intersection == 0 and (nA > 0 or nB > 0)
    has_annotations = (per_with_data["nA"] > 0) | (per_with_data["nB"] > 0)
    complete_disagreement = (per_with_data["n_intersection"] == 0) & has_annotations
    n_complete_disagreement = int(complete_disagreement.sum())
    stats_list.append(("complete_disagreement_count", n_complete_disagreement))
    stats_list.append(("complete_disagreement_percent", (100.0 * n_complete_disagreement / total_comparisons) if total_comparisons > 0 else 0.0))
    
    # Create statistics dataframe
    statistics_df = pd.DataFrame(stats_list, columns=["metric", "value"])
    
    # Return filtered per_with_data so boxplot matches summary statistics
    return per_with_data, summ, statistics_df


def _plot_single_boxplot(
    ax: plt.Axes,
    data: List[np.ndarray],
    bins: List[str],
    per_bin_summary: pd.DataFrame,
    theme: BoxplotTheme,
    ylabel: str,
    xlabel: str,
    aspect_label: Optional[str],
) -> None:
    """Helper function to plot a single boxplot with raw datapoints."""
    # Plot raw datapoints behind boxplot with slight jitter
    positions = np.arange(1, len(bins) + 1)
    for pos, vals in zip(positions, data):
        if len(vals) > 0:
            # Add small random jitter to x-axis to spread points
            jitter = np.random.normal(0, 0.1, size=len(vals))
            ax.scatter(
                pos + jitter,
                vals,
                s=10,
                alpha=0.3,
                color="grey",
                edgecolors="none",
                zorder=1,
            )

    # Update theme to include zorder and disable fliers (since we show raw points)
    boxplot_kwargs = theme.boxplot.copy()
    boxplot_kwargs["zorder"] = 2  # Draw boxplot on top of scatter points
    boxplot_kwargs["showfliers"] = False  # Don't show outliers since we show all points
    # Make median line red and thicker for better visibility
    if "medianprops" in boxplot_kwargs:
        boxplot_kwargs["medianprops"] = boxplot_kwargs["medianprops"].copy()
        boxplot_kwargs["medianprops"]["color"] = "red"
        boxplot_kwargs["medianprops"]["linewidth"] = 2.5
    else:
        boxplot_kwargs["medianprops"] = dict(color="red", linewidth=2.5)
    
    bp = ax.boxplot(data, labels=bins, **boxplot_kwargs)

    if aspect_label:
        ax.set_title(aspect_label, fontsize=12, fontweight="bold", pad=10)
    
    # Push xlabel further down (below counts)
    ax.set_xlabel(xlabel, labelpad=28, fontsize=11)
    ax.grid(alpha=0.3)

    ax.tick_params(axis="x", labelsize=10)
    # Extend y-axis slightly higher so medians at 1.0 are visible
    ax.set_ylim(0, 1.05)

    # ----------------------------
    # Grey counts in parentheses
    # ----------------------------
    for i, (_, row) in enumerate(per_bin_summary.iterrows(), start=1):
        ax.text(
            i,
            -0.1,  # pushed lower
            f"({int(row['N_proteins'])})",
            ha="center",
            va="top",
            rotation=30,
            fontsize=10,
            color="0.5",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def save_concordance_artifacts(
    fig: plt.Figure,
    per_protein: pd.DataFrame,
    *,
    out_prefix: str,
    out_dir: str = "plots",
) -> None:
    """
    Save:
      - SVG: <out_prefix>.svg
      - Source data (per protein): <out_prefix>__source_data.csv
    """
    _out = Path(out_dir)
    _out.mkdir(parents=True, exist_ok=True)

    # 1) Figure (vector)
    fig.savefig(_out / f"{out_prefix}.svg", format="svg", bbox_inches="tight")

    # 2) Source data (exact values underlying each box)
    _cols = [c for c in ["Protein", "ic_bin", "concordance", "IC"] if c in per_protein.columns]
    per_protein.loc[:, _cols].to_csv(
        _out / f"raw_data/{out_prefix}.csv",
        index=False,
    )
  

def plot_concordance_boxplot(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    theme: BoxplotTheme = DEFAULT_THEME,
    figsize: Tuple[float, float] = (6, 4),
    title: Optional[str] = None,
    ylabel: str = "Concordance (0–1)",
    xlabel: str = "Information content",
    aspect_col: str = "Aspect",  # Column name for aspect/ontology
) -> plt.Figure:
    """
    Boxplot of per-protein concordance per IC bin,
    with counts shown as grey numbers in parentheses below bins.
    
    If aspect_col is present in the dataframes, creates separate subplots
    for each ontology (bp, mf, cc).
    """

    # Check if aspect column exists
    has_aspect = aspect_col in per_protein.columns and aspect_col in per_bin_summary.columns
    
    if has_aspect:
        # Get unique aspects and sort them: bp, mf, cc
        aspects = sorted(per_bin_summary[aspect_col].unique(), key=lambda x: {"bp": 0, "mf": 1, "cc": 2}.get(str(x).lower(), 99))
        n_aspects = len(aspects)
        
        # Create subplots: one row, n_aspects columns
        fig, axes = plt.subplots(1, n_aspects, figsize=(figsize[0] * n_aspects, figsize[1]), sharey=False)
        if n_aspects == 1:
            axes = [axes]
        
        aspect_labels = {"bp": "BP", "mf": "MF", "cc": "CC"}
        
        for ax_idx, aspect in enumerate(aspects):
            ax = axes[ax_idx]
            
            # Filter data for this aspect
            per_protein_asp = per_protein[per_protein[aspect_col].astype(str).str.lower() == str(aspect).lower()]
            per_bin_asp = per_bin_summary[per_bin_summary[aspect_col].astype(str).str.lower() == str(aspect).lower()]
            
            # Ensure bins in correct order
            bins = per_bin_asp["ic_bin"].astype(str).tolist()
            
            data: List[np.ndarray] = []
            for b in bins:
                vals = per_protein_asp.loc[
                    per_protein_asp["ic_bin"].astype(str) == b,
                    "concordance",
                ].to_numpy()
                data.append(vals)
            
            _plot_single_boxplot(ax, data, bins, per_bin_asp, theme, ylabel, xlabel, 
                                aspect_labels.get(str(aspect).lower(), str(aspect).upper()))
            
            # Only add ylabel to first subplot
            if ax_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            else:
                ax.set_ylabel("")
    else:
        # Original single plot behavior
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Ensure bins in correct order
        bins = per_bin_summary["ic_bin"].astype(str).tolist()
        
        data: List[np.ndarray] = []
        for b in bins:
            vals = per_protein.loc[
                per_protein["ic_bin"].astype(str) == b,
                "concordance",
            ].to_numpy()
            data.append(vals)
        
        _plot_single_boxplot(ax, data, bins, per_bin_summary, theme, ylabel, xlabel, None)
        ax.set_ylabel(ylabel, fontsize=11)
        
        if title:
            ax.set_title(title, pad=18)

    # Add overall title if provided and we have multiple subplots
    if title and has_aspect:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    # Increase bottom margin so nothing overlaps
    fig.subplots_adjust(bottom=0.32)

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


def _bin_sort_key(label: str) -> Tuple[int, int]:
    """Sort key: numeric bins first, overflow (e.g. '11+') last."""
    if label.endswith("+"):
        try:
            return (1, int(label[:-1]))
        except ValueError:
            return (1, 999999)
    try:
        return (0, int(label))
    except ValueError:
        return (0, 999999)


def _ic_bin_to_interval_ticklabel(
    ic_bin_str: str,
    bin_width: float,
    *,
    display_lo_offset: float = 0.0,
) -> str:
    """
    Map ``ic_bin`` string from :func:`_assign_ic_bins` to half-open interval text,
    e.g. ``\"2\"`` → ``\"[2,3)\"`` (for *bin_width* = 1).  Overflow bins (``\"11+\"``)
    are left unchanged.

    *display_lo_offset* is subtracted from the parsed left edge before forming
    ``[lo, hi)`` (used to show the first bin as ``[0, w)`` when bins are
    1-based left-edge labels).
    """
    s = str(ic_bin_str).strip()
    if s.endswith("+"):
        return s
    try:
        lo = float(s) - float(display_lo_offset)
    except ValueError:
        return s
    hi = lo + float(bin_width)
    if bin_width == int(bin_width) and lo == int(lo) and hi == int(hi):
        return f"[{int(lo)},{int(hi)})"
    return f"[{lo},{hi})"


def plot_concordance_heatmap(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (3.5, 5),
    title: Optional[str] = None,
    ylabel: str = "IC bin",
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
    aspect_col: str = "Aspect",
    annot_fontsize: int = 8,
) -> plt.Figure:
    """
    Compact heatmap of median concordance.
    Rows = IC bins, columns = ontology aspects (BP, MF, CC).
    Each cell annotated with median value and protein count.
    """
    aspect_order = {"bp": 0, "mf": 1, "cc": 2}
    aspect_labels = {"bp": "BP", "mf": "MF", "cc": "CC"}

    summ = per_bin_summary.copy()
    summ["_aspect_lc"] = summ[aspect_col].astype(str).str.lower()

    bins_sorted = sorted(summ["ic_bin"].astype(str).unique(), key=_bin_sort_key)
    aspects_sorted = sorted(
        summ["_aspect_lc"].unique(),
        key=lambda x: aspect_order.get(x, 99),
    )

    n_bins = len(bins_sorted)
    n_aspects = len(aspects_sorted)
    median_mat = np.full((n_bins, n_aspects), np.nan)
    count_mat = np.full((n_bins, n_aspects), 0, dtype=int)

    for _, row in summ.iterrows():
        bi = bins_sorted.index(str(row["ic_bin"]))
        ai = aspects_sorted.index(row["_aspect_lc"])
        median_mat[bi, ai] = row["median"]
        count_mat[bi, ai] = int(row["N_proteins"])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        median_mat, aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax, origin="upper",
    )

    # Annotate each cell with median and count
    for i in range(n_bins):
        for j in range(n_aspects):
            val = median_mat[i, j]
            n = count_mat[i, j]
            if np.isnan(val):
                txt = "\u2014"
                color = "0.5"
            else:
                txt = f"{val:.2f}\n(n={n})"
                color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=annot_fontsize, color=color,
            )

    ax.set_xticks(np.arange(n_aspects))
    ax.set_xticklabels(
        [aspect_labels.get(a, a.upper()) for a in aspects_sorted],
        fontsize=10,
    )
    ax.set_yticks(np.arange(n_bins))
    ax.set_yticklabels(bins_sorted, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Ontology", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Median concordance", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=12, pad=8)

    # Cell gridlines
    ax.set_xticks(np.arange(n_aspects + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_bins + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return fig


def plot_concordance_faceted(
    per_protein: pd.DataFrame,
    per_bin_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (5, 8),
    title: Optional[str] = None,
    ylabel: str = "Concordance (Jaccard)",
    xlabel: str = "Information content",
    aspect_col: str = "Aspect",
    aspects: Optional[Iterable[str]] = None,
    min_n_violin: int = 30,
    violin_color: str = "#1192e8",
    violin_alpha: float = 0.50,
    strip_color: str = "grey",
    strip_alpha: float = 0.45,
    strip_size: float = 12,
    x_bin_labels_as_interval: bool = False,
    ic_bin_width: float = 1.0,
    x_bin_interval_labels_zero_origin: bool = False,
) -> plt.Figure:
    """
    Vertically stacked violin+strip panels (one row per ontology).

    Parameters
    ----------
    aspects : optional list of aspect codes to display, e.g. ``["bp"]``
        or ``["mf", "cc"]``.  ``None`` (default) shows all available
        aspects in canonical order (BP, MF, CC).

    x_bin_labels_as_interval : if True, x tick labels use half-open IC ranges
        (e.g. ``[2,3)``) instead of left-edge labels; overflow bins stay as ``N+``.
        *ic_bin_width* must match the *bin_width* passed to :func:`concordance_by_ic`
        (default 1.0).

    x_bin_interval_labels_zero_origin : if True (only with interval labels),
        subtract the smallest numeric bin edge in each panel so the first bin
        reads ``[0, ic_bin_width)``, then ``[ic_bin_width, 2*ic_bin_width)``, etc.

    If *aspects* is a one-element list (e.g. ``[\"bp\"]``), the y-axis label is
    ``{ylabel}, BP ontology`` (with the short aspect name).  Otherwise
    (including ``aspects=None``), each panel shows the aspect inside the axes,
    slightly above concordance 1.0 so it clears the violin tops.

    For bins with N >= *min_n_violin* a violin is drawn to show
    distribution shape (important for bimodal / zero-inflated concordance).
    For small-N bins only jittered strip points are shown, since a density
    estimate from < 30 observations is unreliable.

    Median is marked with a horizontal tick on every bin regardless of N.
    Sample size is annotated below each bin on every panel.
    """
    aspect_order = {"bp": 0, "mf": 1, "cc": 2}
    aspect_labels = {"bp": "BP", "mf": "MF", "cc": "CC"}

    summ = per_bin_summary.copy()
    summ["_aspect_lc"] = summ[aspect_col].astype(str).str.lower()
    per = per_protein.copy()
    per["_aspect_lc"] = per[aspect_col].astype(str).str.lower()

    if aspects is not None:
        requested = [a.lower() for a in aspects]
        aspects_sorted = sorted(requested, key=lambda x: aspect_order.get(x, 99))
        ontology_on_yaxis = len(requested) == 1
    else:
        aspects_sorted = sorted(
            summ["_aspect_lc"].unique(),
            key=lambda x: aspect_order.get(x, 99),
        )
        ontology_on_yaxis = False
    n_aspects = len(aspects_sorted)

    fig, axes = plt.subplots(
        n_aspects, 1,
        figsize=figsize,
        sharex=False,
        sharey=True,
    )
    if n_aspects == 1:
        axes = [axes]

    for ax_idx, aspect in enumerate(aspects_sorted):
        ax = axes[ax_idx]
        per_asp = per[per["_aspect_lc"] == aspect]
        summ_asp = summ[summ["_aspect_lc"] == aspect]

        bins = summ_asp["ic_bin"].astype(str).tolist()
        positions = np.arange(1, len(bins) + 1)
        data = [
            per_asp.loc[per_asp["ic_bin"].astype(str) == b, "concordance"].to_numpy()
            for b in bins
        ]

        # --- draw violins for large-N bins, strip points for all bins ---
        large_positions = [p for p, d in zip(positions, data) if len(d) >= min_n_violin]
        large_data = [d for d in data if len(d) >= min_n_violin]

        if large_data:
            vp = ax.violinplot(
                large_data,
                positions=large_positions,
                widths=0.65,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(violin_color)
                body.set_edgecolor("black")
                body.set_alpha(violin_alpha)
                body.set_linewidth(0.8)

        # Strip (jittered scatter) on every bin
        for pos, vals in zip(positions, data):
            if len(vals) > 0:
                jitter = np.random.default_rng(42).normal(0, 0.08, size=len(vals))
                ax.scatter(
                    pos + jitter, vals,
                    s=strip_size, alpha=strip_alpha,
                    color=strip_color, edgecolors="none", zorder=1,
                )

        # Median tick on every bin
        for pos, vals in zip(positions, data):
            if len(vals) > 0:
                med = float(np.nanmedian(vals))
                ax.plot(
                    [pos - 0.25, pos + 0.25], [med, med],
                    color="red", lw=2.0, zorder=3, solid_capstyle="round",
                )

        # Light gray background for small-N bins
        for pos, vals in zip(positions, data):
            if 0 < len(vals) < min_n_violin:
                ax.axvspan(pos - 0.4, pos + 0.4, color="0.93", zorder=0)

        label = aspect_labels.get(aspect, aspect.upper())

        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="y", labelsize=9)

        # y-axis: ``{ylabel}, {aspect} ontology`` when a single *aspects* entry;
        # otherwise ontology text inside the panel (data coords, just above 1.0).
        mid = n_aspects // 2
        if ontology_on_yaxis:
            ax.set_ylabel(f"{ylabel}, {label} ontology", fontsize=10)
        else:
            trans_xy = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.02, 1.035, label,
                transform=trans_xy, ha="left", va="bottom",
                fontsize=11, fontweight="bold",
            )
            if ax_idx == mid:
                ax.set_ylabel(ylabel, fontsize=10)
            else:
                ax.set_ylabel("")

        # x-axis labels and N counts on every panel (bins differ per aspect)
        ax.set_xticks(positions)
        if x_bin_labels_as_interval:
            offset = 0.0
            if x_bin_interval_labels_zero_origin:
                numeric_edges: List[float] = []
                for b in bins:
                    bs = str(b).strip()
                    if bs.endswith("+"):
                        continue
                    try:
                        numeric_edges.append(float(bs))
                    except ValueError:
                        continue
                if numeric_edges:
                    offset = min(numeric_edges)
            tick_labels = [
                _ic_bin_to_interval_ticklabel(
                    b, ic_bin_width, display_lo_offset=offset,
                )
                for b in bins
            ]
        else:
            tick_labels = bins
        ax.set_xticklabels(tick_labels, fontsize=9)
        # N counts below each bin
        for i, (_, row) in enumerate(summ_asp.iterrows(), start=1):
            ax.text(
                i, -0.09,
                f"({int(row['N_proteins'])})",
                ha="center", va="top", rotation=30,
                fontsize=8, color="0.5",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
        # xlabel only on the bottom panel to avoid clutter
        if ax_idx == n_aspects - 1:
            ax.set_xlabel(xlabel, labelpad=22, fontsize=11)
        else:
            ax.set_xlabel("")

    fig.subplots_adjust(hspace=0.42, bottom=0.1, top=0.96, left=0.14, right=0.97)
    if title:
        fig.suptitle(title, fontsize=13)

    return fig
