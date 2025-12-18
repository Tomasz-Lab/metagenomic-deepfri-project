import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from collections import defaultdict
import pandas as pd
import os
from functools import lru_cache
from Bio.PDB import PDBParser, MMCIFParser
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec

def plot_f1_by_identity_bin_df(
    f1_long,
    out_path="benchmark_plots/f1_scores_western_blots.png",
    figsize=None,
    colormap=plt.cm.tab10,
    save=True,
    idbin_col="idbin",
    gc_col="gc",
    f1_col="f1",
):
    """
    Plot F1 scores per identity bin and gc from a tidy DataFrame.

    f1_long: DataFrame with columns [idbin_col, gc_col, f1_col]
    """

    # group by identity bin, then we'll iterate bins
    bin_values = sorted(f1_long[idbin_col].unique())
    n_bins = len(bin_values)

    if figsize is None:
        _figsize = (4 * max(1, n_bins), 5)
    else:
        _figsize = figsize

    fig, axes = plt.subplots(1, max(1, n_bins), figsize=_figsize, sharey=True)

    if n_bins == 1:
        axes = [axes]

    for ax, idbin in zip(axes, bin_values):
        sub = f1_long[f1_long[idbin_col] == idbin]
        if sub.empty:
            ax.set_title(f"id={idbin}\n(n=0)")
            continue

        gc_values = sorted(sub[gc_col].unique())

        # colors for each gc
        try:
            cmap = colormap
            n_groups = max(1, len(gc_values))
            colors = [cmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
        except Exception:
            colors = ["C0"] * len(gc_values)

        positions = []
        xticks = []
        bin_ns = []

        pos = 1
        for i, gc in enumerate(gc_values):
            data = sub.loc[sub[gc_col] == gc, f1_col].dropna().values
            if data.size == 0:
                continue

            bin_ns.append(len(data))

            # boxplot
            ax.boxplot(
                [data],
                positions=[pos],
                widths=0.3,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": colors[i], "edgecolor": "k"},
                medianprops={"linewidth": 1.3, "color": "white"},
                whiskerprops={"color": "k"},
                capprops={"color": "k"},
            )

            # scatter jitter
            x_jitter = np.random.normal(pos, 0.04, size=len(data))
            ax.scatter(
                x_jitter,
                data,
                alpha=0.4,
                s=18,
                facecolor=colors[i],
                edgecolor="k",
                linewidth=0.2,
            )

            positions.append(pos)
            xticks.append(f"gc={gc}")
            pos += 1

        n_bin = max(bin_ns) if bin_ns else 0
        ax.set_title(f"id={idbin}\n(n={n_bin})")
        ax.set_xticks(positions)
        ax.set_xticklabels(xticks, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("F1 score")
    fig.suptitle("F1 by identity bin and generate_contacts", y=1.02)
    plt.tight_layout()

    if save and out_path:
        plt.savefig(out_path, bbox_inches="tight")

    return plt.gca()

def sample_examples_by_dense_bin(df, n_per_bin=10, seed=0):
    """
    Add a 'dense_bin' (0.01 width) based on pyopal_identity and
    sample up to n_per_bin rows per bin.
    """
    bins = np.arange(0.2, 1.01, 0.01)
    df = df.copy()
    df["dense_bin"] = pd.cut(df["pyopal_identity"], bins=bins, include_lowest=True)

    sampled_list = []
    rng = np.random.default_rng(seed)

    for dense_bin, group in df.groupby("dense_bin", observed=True):
        if group.empty:
            continue
        if len(group) <= n_per_bin:
            sampled = group
        else:
            sampled = group.sample(n_per_bin, random_state=int(rng.integers(0, 1e9)))
        sampled_list.append(sampled)

    if not sampled_list:
        return df.iloc[0:0]  # empty df

    return pd.concat(sampled_list, ignore_index=True)

def residue_mask_from_alignments(query_aln: str, target_aln: str, n: int | None = None):
    """
    Build a 1D boolean mask over query residues that matches align_contact_map:

      - iterate over alignment columns
      - skip positions where query_aln[i] == '-'
      - for each query residue:
          True  -> aligned to a target residue (target_aln[i] != '-')
                   => structural / inherited
          False -> from column where target_aln[i] == '-'
                   => synthetic (insertion in query)

    If n is given and len(mask) != n, we trim or pad with False.
    """
    assert len(query_aln) == len(target_aln), "alignment lengths differ"

    is_struct = []
    for q, t in zip(query_aln, target_aln):
        if q == "-":
            continue
        if t == "-":
            is_struct.append(False)
        else:
            is_struct.append(True)

    mask = np.array(is_struct, dtype=bool)

    if n is not None and len(mask) != n:
        print(
            f"[WARN] residue_mask_from_alignments: len(mask)={len(mask)} != n={n}. "
            "Trimming or padding with False."
        )
        if len(mask) > n:
            mask = mask[:n]
        else:
            pad = np.zeros(n - len(mask), dtype=bool)
            mask = np.concatenate([mask, pad])

    return mask

def make_pair_masks_from_alignments(query_aln: str, target_aln: str, n: int):
    """
    Build NxN masks:
      mask_struct   : both residues inherited (aligned to target)
      mask_synthetic: at least one residue is a query insertion (synthetic)
    """
    is_struct = residue_mask_from_alignments(query_aln, target_aln, n=n)
    assert len(is_struct) == n, f"len(is_struct)={len(is_struct)} != n={n}"

    mask_struct = np.outer(is_struct, is_struct)
    mask_synthetic = ~mask_struct

    return mask_struct, mask_synthetic


@lru_cache(None)
def load_cmap(path):
    return np.load(path)


# 0: bg, 1: FN, 2: FP, 3: TP, 4: diag
_conf_colors = ["white", "tab:blue", "tab:red", "tab:green", "black"]
_conf_cmap = ListedColormap(_conf_colors)
_conf_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
_conf_norm = BoundaryNorm(_conf_bounds, _conf_cmap.N)


def make_confusion_matrix_map(pred, true, region_mask, include_diag: bool = True):
    """
    Build an int-coded matrix:

       0 = outside region (background)
       1 = FN: true 1, pred 0
       2 = FP: true 0, pred 1
       3 = TP: true 1, pred 1
       4 = diagonal

    region_mask: boolean NxN mask (e.g. inherited or synthetic region)
    """
    assert pred.shape == true.shape == region_mask.shape
    n = pred.shape[0]

    pred_bin = (pred > 0)
    true_bin = (true > 0)

    diag = np.eye(n, dtype=bool)
    region_no_diag = region_mask & ~diag

    tp = pred_bin & true_bin & region_no_diag
    fp = pred_bin & ~true_bin & region_no_diag
    fn = ~pred_bin & true_bin & region_no_diag

    viz = np.zeros_like(pred, dtype=np.int8)
    viz[fn] = 1
    viz[fp] = 2
    viz[tp] = 3

    if include_diag:
        viz[diag] = 4

    return viz

@lru_cache(None)
def load_cmap(path):
    return np.load(path)


def plot_example_row_cmap_only(
    row: pd.Series,
    gc: int,
    out_dir: str,
    show=False,
):
    """
    Clean, non-overlapping visualization:
    
      ┌─────────────────────────────┬────────────────────────────┬──────┐
      │      Inherited cmap         │      Synthetic cmap        │ CBAR │
      └─────────────────────────────┴────────────────────────────┴──────┘
      │                    Alignment string                     │
      └──────────────────────────────────────────────────────────┘
    """

    os.makedirs(out_dir, exist_ok=True)

    pred_path = row.get(f"cmap_path_gc{gc}")
    true_path = row.get("true_cmap_path")

    if pd.isna(pred_path) or pd.isna(true_path):
        print(f"[WARN] Missing cmaps for {row['query']} {row['target']}")
        return

    pred = load_cmap(pred_path)
    true = load_cmap(true_path)

    if pred.shape != true.shape:
        print(f"[WARN] Shape mismatch {pred.shape} vs {true.shape}")
        return

    n = pred.shape[0]

    # build masks
    query_aln  = row["pyopal_query_aln"]
    target_aln = row["pyopal_target_aln"]
    mask_struct, mask_syn = make_pair_masks_from_alignments(query_aln, target_aln, n=n)

    viz_inh = make_confusion_matrix_map(pred, true, mask_struct)
    viz_syn = make_confusion_matrix_map(pred, true, mask_syn)

    # metadata
    tm  = row.get("tm_score", np.nan)
    idy = row.get("pyopal_identity", np.nan)
    f1_inh = row.get(f"f1_inh_gc{gc}", np.nan)
    f1_syn = row.get(f"f1_syn_gc{gc}", np.nan)

    query  = row["query"]
    target = row["target"]

    # -------------------------------
    #           FIGURE LAYOUT
    # -------------------------------
    fig = plt.figure(figsize=(12, 7))
    
    # rows: 2 → [0] contact maps, [1] alignment
    # cols: 3 → [0] inh, [1] syn, [2] colorbar
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[5.5, 1.2],     # a bit more height for alignment box
        width_ratios=[4, 4, 0.3],
        hspace=0.45,                  # increase vertical space between maps & alignment
        wspace=0.3,                   # more space between plots and colorbar
        top=0.80,                     # push everything DOWN
        bottom=0.10,                  # use bottom whitespace
        left=0.08,
        right=0.95,
    )

    ax_inh = fig.add_subplot(gs[0, 0])
    ax_syn = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_align = fig.add_subplot(gs[1, :])
    ax_align.axis("off")

    # -------------------------------
    #        PLOT CONTACT MAPS
    # -------------------------------
    im0 = ax_inh.imshow(viz_inh, cmap=_conf_cmap, norm=_conf_norm, interpolation="none")
    ax_inh.set_title(f"Inherited (gc={gc})\nF1_inh={f1_inh:.3f}")
    ax_inh.set_aspect("equal")
    ax_inh.set_xlabel("Residue index")
    ax_inh.set_ylabel("Residue index")

    im1 = ax_syn.imshow(viz_syn, cmap=_conf_cmap, norm=_conf_norm, interpolation="none")
    ax_syn.set_title(f"Synthetic (gc={gc})\nF1_syn={f1_syn:.3f}")
    ax_syn.set_aspect("equal")
    ax_syn.set_xlabel("Residue index")

    # colorbar
    cbar = fig.colorbar(im1, cax=ax_cbar)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["bg", "FN", "FP", "TP", "diag"])

    # -------------------------------
    #         ALIGNMENT STRING
    # -------------------------------
    align_str = row["pyopal_alignment_string"]
    if len(align_str) > 300:
        align_str = align_str[:300] + "..."

    ax_align.text(
        0.5, 0.5,
        f"Alignment: {align_str}",
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
    )

    # -------------------------------
    #            SUPERTITLE
    # -------------------------------
    fig.suptitle(
        f"{query} vs {target}\nPyOpal identity={idy:.3f}, TM-score={tm:.3f}",
        fontsize=12,
        y=0.98,
    )

    # -------------------------------
    #            SAVE FIGURE
    # -------------------------------
    safe_query = query.replace("/", "_")
    safe_target = target.replace("/", "_")
    fname = f"example_idx{row.name}_gc{gc}_{safe_query}_{safe_target}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200)
    
    if show:
        plt.show()
    else:
        plt.close(fig)

def make_confusion_matrix_map(pred, true, region_mask, include_diag=True):
    """
    Build an int-coded matrix for visualization:

       0 = outside region (background)
       1 = FN: true 1, pred 0
       2 = FP: true 0, pred 1
       3 = TP: true 1, pred 1
       4 = diagonal

    region_mask: boolean NxN mask (e.g. inherited or synthetic region)
    include_diag: if True, mark diagonal separately as 4
    """
    assert pred.shape == true.shape == region_mask.shape
    n = pred.shape[0]

    pred_bin = (pred > 0)
    true_bin = (true > 0)

    diag = np.eye(n, dtype=bool)

    region_no_diag = region_mask & ~diag

    tp = pred_bin & true_bin & region_no_diag
    fp = pred_bin & ~true_bin & region_no_diag
    fn = ~pred_bin & true_bin & region_no_diag

    viz = np.zeros_like(pred, dtype=np.int8)

    viz[fn] = 1
    viz[fp] = 2
    viz[tp] = 3

    if include_diag:
        viz[diag] = 4

    return viz

@lru_cache(None)
def load_ca_coords(struct_path: str) -> np.ndarray:
    """
    Load Cα coordinates from a PDB or mmCIF file.
    Returns an (N, 3) array. If parsing fails, returns an empty array.
    """
    struct_path = str(struct_path)

    if struct_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("struct", struct_path)
    except Exception as e:
        print(f"[WARN] Failed to parse structure {struct_path}: {e}")
        return np.zeros((0, 3), dtype=float)

    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() == "CA":
            coords.append(atom.get_coord())

    if not coords:
        return np.zeros((0, 3), dtype=float)

    coords = np.array(coords, dtype=float)

    # center for nicer display
    coords -= coords.mean(axis=0, keepdims=True)
    return coords

############################################################
################### Gap size benchmark ####################
############################################################
def extract_synthetic_gap_runs(query_aln: str, target_aln: str):
    """
    Return a list of lengths of continuous synthetic gaps:
        synthetic = (query != '-' and target == '-')
    """
    assert len(query_aln) == len(target_aln)
    runs = []
    current = 0

    for q, t in zip(query_aln, target_aln):
        is_syn = (q != '-') and (t == '-')
        if is_syn:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0

    # close last run
    if current > 0:
        runs.append(current)

    return runs

############################################################
################### Pymol helpers ####################
############################################################

def write_pymol_script_for_row(
    row: pd.Series,
    out_dir: str,
    save_session: bool = False,
    image_width: int = 1200,
    image_height: int = 900,
):
    """
    Generate a .pml PyMOL script for one query–target pair.

    Requires columns in the row:
      - query_true_struct_path
      - target_hit_struct_path
      - query
      - target
    """
    os.makedirs(out_dir, exist_ok=True)

    q_path = row["query_true_struct_path"]
    t_path = row["target_hit_struct_path"]

    if pd.isna(q_path) or pd.isna(t_path):
        print(f"[WARN] Missing structure path for {row['query']} vs {row['target']}")
        return

    q_path = str(q_path)
    t_path = str(t_path)

    safe_query  = row["query"].replace("/", "_")
    safe_target = row["target"].replace("/", "_")
    base_name   = f"{safe_query}__vs__{safe_target}"

    # Output files (relative to out_dir)
    png_name = f"{base_name}.png"
    pse_name = f"{base_name}.pse"

    script_lines = [
        f'load "{q_path}", query',
        f'load "{t_path}", hit',
        "",
        "hide everything",
        "show cartoon, query",
        "color cyan, query",
        "show cartoon, hit",
        "color magenta, hit",
        "",
        # optional alignment for visualization (you already have USalign for scoring)
        "align hit, query",
        "",
        "bg_color white",
        "set ray_opaque_background, 0",
        "orient",
        "",
        f"ray {image_width}, {image_height}",
        f'png "{png_name}", dpi=300',
    ]

    if save_session:
        script_lines.append(f'save "{pse_name}"')

    script_lines.append("quit")

    script_text = "\n".join(script_lines)

    script_path = os.path.join(out_dir, f"{base_name}.pml")
    with open(script_path, "w") as fh:
        fh.write(script_text)

    print(f"[INFO] Wrote PyMOL script: {script_path}")

def write_pymol_frames_script_for_row_af3(
    row: pd.Series,
    out_dir: str,
    n_frames: int = 120,
    image_width: int = 1200,
    image_height: int = 900,
):
    """
    Generate a PyMOL .pml script that:
      - loads query + hit
      - aligns hit -> query
      - colors both by AF3-style pLDDT (B-factor-based palette)
      - defines a movie:
          * first half: pLDDT coloring, spinning
          * second half: recolor query vs hit, still spinning
      - exports PNG frames with mpng
      - builds a GIF using ImageMagick
    """
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    q_path = row["query_true_struct_path"]
    t_path = row["target_hit_struct_path"]

    if pd.isna(q_path) or pd.isna(t_path):
        print(f"[WARN] Missing structure for {row['query']} vs {row['target']}")
        return

    q_path = str(q_path)
    t_path = str(t_path)

    safe_query  = row["query"].replace("/", "_")
    safe_target = row["target"].replace("/", "_")
    base_name   = f"{safe_query}__vs__{safe_target}"

    # frames will be out_dir_abs/base_name0001.png etc.
    frames_prefix = os.path.join(out_dir_abs, base_name)

    script_lines = [
        f'load "{q_path}", query',
        f'load "{t_path}", hit',
        "",
        "hide everything",
        "show cartoon, query",
        "show cartoon, hit",
        "",
        # align hit onto query
        "align hit, query",
        "orient",
        "",
        # AF3-like pLDDT coloring
        "set_color n0, [0.0, 0.325, 1.0]",       # >90
        "set_color n1, [0.0, 0.78, 1.0]",        # 71–90
        "set_color n2, [1.0, 0.92, 0.0]",        # 51–70
        "set_color n3, [0.902, 0.494, 0.133]",   # <=50",
        "",
        "color n0, (query or hit) and b > 90",
        "color n1, (query or hit) and b > 70 and b < 91",
        "color n2, (query or hit) and b > 50 and b < 71",
        "color n3, (query or hit) and b < 51",
        "",
        "bg_color white",
        "set ray_opaque_background, 0",
        "set depth_cue, off",
        "",
        f"mset 1 x{n_frames}",
        "",
        # movie definition: rotate every frame; recolor halfway
        "python",
        "from pymol import cmd",
        f"n_frames = {n_frames}",
        "angle = 360.0 / n_frames",
        "color_switch = n_frames // 2",
        "cmd.orient()",
        "for i in range(1, n_frames+1):",
        "    if i == color_switch:",
        "        # rotate AND recolor at this frame",
        "        cmd.mdo(i, "
        "f'turn y, {angle}; color magenta, query; color cyan, hit')",
        "    else:",
        "        # just rotate",
        "        cmd.mdo(i, f'turn y, {angle}')",
        "python end",
        "",
        "set ray_trace_frames, 1",
        f"viewport {image_width}, {image_height}",
        "",
        # write frames as PNGs
        f"mpng {frames_prefix}",
        "",
        # build GIF with ImageMagick; avoid ghost trails by clearing background
        f"system magick -delay 5 -dispose background -loop 0 {frames_prefix}*.png -layers Optimize {frames_prefix}.gif",
        "",
        "quit",
    ]

    script_path = os.path.join(out_dir_abs, base_name + ".pml")
    with open(script_path, "w") as fh:
        fh.write("\n".join(script_lines))

    print(f"[INFO] PyMOL AF3-frames script written: {script_path}")