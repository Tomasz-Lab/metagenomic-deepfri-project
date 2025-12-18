import numpy as np
import os
import matplotlib.pyplot as plt
import textwrap
from glob import glob
import pandas as pd

# helpers
def f1_from_binary(pred: np.ndarray, true: np.ndarray):
    p = pred.reshape(-1)
    t = true.reshape(-1)
    tp = np.sum((p == 1) & (t == 1))
    fp = np.sum((p == 1) & (t == 0))
    fn = np.sum((p == 0) & (t == 1))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def metrics_for_pair(pred: np.ndarray, true: np.ndarray, pid: str | None = None):
    """
    Compute precision, recall, F1 for a single pair of cmaps.
    Reuses existing f1_from_binary.
    """
    if pid is not None:
        assert pred.shape == true.shape, f"Shape mismatch for {pid}: {pred.shape} vs {true.shape}"
    else:
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"

    prec, rec, f1 = f1_from_binary(pred, true)
    return prec, rec, f1
    

def compare_setting(pred_maps: dict, true_maps: dict):
    f1_scores, precisions, recalls, files = [], [], [], []
    used_ids = []       # IDs that actually contributed to F1
    missing_ids = []    # IDs present in pred_maps but not in true_maps

    for pid, pred in pred_maps.items():
        if pid not in true_maps:
            print(f"ID {pid} not found in ground-truth cmaps!")
            missing_ids.append(pid)
            continue

        true = true_maps[pid]
        prec, rec, f1 = metrics_for_pair(pred, true, pid=pid)

        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        files.append(pid + ".npy")
        used_ids.append(pid)

    return {
        "f1":    np.array(f1_scores),
        "prec":  np.array(precisions),
        "rec":   np.array(recalls),
        "files": files,
        "ids":   np.array(used_ids),
        "missing_ids": np.array(missing_ids),
    }

    
def build_generated_mask(query_alignment: str,
                         target_alignment: str,
                         generated_contacts: int) -> np.ndarray:
    """
    Returns a boolean (Lq x Lq) mask where True = contact that would have been
    generated in the `target_alignment[i] == '-'` branch, i.e. hallucinated.
    """
    assert len(query_alignment) == len(target_alignment)
    
    # length of *ungapped* query
    Lq = sum(aa != "-" for aa in query_alignment)
    mask = np.zeros((Lq, Lq), dtype=bool)

    query_index = 0  # index in ungapped query

    for qa, ta in zip(query_alignment, target_alignment):
        if qa == "-":
            # gap in query: no query residue here ⇒ no query_index++
            continue

        if ta == "-":
            # this is exactly the "generated contacts" branch
            for j in range(1, generated_contacts + 1):
                # (query_index + j, query_index)
                i1 = query_index + j
                i2 = query_index
                if 0 <= i1 < Lq:
                    mask[i1, i2] = True
                    mask[i2, i1] = True

                # (query_index - j, query_index)
                i1 = query_index - j
                if 0 <= i1 < Lq:
                    mask[i1, i2] = True
                    mask[i2, i1] = True

            query_index += 1
        else:
            # aligned residue (match or mismatch)
            query_index += 1

    return mask

def build_provenance_from_pred(pred: np.ndarray,
                               generated_mask: np.ndarray) -> np.ndarray:
    """
    provenance:
        0 = no predicted contact
        1 = inherited contact (from target structure)
        2 = generated/hallucinated contact (from gap branch)
        3 = diagonal
    """
    assert pred.shape == generated_mask.shape
    Lq = pred.shape[0]

    prov = np.zeros_like(pred, dtype=np.int32)

    # diagonal
    np.fill_diagonal(prov, 3)

    # generated predicted contacts
    gen_pred = (pred == 1) & generated_mask
    prov[gen_pred] = 2

    # inherited predicted contacts = predicted 1's that are not generated and not diagonal
    inh_pred = (pred == 1) & (~generated_mask)
    inh_pred[np.eye(Lq, dtype=bool)] = False  # remove diagonal
    prov[inh_pred] = 1

    return prov

    
def f1_from_binary_masked(pred: np.ndarray,
                          true: np.ndarray,
                          mask: np.ndarray):
    """
    Compute precision/recall/F1 only on entries where mask == True.
    """
    p = pred[mask].reshape(-1)
    t = true[mask].reshape(-1)

    tp = np.sum((p == 1) & (t == 1))
    fp = np.sum((p == 1) & (t == 0))
    fn = np.sum((p == 0) & (t == 1))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1, tp, fp, fn
    
def eval_one(pred: np.ndarray,
             true: np.ndarray,
             query_alignment: str,
             target_alignment: str,
             generated_contacts: int):
    assert pred.shape == true.shape

    Lq = pred.shape[0]
    non_diag = ~np.eye(Lq, dtype=bool)

    # 1) build generated mask from alignment (Python-only)
    gen_mask = build_generated_mask(query_alignment, target_alignment, generated_contacts)

    # 2) provenance from pred + gen_mask
    prov = build_provenance_from_pred(pred, gen_mask)

    # 3) masks
    mask_all       = non_diag
    mask_inherited = (prov == 1) & non_diag
    mask_generated = (prov == 2) & non_diag

    # 4) metrics
    prec_all, rec_all, f1_all, *_ = f1_from_binary_masked(pred, true, mask_all)
    prec_inh, rec_inh, f1_inh, *_ = f1_from_binary_masked(pred, true, mask_inherited)
    prec_gen, rec_gen, f1_gen, *_ = f1_from_binary_masked(pred, true, mask_generated)

    return {
        "prec_all": prec_all, "rec_all": rec_all, "f1_all": f1_all,
        "prec_inh": prec_inh, "rec_inh": rec_inh, "f1_inh": f1_inh,
        "prec_gen": prec_gen, "rec_gen": rec_gen, "f1_gen": f1_gen,
    }

def load_alignments_fasta(path):
    """
    Robust parser for PyOpal-style alignment FASTA.

    Expected pattern, but allowing wrapped sequences:

        >AF-K9P698-F1-model_v4|target=...
        MTPSL...
        (possibly more seq1 lines)
        >AF-A0A1J0PB53-F1-model_v4|query=...
        MTPSL...
        (possibly more seq2 lines)
        #alignment_string: MMMMM...

    Returns:
        alignments: dict
            query_id -> {
                "query_alignment":  str,
                "target_alignment": str,
                "alignment_string": str,
                "target_id":        str,
            }
    """
    alignments = {}

    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    i = 0
    n = len(lines)

    while i < n:
        # skip empty lines
        if not lines[i].strip():
            i += 1
            continue

        # first header
        if not lines[i].startswith(">"):
            # unexpected line; skip and continue
            # you can print a warning if you want:
            # print(f"Skipping unexpected line in {path}: {lines[i]!r}")
            i += 1
            continue

        header1 = lines[i].strip()
        i += 1

        # seq1: until next header or alignment_string
        seq1_parts = []
        while i < n and not lines[i].startswith(">") and not lines[i].startswith("#alignment_string"):
            if lines[i].strip():
                seq1_parts.append(lines[i].strip())
            i += 1
        seq1 = "".join(seq1_parts)

        if i >= n or not lines[i].startswith(">"):
            # malformed block; no second header
            # print(f"Warning: incomplete alignment block after {header1}")
            break

        # second header
        header2 = lines[i].strip()
        i += 1

        # seq2: until alignment_string or next header (but we expect alignment_string)
        seq2_parts = []
        while i < n and not lines[i].startswith("#alignment_string") and not lines[i].startswith(">"):
            if lines[i].strip():
                seq2_parts.append(lines[i].strip())
            i += 1
        seq2 = "".join(seq2_parts)

        if i >= n or not lines[i].startswith("#alignment_string"):
            # malformed block; no alignment_string
            # print(f"Warning: missing alignment_string for {header1} / {header2}")
            break

        aln_line = lines[i].strip()
        i += 1

        # parse alignment_string
        try:
            aln_str = aln_line.split(":", 1)[1].strip()
        except IndexError:
            # malformed alignment_string line
            # print(f"Warning: bad alignment_string line: {aln_line!r}")
            continue

        # first ID = query id, second = target id (same convention as before)
        id1 = header1[1:].split("|")[0]
        id2 = header2[1:].split("|")[0]

        alignments[id1] = {
            "query_alignment":  seq1,
            "target_alignment": seq2,
            "alignment_string": aln_str,
            "target_id":        id2,
        }

    return alignments

def plot_cmap_with_alignments(true_cmap: np.ndarray,
                              pred_cmap: np.ndarray,
                              pid: str,
                              setting_label: str,
                              query_alignment: str,
                              target_alignment: str,
                              alignment_string: str,
                              f1_value: float,
                              out_dir: str,
                              wrap_width: int = 120):
    """
    Make a figure with:
      - top: 2 subplots (GT cmap, predicted cmap)
      - bottom: query + target alignments, alignment_string, and F1 score as text
    """

    assert true_cmap.shape == pred_cmap.shape, (
        f"Shape mismatch for {pid}: {true_cmap.shape} vs {pred_cmap.shape}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # A bit taller to give room for text
    fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 4),
    sharex=True, sharey=True,
    constrained_layout=True,
)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])  # spans both columns

    vmin, vmax = 0, 1

    # --- top: contact maps ---
    im0 = ax_true.imshow(true_cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax_true.set_title("Ground truth")
    ax_true.set_xlabel("Residue index")
    ax_true.set_ylabel("Residue index")

    im1 = ax_pred.imshow(pred_cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax_pred.set_title("Predicted")
    ax_pred.set_xlabel("Residue index")
    ax_pred.set_ylabel("Residue index")

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=[ax_true, ax_pred], shrink=0.8, pad=0.02)
    cbar.set_label("Contact (0/1)")

    # --- bottom: alignments + alignment_string + F1 ---
    ax_text.axis("off")

    # wrap all three to the same width so they’re visually aligned
    q_wrapped = textwrap.fill(query_alignment,    width=wrap_width)
    t_wrapped = textwrap.fill(target_alignment,   width=wrap_width)
    a_wrapped = textwrap.fill(alignment_string,   width=wrap_width)

    text = (
        f"Query ID:  {pid}\n"
        f"Query aln:\n{q_wrapped}\n\n"
        f"Target ({setting_label}) aln:\n{t_wrapped}\n\n"
        f"Alignment string:\n{a_wrapped}\n\n"
        f"F1 (contacts): {f1_value:.3f}"
    )

    ax_text.text(
        0.01, 0.98, text,
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )

    # Keep suptitle inside the figure
    fig.suptitle(f"{pid}   [{setting_label}]", fontsize=11, y=0.99)

    out_path = os.path.join(out_dir, f"{pid}_{setting_label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_pred_cmaps_arrays(cmap_paths_by_setting):
    """
    From (idbin_str, gc) -> {pid: path}
    to (idbin_str, gc) -> {pid: np.ndarray}
    """
    pred_cmaps = {}
    for setting, pid_to_path in cmap_paths_by_setting.items():
        cmaps_for_setting = {}
        for pid, path in pid_to_path.items():
            cmaps_for_setting[pid] = np.load(path)
        pred_cmaps[setting] = cmaps_for_setting
    return pred_cmaps

def run_all_f1(pred_cmaps, ground_truth_cmaps, setting_labels, compare_setting):
    """
    pred_cmaps: (idbin_str, gc) -> {pid: cmap_array}
    ground_truth_cmaps: pid -> cmap_array
    setting_labels: (idbin_str, gc) -> nice label string
    compare_setting: function(pred_maps_dict, ground_truth_cmaps_dict) -> result
    """
    results = {}
    for setting, pred_maps in pred_cmaps.items():
        label = setting_labels.get(setting, str(setting))
        print("Processing", label)
        results[label] = compare_setting(pred_maps, ground_truth_cmaps)
    return results

def make_f1_long_df(main_results_df, gc_values,
                    idbin_col="idbin",
                    f1_prefix="f1_gc"):
    value_vars = [f"{f1_prefix}{gc}" for gc in gc_values]

    f1_long = main_results_df.melt(
        id_vars=[idbin_col],
        value_vars=value_vars,
        var_name="gc_col",
        value_name="f1",
    )

    # extract gc index from 'f1_gc0' -> 0, 'f1_gc2' -> 2
    f1_long["gc"] = f1_long["gc_col"].str.replace(f1_prefix, "", regex=False).astype(int)
    f1_long = f1_long.drop(columns=["gc_col"])

    # drop rows without F1
    f1_long = f1_long.dropna(subset=["f1"])

    return f1_long

def add_ground_truth_path_column(main_results_df, ground_truth_paths,
                                 query_col="query", prefix="gt_cmap_path"):
    """
    Add column 'gt_cmap_path' that maps main_results_df[query] -> ground truth cmap file path.
    """
    colname = prefix
    main_results_df[colname] = main_results_df[query_col].map(ground_truth_paths)
    return main_results_df


def parse_pyopal_to_df(pyopal_root):
    """
    Parse all identity_bin_* alignment files under pyopal_root into a tidy DataFrame:
      columns:
        [idbin, query, target,
         pyopal_identity, pyopal_coverage, pyopal_score,
         pyopal_alignment_string,
         pyopal_query_aln, pyopal_target_aln]

    We keep the *pair* of sequences and then select the header that contains '|query='
    to define (query, target, alignments).
    """
    records = []

    for bin_dir_name in os.listdir(pyopal_root):
        if not bin_dir_name.startswith("identity_bin_"):
            continue

        bin_dir = os.path.join(pyopal_root, bin_dir_name)
        if not os.path.isdir(bin_dir):
            continue

        idbin_str = bin_dir_name.replace("identity_bin_", "")

        fasta_path = os.path.join(bin_dir, "afdb_uniprot_v4_raw_alignments.fasta")
        if not os.path.isfile(fasta_path):
            continue

        with open(fasta_path, "r") as fh:
            header1 = None
            header2 = None
            seq1_lines = []
            seq2_lines = []

            for raw in fh:
                line = raw.strip()
                if not line:
                    continue

                # Start of a sequence header
                if line.startswith(">"):
                    header = line[1:]

                    if header1 is None:
                        header1 = header
                        seq1_lines = []
                    elif header2 is None:
                        header2 = header
                        seq2_lines = []
                    else:
                        # Unexpected third header before an alignment_string line; reset
                        header1 = header
                        header2 = None
                        seq1_lines = []
                        seq2_lines = []

                # Alignment string line ends a pair block
                elif line.startswith("#alignment_string:"):
                    align_str = line.split(":", 1)[1].strip()
                    if header1 is None or header2 is None:
                        # malformed block; skip
                        header1 = header2 = None
                        seq1_lines = seq2_lines = []
                        continue

                    seq1 = "".join(seq1_lines)
                    seq2 = "".join(seq2_lines)

                    # Parse headers into id + attrs dict
                    def parse_header(h):
                        parts = h.split("|")
                        seq_id = parts[0]
                        attrs = {
                            p.split("=", 1)[0]: p.split("=", 1)[1]
                            for p in parts[1:] if "=" in p
                        }
                        return seq_id, attrs

                    id1, attrs1 = parse_header(header1)
                    id2, attrs2 = parse_header(header2)

                    # We want the header that has '|query='
                    # According to your original logic, that's the one that defines (query, target).
                    header_q = None
                    attrs_q = None
                    seq_q_header_id = None
                    other_header = None
                    seq1_id = id1
                    seq2_id = id2

                    if "query" in attrs1:
                        header_q = header1
                        attrs_q = attrs1
                        seq_q_header_id = id1
                        other_header = (header2, attrs2, seq2)
                        this_seq = seq1
                        other_seq = seq2
                    elif "query" in attrs2:
                        header_q = header2
                        attrs_q = attrs2
                        seq_q_header_id = id2
                        other_header = (header1, attrs1, seq1)
                        this_seq = seq2
                        other_seq = seq1
                    else:
                        # No '|query=' in either header: skip
                        header1 = header2 = None
                        seq1_lines = seq2_lines = []
                        continue

                    query_id = attrs_q["query"]
                    target_id = header_q.split("|")[0]

                    # Retrieve identity/coverage/score from attrs_q
                    identity = float(attrs_q.get("identity", "nan"))
                    coverage = float(attrs_q.get("coverage", "nan"))
                    score    = float(attrs_q.get("score", "nan"))

                    # Now we must map from IDs to sequences:
                    # seq1 is for id1, seq2 is for id2
                    id_to_seq = {id1: seq1, id2: seq2}

                    query_aln  = id_to_seq.get(query_id, "")
                    target_aln = id_to_seq.get(target_id, "")

                    if not query_aln or not target_aln:
                        # If mapping failed, still record basic info but leave aln strings empty
                        print(f"[WARN] Could not map query/target alignments for {query_id} vs {target_id} in {fasta_path}")

                    records.append({
                        "idbin": idbin_str,
                        "query": query_id,
                        "target": target_id,
                        "pyopal_identity": identity,
                        "pyopal_coverage": coverage,
                        "pyopal_score": score,
                        "pyopal_alignment_string": align_str,
                        "pyopal_query_aln": query_aln,
                        "pyopal_target_aln": target_aln,
                    })

                    # reset for next block
                    header1 = header2 = None
                    seq1_lines = seq2_lines = []

                else:
                    # sequence line
                    if header1 is not None and header2 is None:
                        seq1_lines.append(line)
                    elif header2 is not None:
                        seq2_lines.append(line)

    return pd.DataFrame.from_records(records)


def add_pyopal_to_df(
    main_results_df,
    pyopal_info,
    idbin_col="idbin",
    query_col="query",
    target_col="target",
    prefix="pyopal",
):
    """
    Add columns:
        pyopal_identity, pyopal_coverage, pyopal_score, pyopal_alignment_string
    for each (idbin, query, target) pair.
    """

    def get_pyopal(row):
        key = (row[idbin_col], row[query_col], row[target_col])
        return pyopal_info.get(key)

    identities = []
    coverages = []
    scores = []
    align_strings = []

    for _, row in main_results_df.iterrows():
        info = get_pyopal(row)
        if info is None:
            identities.append(np.nan)
            coverages.append(np.nan)
            scores.append(np.nan)
            align_strings.append(None)
        else:
            identities.append(info.get("identity", np.nan))
            coverages.append(info.get("coverage", np.nan))
            scores.append(info.get("score", np.nan))
            align_strings.append(info.get("alignment_string"))

    main_results_df[f"{prefix}_identity"] = identities
    main_results_df[f"{prefix}_coverage"] = coverages
    main_results_df[f"{prefix}_score"] = scores
    main_results_df[f"{prefix}_alignment_string"] = align_strings

    return main_results_df

from glob import glob

def discover_true_structures(true_struct_root):
    """
    true_struct_root: directory with AF-...-model_v4.cif files
    Returns:
        true_struct_paths[pid] = full_path_to_cif
    where pid is like 'AF-A0A009F2M5-F1-model_v4'.
    """
    true_struct_paths = {}
    for f in glob(os.path.join(true_struct_root, "*.cif")):
        base = os.path.basename(f)
        pid, _ = os.path.splitext(base)  # drop ".cif"
        true_struct_paths[pid] = f
    return true_struct_paths

def discover_hit_structures(structures_root):
    """
    structures_root:
      .../structures
    contains:
      identity_bin_X-Y/afdb_uniprot_v4/*.cif.gz.pdb

    Returns:
      hit_struct_paths[(idbin_str, pid)] = full_path_to_pdb
    """
    hit_struct_paths = {}

    for bin_dir_name in os.listdir(structures_root):
        if not bin_dir_name.startswith("identity_bin_"):
            continue

        bin_dir = os.path.join(structures_root, bin_dir_name)
        if not os.path.isdir(bin_dir):
            continue

        idbin_str = bin_dir_name.replace("identity_bin_", "")

        afdb_dir = os.path.join(bin_dir, "afdb_uniprot_v4")
        if not os.path.isdir(afdb_dir):
            continue

        for fname in os.listdir(afdb_dir):
            if not fname.endswith(".pdb"):
                continue

            full_path = os.path.join(afdb_dir, fname)
            # strip .pdb, .gz, .cif
            base, _ = os.path.splitext(fname)          # remove .pdb -> '*.cif.gz'
            base2, _ = os.path.splitext(base)          # remove .gz  -> '*.cif'
            pid, _ = os.path.splitext(base2)           # remove .cif -> 'AF-...-model_v4'

            hit_struct_paths[(idbin_str, pid)] = full_path

    return hit_struct_paths

def add_structure_paths_to_df(
    main_results_df,
    true_struct_paths,
    hit_struct_paths,
    query_col="query",
    target_col="target",
    idbin_col="idbin",
    query_prefix="query_true_struct_path",
    target_prefix="target_hit_struct_path",
):
    # query true structure (no idbin dependence)
    main_results_df[query_prefix] = main_results_df[query_col].map(true_struct_paths)

    # target hit structure depends on idbin
    def get_hit_path(row):
        key = (row[idbin_col], row[target_col])
        return hit_struct_paths.get(key)

    main_results_df[target_prefix] = main_results_df.apply(get_hit_path, axis=1)

    return main_results_df

def add_prf1_columns_from_paths(
    df: pd.DataFrame,
    gc_values,
    true_cmap_col: str = "true_cmap_path",
    pred_prefix: str = "cmap_path_gc",
    out_prefix: str = "",  # you can set e.g. "base_" if you later add modified metrics
):
    """
    For each gc in gc_values, add columns:
        {out_prefix}prec_gc{gc}
        {out_prefix}rec_gc{gc}
        {out_prefix}f1_gc{gc}

    Uses caching so each .npy is loaded at most once.
    """
    true_cache: dict[str, np.ndarray] = {}
    pred_cache: dict[str, np.ndarray] = {}

    def load_cached(path: str, cache: dict[str, np.ndarray]) -> np.ndarray | None:
        if path is None or (isinstance(path, float) and np.isnan(path)):
            return None
        if path not in cache:
            cache[path] = np.load(path)
        return cache[path]

    def compute_row_metrics(row, gc: int):
        pred_path_col = f"{pred_prefix}{gc}"
        pred_path = row[pred_path_col]
        true_path = row[true_cmap_col]

        # missing paths → NaN
        if pd.isna(pred_path) or pd.isna(true_path):
            return np.nan, np.nan, np.nan

        pred = load_cached(pred_path, pred_cache)
        true = load_cached(true_path, true_cache)

        if pred is None or true is None:
            return np.nan, np.nan, np.nan

        try:
            prec, rec, f1 = metrics_for_pair(pred, true, pid=row["query"])
        except AssertionError as e:
            # shape mismatch or something → mark as NaN, log if you want
            print(f"[WARN] {e}")
            return np.nan, np.nan, np.nan

        return prec, rec, f1

    for gc in gc_values:
        prec_col = f"{out_prefix}prec_gc{gc}"
        rec_col  = f"{out_prefix}rec_gc{gc}"
        f1_col   = f"{out_prefix}f1_gc{gc}"

        df[[prec_col, rec_col, f1_col]] = df.apply(
            lambda row, g=gc: pd.Series(compute_row_metrics(row, g)),
            axis=1,
        )

    return df
##########################################################################################
############################## MODIFIED CMAP F1 CALCULATION ##############################
##########################################################################################
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
            continue   # no query residue here
        if t == "-":
            is_struct.append(False)  # synthetic residue
        else:
            is_struct.append(True)   # inherited residue

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



def masked_prf1_from_binary(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    pid: str | None = None,
    eps: float = 1e-8,
):
    if pid is not None:
        assert pred.shape == true.shape == mask.shape, (
            f"Shape mismatch for {pid}: pred {pred.shape}, true {true.shape}, mask {mask.shape}"
        )

    # upper triangle only, no diagonal
    tri = np.triu(mask, k=1)

    pred_sel = pred[tri]
    true_sel = true[tri]

    if pred_sel.size == 0:
        return np.nan, np.nan, np.nan

    tp = np.sum((pred_sel == 1) & (true_sel == 1))
    fp = np.sum((pred_sel == 1) & (true_sel == 0))
    fn = np.sum((pred_sel == 0) & (true_sel == 1))

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps) if (prec + rec) > 0 else 0.0

    return prec, rec, f1


def add_prf1_with_masks_from_paths(
    df: pd.DataFrame,
    gc_values,
    true_cmap_col: str = "true_cmap_path",
    pred_prefix: str = "cmap_path_gc",
    query_aln_col: str = "pyopal_query_aln",
    target_aln_col: str = "pyopal_target_aln",
):
    true_cache: dict[str, np.ndarray] = {}
    pred_cache: dict[str, np.ndarray] = {}

    def load_cached(path: str, cache: dict[str, np.ndarray]) -> np.ndarray | None:
        if path is None or (isinstance(path, float) and np.isnan(path)):
            return None
        if path not in cache:
            cache[path] = np.load(path)
        return cache[path]

    def compute_row_metrics(row, gc: int):
        pred_path_col = f"{pred_prefix}{gc}"
        pred_path = row[pred_path_col]
        true_path = row[true_cmap_col]
        query_aln  = row[query_aln_col]
        target_aln = row[target_aln_col]

        if pd.isna(pred_path) or pd.isna(true_path) or pd.isna(query_aln) or pd.isna(target_aln):
            return [np.nan] * 9

        pred = load_cached(pred_path, pred_cache)
        true = load_cached(true_path, true_cache)
        if pred is None or true is None:
            return [np.nan] * 9

        try:
            # base F1 over whole map
            base_prec, base_rec, base_f1 = metrics_for_pair(pred, true, pid=row["query"])

            n = pred.shape[0]
            mask_struct, mask_syn = make_pair_masks_from_alignments(query_aln, target_aln, n=n)

            inh_prec, inh_rec, inh_f1 = masked_prf1_from_binary(pred, true, mask_struct, pid=row["query"])
            syn_prec, syn_rec, syn_f1 = masked_prf1_from_binary(pred, true, mask_syn, pid=row["query"])
        except AssertionError as e:
            print(f"[WARN] {e}")
            return [np.nan] * 9

        return [
            base_prec, base_rec, base_f1,
            inh_prec, inh_rec, inh_f1,
            syn_prec, syn_rec, syn_f1,
        ]

    for gc in gc_values:
        cols = [
            f"prec_gc{gc}", f"rec_gc{gc}", f"f1_gc{gc}",
            f"prec_inh_gc{gc}", f"rec_inh_gc{gc}", f"f1_inh_gc{gc}",
            f"prec_syn_gc{gc}", f"rec_syn_gc{gc}", f"f1_syn_gc{gc}",
        ]
        df[cols] = df.apply(
            lambda row, g=gc: pd.Series(compute_row_metrics(row, g)),
            axis=1,
        )

    return df
