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
    # Move parse_header outside loop for better performance
    def parse_header(h):
        parts = h.split("|")
        seq_id = parts[0]
        attrs = {
            p.split("=", 1)[0]: p.split("=", 1)[1]
            for p in parts[1:] if "=" in p
        }
        return seq_id, attrs

    records = []
    
    # Get all bin directories first for progress tracking
    bin_dirs = [
        d for d in os.listdir(pyopal_root)
        if d.startswith("identity_bin_") and os.path.isdir(os.path.join(pyopal_root, d))
    ]
    bin_dirs.sort()  # process in consistent order
    
    for bin_idx, bin_dir_name in enumerate(bin_dirs, 1):
        bin_dir = os.path.join(pyopal_root, bin_dir_name)
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
                    id1, attrs1 = parse_header(header1)
                    id2, attrs2 = parse_header(header2)

                    # We want the header that has '|query='
                    # According to your original logic, that's the one that defines (query, target).
                    # seq1 corresponds to id1, seq2 to id2
                    id_to_seq = {id1: seq1, id2: seq2}
                    
                    if "query" in attrs1:
                        attrs_q = attrs1
                        query_id = attrs1["query"]
                        target_id = id1  # target is the ID from the header that contains query=
                        # query_id might match id2 if header2 is the query sequence
                        # target_id matches id1 (the structure being searched)
                    elif "query" in attrs2:
                        attrs_q = attrs2
                        query_id = attrs2["query"]
                        target_id = id2  # target is the ID from the header that contains query=
                        # query_id might match id1 if header1 is the query sequence
                        # target_id matches id2 (the structure being searched)
                    else:
                        # No '|query=' in either header: skip
                        header1 = header2 = None
                        seq1_lines = seq2_lines = []
                        continue

                    # Map from IDs to sequences
                    # query_id should match one of the header IDs (id1 or id2)
                    # target_id is the structure ID from the header with query=
                    query_aln  = id_to_seq.get(query_id, "")
                    target_aln = id_to_seq.get(target_id, "")

                    # Retrieve identity/coverage/score from attrs_q
                    identity = float(attrs_q.get("identity", "nan"))
                    coverage = float(attrs_q.get("coverage", "nan"))
                    score    = float(attrs_q.get("score", "nan"))

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
        
        bin_records = len([r for r in records if r['idbin'] == idbin_str])
        print(f"[INFO] Parsed {bin_dir_name}: {bin_records} records")

    print(f"[INFO] Total records parsed: {len(records)}")
    return pd.DataFrame.from_records(records)


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

    # target hit structure depends on idbin - use vectorized operation
    keys = main_results_df[[idbin_col, target_col]].apply(tuple, axis=1)
    main_results_df[target_prefix] = keys.map(hit_struct_paths)

    return main_results_df

def add_prf1_columns_from_paths(
    df: pd.DataFrame,
    gc_values,
    true_cmap_col: str = "true_cmap_path",
    pred_prefix: str = "cmap_path_gc",
    out_prefix: str = "",
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

    # Pre-load all unique files into cache for better performance
    print(f"[INFO] Pre-loading contact maps into cache...")
    all_pred_paths = set()
    all_true_paths = set()
    for gc in gc_values:
        pred_path_col = f"{pred_prefix}{gc}"
        if pred_path_col in df.columns:
            all_pred_paths.update(df[pred_path_col].dropna().unique())
    if true_cmap_col in df.columns:
        all_true_paths.update(df[true_cmap_col].dropna().unique())
    
    # Load all files into cache
    for path in all_pred_paths:
        load_cached(path, pred_cache)
    for path in all_true_paths:
        load_cached(path, true_cache)
    print(f"[INFO] Loaded {len(pred_cache)} prediction maps and {len(true_cache)} ground truth maps")

    # Process each gc value
    # Convert to list of rows for faster access (avoids repeated iloc calls)
    rows_list = [row for _, row in df.iterrows()]
    n_rows = len(df)
    
    for gc_idx, gc in enumerate(gc_values, 1):
        print(f"[INFO] Computing metrics for gc={gc} ({gc_idx}/{len(gc_values)})...")
        prec_col = f"{out_prefix}prec_gc{gc}"
        rec_col  = f"{out_prefix}rec_gc{gc}"
        f1_col   = f"{out_prefix}f1_gc{gc}"

        # Use list comprehension with pre-loaded rows - much faster than df.apply()
        results = []
        for row in rows_list:
            pred_path_col = f"{pred_prefix}{gc}"
            pred_path = row[pred_path_col]
            true_path = row[true_cmap_col]

            # missing paths → NaN
            if pd.isna(pred_path) or pd.isna(true_path):
                results.append((np.nan, np.nan, np.nan))
                continue

            pred = load_cached(pred_path, pred_cache)
            true = load_cached(true_path, true_cache)

            if pred is None or true is None:
                results.append((np.nan, np.nan, np.nan))
                continue

            try:
                prec, rec, f1 = metrics_for_pair(pred, true, pid=row["query"])
                results.append((prec, rec, f1))
            except AssertionError as e:
                print(f"[WARN] {e}")
                results.append((np.nan, np.nan, np.nan))
        
        df[prec_col] = [r[0] for r in results]
        df[rec_col] = [r[1] for r in results]
        df[f1_col] = [r[2] for r in results]

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

    # Pre-load all unique files into cache for better performance
    print(f"[INFO] Pre-loading contact maps into cache...")
    all_pred_paths = set()
    all_true_paths = set()
    for gc in gc_values:
        pred_path_col = f"{pred_prefix}{gc}"
        if pred_path_col in df.columns:
            all_pred_paths.update(df[pred_path_col].dropna().unique())
    if true_cmap_col in df.columns:
        all_true_paths.update(df[true_cmap_col].dropna().unique())
    
    # Load all files into cache
    for path in all_pred_paths:
        load_cached(path, pred_cache)
    for path in all_true_paths:
        load_cached(path, true_cache)
    print(f"[INFO] Loaded {len(pred_cache)} prediction maps and {len(true_cache)} ground truth maps")

    # Process each gc value
    # Convert to list of rows for faster access (avoids repeated iloc calls)
    rows_list = [row for _, row in df.iterrows()]
    n_rows = len(df)
    
    for gc_idx, gc in enumerate(gc_values, 1):
        print(f"[INFO] Computing masked metrics for gc={gc} ({gc_idx}/{len(gc_values)})...")
        cols = [
            f"prec_gc{gc}", f"rec_gc{gc}", f"f1_gc{gc}",
            f"prec_inh_gc{gc}", f"rec_inh_gc{gc}", f"f1_inh_gc{gc}",
            f"prec_syn_gc{gc}", f"rec_syn_gc{gc}", f"f1_syn_gc{gc}",
        ]
        
        # Use list comprehension with pre-loaded rows - much faster than df.apply()
        results = []
        for row in rows_list:
            pred_path_col = f"{pred_prefix}{gc}"
            pred_path = row[pred_path_col]
            true_path = row[true_cmap_col]
            query_aln  = row[query_aln_col]
            target_aln = row[target_aln_col]

            if pd.isna(pred_path) or pd.isna(true_path) or pd.isna(query_aln) or pd.isna(target_aln):
                results.append([np.nan] * 9)
                continue

            pred = load_cached(pred_path, pred_cache)
            true = load_cached(true_path, true_cache)
            if pred is None or true is None:
                results.append([np.nan] * 9)
                continue

            try:
                # base F1 over whole map
                base_prec, base_rec, base_f1 = metrics_for_pair(pred, true, pid=row["query"])

                n = pred.shape[0]
                mask_struct, mask_syn = make_pair_masks_from_alignments(query_aln, target_aln, n=n)

                inh_prec, inh_rec, inh_f1 = masked_prf1_from_binary(pred, true, mask_struct, pid=row["query"])
                syn_prec, syn_rec, syn_f1 = masked_prf1_from_binary(pred, true, mask_syn, pid=row["query"])
                
                results.append([
                    base_prec, base_rec, base_f1,
                    inh_prec, inh_rec, inh_f1,
                    syn_prec, syn_rec, syn_f1,
                ])
            except AssertionError as e:
                print(f"[WARN] {e}")
                results.append([np.nan] * 9)
        
        for col_idx, col_name in enumerate(cols):
            df[col_name] = [r[col_idx] for r in results]

    return df
