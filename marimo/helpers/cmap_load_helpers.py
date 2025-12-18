import numpy as np
import os
import matplotlib.pyplot as plt
import textwrap
from glob import glob
import pandas as pd

def discover_pred_cmaps(results_cmaps_dir):
    """
    Walk results_cmaps_dir and collect:
      - cmap_paths_by_pid[(pid, idbin_str, gc)] = full_path
      - gc_values = sorted set of all gc
    """
    cmap_paths_by_pid = {}
    gc_values = set()

    for idbin_dir_name in os.listdir(results_cmaps_dir):
        idbin_dir = os.path.join(results_cmaps_dir, idbin_dir_name)
        if not os.path.isdir(idbin_dir):
            continue

        # 'identity_bin_0.40-0.50' -> '0.40-0.50'
        idbin_str = idbin_dir_name.replace("identity_bin_", "")

        for gc_dir_name in os.listdir(idbin_dir):
            gc_dir = os.path.join(idbin_dir, gc_dir_name)
            if not os.path.isdir(gc_dir):
                continue

            # 'generate_contacts_2' -> 2
            gc_str = gc_dir_name.replace("generate_contacts_", "")
            try:
                gc = int(gc_str)
            except ValueError:
                continue

            contact_maps_dir = os.path.join(gc_dir, "contact_maps")
            if not os.path.isdir(contact_maps_dir):
                continue

            gc_values.add(gc)

            for cmap_file in os.listdir(contact_maps_dir):
                if not cmap_file.endswith(".npy"):
                    continue

                pid = os.path.splitext(cmap_file)[0]  # 'AF-A0A009QCT7-F1-model_v4'
                full_path = os.path.join(contact_maps_dir, cmap_file)

                cmap_paths_by_pid[(pid, idbin_str, gc)] = full_path

    return cmap_paths_by_pid, sorted(gc_values)

def add_cmap_path_columns(main_results_df, cmap_paths_by_pid, gc_values,
                          query_col="query", idbin_col="idbin",
                          prefix="cmap_path_gc"):
    def get_cmap_path(row, gc):
        key = (row[query_col], row[idbin_col], gc)
        return cmap_paths_by_pid.get(key)

    for gc in gc_values:
        colname = f"{prefix}{gc}"
        main_results_df[colname] = main_results_df.apply(
            lambda row, g=gc: get_cmap_path(row, g),
            axis=1,
        )
    return main_results_df

def load_ground_truth_cmaps(ground_truth_cmaps_dir):
    gt = {}
    for f in glob(os.path.join(ground_truth_cmaps_dir, "*.npy")):
        base = os.path.basename(f)
        name_no_npy, _ = os.path.splitext(base)
        name_no_cif, _ = os.path.splitext(name_no_npy)
        pid = name_no_cif
        gt[pid] = np.load(f)
    return gt

def metrics_for_pair(pred: np.ndarray, true: np.ndarray, pid: str | None = None):
    """
    Compute precision, recall, F1 for a single pair of cmaps.
    Reuses your existing f1_from_binary.
    """
    if pid is not None:
        assert pred.shape == true.shape, f"Shape mismatch for {pid}: {pred.shape} vs {true.shape}"
    else:
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"

    prec, rec, f1 = f1_from_binary(pred, true)
    return prec, rec, f1

def add_f1_columns(main_results_df, gc_values, ground_truth_cmaps,
                   query_col="query", idbin_col="idbin",
                   path_prefix="cmap_path_gc", f1_prefix="f1_gc"):
    def compute_row_f1(row, gc):
        path_col = f"{path_prefix}{gc}"
        path = row[path_col]
        if pd.isna(path):
            return np.nan

        pid = row[query_col]
        true = ground_truth_cmaps.get(pid)
        if true is None:
            return np.nan

        pred = np.load(path)
        _, _, f1 = metrics_for_pair(pred, true, pid=pid)
        return f1

    for gc in gc_values:
        colname = f"{f1_prefix}{gc}"
        main_results_df[colname] = main_results_df.apply(
            lambda row, g=gc: compute_row_f1(row, g),
            axis=1,
        )

    return main_results_df

def discover_ground_truth_cmap_paths(ground_truth_cmaps_dir):
    """
    Return a dict: pid -> full path to *.npy ground truth cmap file.
    """
    gt_paths = {}
    for f in glob(os.path.join(ground_truth_cmaps_dir, "*.npy")):
        base = os.path.basename(f)
        name_no_npy, _ = os.path.splitext(base)
        name_no_cif, _ = os.path.splitext(name_no_npy)
        pid = name_no_cif
        gt_paths[pid] = f
    return gt_paths