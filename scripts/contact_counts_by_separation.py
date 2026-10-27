"""Count contact-map agreement per sequence-separation band.

For every query-template pair of the landscape benchmark and every gc value,
counts true, predicted and true-positive contacts in the upper triangle,
split into separation bands |i-j| = 0 (diagonal), 1-2, 3-5, 6-11, 12-23 and
>=24. Counts are additive, so the published full-map F1 (both triangles plus
diagonal), the F1 without mirror and diagonal and the CASP short/medium/long
range F1 all follow from these columns.
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd

PAIRS = "data/generated/gc_benchmark_with_tm.csv"
OUTPUT = "data/generated/gc_benchmark_contact_counts.csv"
GC_VALUES = [0, 1, 2, 3, 4]
BAND_EDGES = [1, 3, 6, 12, 24]
BANDS = ["0", "1-2", "3-5", "6-11", "12-23", "24+"]


def band_index(length):
    rows, cols = np.triu_indices(length)
    return rows, cols, np.searchsorted(BAND_EDGES, cols - rows, side="right")


def count(values, band):
    return np.bincount(band, weights=values, minlength=len(BANDS)).astype(int)


def pair_counts(row):
    query, target, idbin, true_path, pred_paths = row
    true = np.load(true_path)
    rows, cols, band = band_index(true.shape[0])
    true_upper = true[rows, cols]
    n_pos = np.bincount(band, minlength=len(BANDS))
    n_true = count(true_upper, band)

    records = []
    for gc, pred_path in zip(GC_VALUES, pred_paths):
        if not isinstance(pred_path, str):
            continue
        pred = np.load(pred_path)
        if pred.shape != true.shape:
            continue
        pred_upper = pred[rows, cols]
        record = {
            "query": query,
            "target": target,
            "idbin": idbin,
            "gc": gc,
            "length": true.shape[0],
        }
        for name, values in [
            ("pos", n_pos),
            ("true", n_true),
            ("pred", count(pred_upper, band)),
            ("tp", count(pred_upper & true_upper, band)),
        ]:
            record.update({f"n_{name}_{b}": v for b, v in zip(BANDS, values)})
        records.append(record)
    return records


def main():
    pairs = pd.read_csv(
        PAIRS,
        usecols=["query", "target", "idbin", "true_cmap_path"]
        + [f"cmap_path_gc{gc}" for gc in GC_VALUES],
    )
    rows = [
        (
            r.query,
            r.target,
            r.idbin,
            r.true_cmap_path,
            [getattr(r, f"cmap_path_gc{gc}") for gc in GC_VALUES],
        )
        for r in pairs.itertuples(index=False)
    ]
    with Pool(8) as pool:
        records = [
            rec
            for recs in pool.imap(pair_counts, rows, chunksize=64)
            for rec in recs
        ]
    pd.DataFrame(records).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
