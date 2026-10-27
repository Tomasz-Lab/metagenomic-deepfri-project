"""Per-structure features of the landscape benchmark queries and templates.

For every query (true AlphaFold model) and template (retrieved AFDB model):
length, pLDDT summary (B-factor of CA atoms) and P-SEA secondary-structure
fractions. Queries also get the relative contact order and the number of
long-range contacts (|i-j| >= 24) per residue of their true 6 A CA map.

Usage: uv run --with biotite python scripts/structure_features.py
"""

from multiprocessing import Pool

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd

PAIRS = "data/generated/gc_benchmark_with_tm.csv"
OUTPUT = "data/generated/landscape_structure_features.csv"


def load_ca(path):
    if path.endswith(".cif"):
        atoms = pdbx.get_structure(
            pdbx.CIFFile.read(path), model=1, extra_fields=["b_factor"]
        )
    else:
        atoms = pdb.PDBFile.read(path).get_structure(
            model=1, extra_fields=["b_factor"]
        )
    return atoms[(atoms.atom_name == "CA") & ~atoms.hetero]


def features(task):
    kind, name, path, cmap_path = task
    ca = load_ca(path)
    plddt = ca.b_factor
    sse = struc.annotate_sse(ca)
    record = {
        "kind": kind,
        "id": name,
        "length": len(ca),
        "mean_plddt": plddt.mean(),
        "frac_plddt_lt50": (plddt < 50).mean(),
        "frac_plddt_lt70": (plddt < 70).mean(),
        "frac_helix": (sse == struc.PSeaElement.HELIX).mean(),
        "frac_strand": (sse == struc.PSeaElement.STRAND).mean(),
    }
    if isinstance(cmap_path, str):
        rows, cols = np.nonzero(np.triu(np.load(cmap_path), k=1))
        sep = cols - rows
        length = len(ca)
        record["relative_contact_order"] = sep.mean() / length
        record["long_range_contacts_per_res"] = (sep >= 24).sum() / length
    return record


def main():
    pairs = pd.read_csv(
        PAIRS,
        usecols=[
            "query",
            "target",
            "true_cmap_path",
            "query_true_struct_path",
            "target_hit_struct_path",
        ],
    )
    queries = pairs.drop_duplicates("query").dropna(
        subset=["query_true_struct_path"]
    )
    targets = pairs.drop_duplicates("target").dropna(
        subset=["target_hit_struct_path"]
    )
    tasks = [
        ("query", r.query, r.query_true_struct_path, r.true_cmap_path)
        for r in queries.itertuples(index=False)
    ]
    tasks += [
        ("target", r.target, r.target_hit_struct_path, None)
        for r in targets.itertuples(index=False)
    ]
    with Pool(8) as pool:
        records = pool.map(features, tasks, chunksize=64)
    pd.DataFrame(records).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
