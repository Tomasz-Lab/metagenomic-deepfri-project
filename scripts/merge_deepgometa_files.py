import pandas as pd
from pathlib import Path

files = {
    "BP": "/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/uhgp_50_mags/deep_go_meta/uhgp_50_mags_preds_bp.tsv",
    "MF": "/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/uhgp_50_mags/deep_go_meta/uhgp_50_mags_preds_mf.tsv",
    "CC": "/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/uhgp_50_mags/deep_go_meta/uhgp_50_mags_preds_cc.tsv",
}

dfs = []
for aspect, fn in files.items():
    df = pd.read_csv(fn, sep="\t", header=None,
                     names=["Protein", "go_term", "Score", "Annotation"])
    df["aspect"] = aspect
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

# (optional) sort / reorder columns
merged = merged[["Protein", "go_term", "aspect", "Score", "Annotation"]]

merged.to_csv("uhgp_50_mags_preds_merged.tsv", sep="\t", index=False)
