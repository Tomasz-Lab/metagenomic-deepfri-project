import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    # import landscape data (https://figshare.com/articles/dataset/Large_protein_databases_reveal_structural_complementarity_and_functional_locality/27203073?file=54271973)
    coords = pd.read_parquet("data/external/coordinates.parquet")
    return (coords,)


@app.cell
def _(coords):
    coords
    return


@app.cell
def _(coords):
    coords["is_cluster"].value_counts()
    return


@app.cell
def _(coords):
    coords.query("origin")
    return


@app.cell
def _(coords):
    _df = coords[coords["afdb_pLDDT"].isna() | (coords["afdb_pLDDT"] >= 70)]
    _df.reset_index()["protein"].to_csv(
        "/home/FilipS/2026/landscape_mdf/data/source/hq_sample_mdf.csv",
        index=False,
        header=False,
    )
    return


@app.cell
def _(coords):
    coords.query("origin == 'AFDB dark clusters'").index.nunique()
    return


@app.cell
def _(coords):
    coords.query(
        "(origin == 'AFDB dark clusters' | origin == 'AFDB light clusters') & afdb_pLDDT > 90"
    )
    return


@app.cell
def _(coords):
    coords["origin"].value_counts()
    return


@app.cell
def _(coords):
    coords_1 = coords[["origin", "afdb_pLDDT", "is_cluster"]]
    return (coords_1,)


@app.cell
def _(coords_1):
    coords_1
    return


@app.cell
def _(coords_1):
    # mapping origins into 3 methodology buckets
    merge_map = {
        "AFDB light clusters": "AFDB",
        "AFDB dark clusters": "AFDB",
        "ESMAtlas clusters": "ESMAtlas",
        "MIP singletons": "MIP",
        "MIP clusters": "MIP",
    }
    df = coords_1.copy()
    df = df.assign(merged_origin=df["origin"].map(merge_map))
    plddt_thresh = 90.0
    afdb_df = df[
        (df["merged_origin"] == "AFDB") & (df["afdb_pLDDT"] > plddt_thresh)
    ]


    def sample_n(d, n=50000, seed=42):
        k = min(len(d), n)
        return d.sample(n=k, random_state=seed) if k > 0 else d.iloc[0:0]


    afdb_s = sample_n(afdb_df, 50_000, seed=1 * 10 + 1)
    afdb_s.to_csv(f"data/source/sample_landscape_50k.csv")
    return (afdb_s,)


@app.cell
def _(afdb_s):
    # final table of 50k proteins that go into the benchmark
    afdb_s
    return


if __name__ == "__main__":
    app.run()
