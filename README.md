Main raw data for the publication is present in the "data" directory on Zenodo. The full uncompressed data can take up to 100GB of space, the zip archive is ~5.4GB. 

Marimo notebooks used for the data analysis are uploaded to: https://github.com/Tomasz-Lab/metagenomic-deepfri-project

### REPRODUCIBILITY
After cloning the GitHub repo, please make sure that the "data" directory is present in the "metagenomic-deepfri-project" directory. To replicate the analysis you need only:

-data/source/

-data/external/

Then you can remove or archive downloaded intermediate directories if you wish to recreate them yourself:

-data/generated

-plots/raw_data

*Of course you can keep them, if you want to explore the results without doing the analysis from scratch.*

To prepare the libraries:

1. Install uv [https://docs.astral.sh/uv/].

2. From the repo root, run: $ uv sync (prepares libraries like pandas etc.)

3. To open the interactive notebooks: uv run marimo edit (runs marimo notebooks in the browser).

**To help you match plots with the underlying data and analysis notebook, the helper csv is included at:**

-data/generated/plots_reproducibility.csv

### Data files

"data/source" contains basic data used as input (e.g structures, contact maps, predicted GO terms) and is divided into three directories referencing benchmarks from the publication.

"data/generated" contains intermediate data, processed from source data.

"data/external" contains some information from auxilary sources (e.g IC table, go obo file).

"plots" directory contains picture files generated during the analysis along with data on which the plots are build (csv files in the "raw_data" subdirectory).

### Analysis files

"marimo/landscape_sample.py" is a helper notebook used to recreate drawing a sample from the protein landscape resource for the protein diversity benchmark.

"marimo/gc_data_harmonization.py" is the first notebook, used to gather and pre-process data from the protein diversity benchmark

"marimo/gc_data_analysis.py" is the second notebook, used to analyze the data from the protein diversity benchmark. 

"marimo/reference_protemes_benchmark.py" is the third notebook, used to analyze the data from the reference proteomes benchmark.

"marimo/mag__analysis.py" is the fourth notebook, used to analyze the data from the MAG benchmark.

"marimo/helpers" contain custom functions used in the notebooks above. 

### Scripts

"scripts" directory contain commands used to run metagenomic-deepFRI for all the benchmarks and some Python scripts for parsing and merging files. 
