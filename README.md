# State retail nowcasting: replication package

This package reproduces the paper's 17-specification nowcast comparison, transaction-source replacement comparison, and national-information timing comparison. It includes fixed model specifications, estimation code, aggregate forecast errors, and commands for every table in the paper and appendix.

## Get the repository

```sh
git clone https://github.com/hoodkyle/nowcast_comparisons.git
cd nowcast_comparisons
```

This repository contains the reproduction code, configuration, documentation, public state crosswalk, and aggregate summaries. The manuscript, generated `.tex` tables, restricted source data, individual forecasts, and fitted models are excluded from version control.

## Recreate the tables

Use Python 3.12 and run commands from the repository root. Install the table dependencies in a virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tables.txt
```

Generate all 21 tables, including the additional MRTS-lag appendix table. This workflow needs neither commercial data nor a GPU:

```sh
python reproduce.py tables --output output/tables
python reproduce.py mrts-lag-tables --output output/tables
```

The first command creates Tables 1–19 and 21; the second creates Table 20, the MRTS-lag comparison; each table is generated as a CSV and a LaTeX fragment. The 17 results tables are calculated from the saved error sums and observation counts in `summaries/`; the four descriptive tables use `config/descriptive_tables.json`. Numerical entries in the results tables reproduce the paper at its displayed precision. CSV files retain additional precision. To include a generated fragment in a LaTeX document, load `booktabs` and `graphicx`.

To create one table, substitute its identifier from the table below:

```sh
python reproduce.py tables --table main_full_comparison --output output/tables
```

## Re-estimate from source inputs

Obtain authorized access to the study's prepared source panels and official target file, arranged as described in [Input data](docs/INPUTS.md). Commercial observations are not distributed. The estimator starts from those prepared panels; it does not download commercial records or reconstruct them from vendor deliveries. File checksums in `config/inputs.json` identify the exact study snapshots and must match before estimation begins.

Install the estimation dependencies with `python -m pip install -r requirements-estimation.txt` and use a CUDA-capable GPU. [The recorded environment](config/environment.json) gives the versions and device used for the paper. PyTorch must have CUDA support (the study used its CUDA 13.0 build). GPU and library differences can affect floating-point results. The CPU and memory description in the computing-environment table describes the study machine, not minimum requirements.

To estimate all three comparisons plus the MRTS-lag appendix comparison and build all 21 tables from the new estimates:

```sh
python reproduce.py estimate --exercise all --data-root /path/to/study-inputs --output output/estimates
python reproduce.py tables --results output/estimates/summaries --output output/reestimated_tables
python reproduce.py mrts-lag --data-root /path/to/study-inputs --output output/mrts_lag_appendix
python reproduce.py mrts-lag-tables --results output/mrts_lag_appendix/summaries --output output/reestimated_tables
```

`--data-root` is the directory containing `data/`, not `data/` itself. Estimation first fits the fixed specifications at the closed March 2023 origin, then evaluates the reported months. It does not select specifications using evaluation accuracy. It fits only targets dated before each nowcast month, preserves calendar gaps, and never fills unavailable observations. A missing required input or forecast stops the requested comparison.

You can estimate each comparison separately:

```sh
python reproduce.py estimate --exercise main --data-root /path/to/study-inputs --output output/main
python reproduce.py tables --results output/main/summaries --table main_full_comparison --output output/main/tables

python reproduce.py estimate --exercise replacement --data-root /path/to/study-inputs --output output/replacement
python reproduce.py tables --results output/replacement/summaries --table source_replacement_accuracy --output output/replacement/tables

python reproduce.py estimate --exercise timing --data-root /path/to/study-inputs --output output/timing
python reproduce.py tables --results output/timing/summaries --table timing_accuracy --output output/timing/tables
```

For the other tables, use the same command with the corresponding identifier below and the results directory for its comparison. A table requested with `--results` requires that comparison's newly estimated summaries; it does not silently substitute saved paper results. Metadata tables require no estimation.

Estimation writes observation-level forecasts and completion records under the selected output directory, with aggregate errors under `summaries/`. Repeating an interrupted command reuses completed fits only when source inputs, specifications, and estimation code match. Keep the output directory local: it can contain restricted information. Outputs and input data are excluded from version control. A full run fits 54 tree members plus the remaining specifications at each origin and target and may take substantial time; the package does not prescribe a runtime.

## Table guide

The numbers below identify the tables in the paper, including its appendix. For every row except Table 20, use `python reproduce.py tables --table IDENTIFIER --output output/tables`. For Table 20, use `python reproduce.py mrts-lag-tables --output output/tables`. The two commands in the first example generate the complete set.

For re-estimated results, add `--results output/COMPARISON/summaries` with the output directory used for that comparison. Metadata rows need no source data or estimation.

| Paper table | Title | Identifier | Estimation comparison |
| --- | --- | --- | --- |
| 1 | Historical coverage of the target and private transaction sources | `data_coverage` | `Metadata` |
| 2 | Assumed information availability at consecutive nowcast dates | `information_timing` | `Metadata` |
| 3 | Information sets in the timing comparison | `timing_information` | `Metadata` |
| 4 | Selected nowcast specifications: state total-retail accuracy | `main_headline` | `main` |
| 5 | Nowcast RMSE by retail subsector | `main_categories` | `main` |
| 6 | Neural forecast averaging: state total-retail accuracy | `neural_seed_summary` | `main` |
| 7 | Forecast accuracy under an assumed April 2023 source replacement | `source_replacement_accuracy` | `replacement` |
| 8 | Forecast error reductions under alternative transaction-source choices | `source_replacement_gains` | `replacement` |
| 9 | Single-month CatBoost accuracy as national information becomes available | `timing_accuracy` | `timing` |
| 10 | Incremental accuracy gains from additional national information | `timing_gains` | `timing` |
| 11 | Full nowcast comparison: total retail and mean category accuracy | `main_full_comparison` | `main` |
| 12 | Common evaluation-sample coverage | `main_coverage` | `main` |
| 13 | Full nowcast comparison: pooled category accuracy | `main_pooled` | `main` |
| 14 | Full nowcast comparison: category RMSE | `main_category_rmse_detail` | `main` |
| 15 | Full nowcast comparison: category MAE | `main_category_mae_detail` | `main` |
| 16 | Neural estimation variability and forecast averaging by target | `neural_seed_detail` | `main` |
| 17 | Local lag ridge under alternative estimation-sample restrictions | `local_ridge_training_sample` | `main` |
| 18 | Source-replacement forecast accuracy by retail target | `source_replacement_categories` | `replacement` |
| 19 | Monthly total-retail forecast accuracy under alternative source choices | `source_replacement_monthly` | `replacement` |
| 20 | Sensitivity to the assumed MRTS information lag | `mrts_lag_comparison` | `mrts-lag` |
| 21 | Computing environment | `computing_environment` | `Metadata` |

## Estimation and scoring

See [Methods and scoring](docs/METHODS.md) for the target, evaluation samples, information dates, model definitions, and aggregation rules. Exact numerical settings and seeds are in `config/models.json`. `config/tables.json` maps every table to its required summaries and output filenames. The summary manifest records checksums of the distributed aggregate results.

The repository contains estimation and table-generation materials only. Commercial input data, individual predictions, fitted models, and manuscript source files are not distributed.

## Appendix alternative: MRTS through the preceding month

The additional appendix comparison changes only the MRTS lag for single-month CatBoost and re-estimates both information sets on the common main sample. See [the appendix instructions](docs/MRTS_LAG.md) for its publication-date interpretation and reproduction commands. Its Table 20 is generated separately with `python reproduce.py mrts-lag-tables`. Estimation runs locally and does not require a push or remote access.

## Check the reproduction suite

```sh
python -m unittest discover -s tests -v
```

The check copies the repository files into an isolated directory, verifies that manuscript and restricted files are excluded, and regenerates all 21 tables with only the table dependencies. Full re-estimation additionally needs the authorized input snapshots and estimation environment described above.
