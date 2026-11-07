# Appendix alternative: MRTS through the preceding month

This comparison changes the MRTS information lag from two months to one month for the fixed single-month CatBoost specification. Both versions are re-estimated on the same target months and evaluated on the main comparison's common observations. All other predictor restrictions, source histories, model settings, and the estimation seed remain fixed. The comparison includes total retail and eleven subsectors over April 2023–May 2026, excluding March 2024.

The alternative can be interpreted as an estimate of the next unreleased MSRS month, produced when the preceding month's MSRS becomes available. This is an estimate ahead of the target's publication, not necessarily an estimate of current-calendar-month activity. National data for the target month may already have been released at that date. Restricting MARTS and MRTS to the preceding month is therefore a deliberate information restriction, not a claim that those are the latest available national observations.

The calculation uses revised study snapshots. It does not reconstruct historical first-release MSRS or national retail vintages, or verify the acquisition dates of every private observation. Its operational interpretation concerns the assumed reference-month availability; the measured accuracy remains retrospective.

The [Census MSRS schedule](https://www.census.gov/retail/mrts/www/statedata/msrs_release_schedule.pdf) and [national retail release schedule](https://www.census.gov/retail/release_schedule.html) distinguish target reference months from publication dates.

## Reproduce Table 20

Using the distributed aggregate errors:

```sh
python reproduce.py mrts-lag-tables --output output/mrts_lag_tables
```

Using the authorized input files described in [Input data](INPUTS.md):

```sh
python reproduce.py mrts-lag --data-root /path/to/study-inputs --output output/mrts_lag_appendix
python reproduce.py mrts-lag-tables --results output/mrts_lag_appendix/summaries --output output/mrts_lag_tables
```

The table is `mrts_lag_comparison.tex`, accompanied by a CSV. It reports RMSE and MAE under both lags, percentage reductions, and total-retail, mean-category, pooled-category, and individual-subsector results. Positive reductions indicate lower errors with MRTS through the preceding month. All input observations and individual forecasts remain local.

`config/mrts_lag.json` fixes the comparison. Estimation first calibrates both MRTS lags at the closed March 2023 origin, then evaluates the reported months. Input checksums identify the study snapshots, and each run records its implementation identity and calibration result. No Git checkout, remote access, or push is required. No settings are chosen from evaluation accuracy.

## Reported comparison

The re-estimated two-month-lag baseline reproduces the original single-month CatBoost losses to numerical precision. Total-retail RMSE is 2.230 with MRTS through t−2 and 2.281 with MRTS through t−1; MAE is 1.616 and 1.662. Mean category RMSE is 4.613 and 4.648, respectively, and mean category MAE is 3.374 and 3.415. The more recent MRTS measure produces modestly higher aggregate errors under this fixed specification. These differences are descriptive; no significance claim is made.
