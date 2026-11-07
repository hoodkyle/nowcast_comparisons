# Methods and scoring

## Target and evaluation dates

The nowcast target is the change in a state's published year-over-year retail growth rate between adjacent calendar months. An error in this change equals the error in the corresponding growth nowcast when both use the same observed lagged growth rate. RMSE and MAE are reported in percentage points.

The main comparison evaluates April 2023–May 2026 on common observations, excluding March 2024: 1,887 total-retail observations and 20,307 across eleven subsectors. Models use Fiserv information before April 2024 and Gamma information with its observed history thereafter. These dates describe the study's information sets; they do not imply discontinuation of either service.

The replacement comparison assumes loss of access to Fiserv in April 2023 and immediate acquisition of Gamma with its observed history. It evaluates April 2023–March 2024, including March 2024, for total retail and ten subsectors (excluding electronics, 443). The common samples contain 612 total-retail and 6,002 subsector observations. A third alternative uses official information alone. All alternatives use the same single-month CatBoost specification.

The timing comparison uses the main comparison's evaluation observations and source schedule. It adds current national MARTS information, then current national MRTS information, to the early information set. The fixed single-month CatBoost specification is re-estimated for each information set. Tables report the full period and the two periods separated at April 2024.

## Forecasting specifications

`config/models.json` contains the fixed settings, feature scopes, and seeds. The 17 reported specifications comprise persistence; local lag ridge; OLS, ridge, lasso, and elastic net; two-month PCA ridge; state-and-industry factor ridge; scalar AR(1) state space; random forest; histogram gradient boosting; single-month and two-month CatBoost; two-month XGBoost; a 54-member boosted-tree median; an MLP; and a temporal transformer.

The common two-month construction retains calendar-adjacent information vectors and the shared observed estimation support. Predictor selection and scaling use estimation observations dated strictly before the origin. Persistence predicts no change in year-over-year growth. State effects and own-target lagged information enter as specified by each estimator. The appendix's additional local-ridge table relaxes only the estimation-sample restriction to require the target and its own lag, keeping the evaluation observations fixed.

The single-month tree models retain native missing-value treatment and observed availability indicators. The selected single-month CatBoost uses 400 iterations, depth 6, learning rate 0.05, leaf regularization 3, and seed 17. The boosted-tree median takes the observation-wise median of the 54 fixed CatBoost and XGBoost members; all members are required. The member settings are listed individually in the configuration file.

The two-month PCA specification estimates its dimension reduction using the available estimation sample. The state-and-industry factor specification instead uses expanding source-block factor scores, retaining factors observed at the origin and over at least 24 prior months, with a ridge penalty of 1. Its factor construction is in `src/retail_nowcast/factors.py`.

The scalar state-space model estimates an AR(1) latent state with its specified observation equation and EM settings, then filters forward using available information. Its implementation is in `src/retail_nowcast/state_space.py`. Neural architectures, loss, optimization, and stopping settings are defined in the configuration and implemented in `src/retail_nowcast/models.py`. Each neural specification is fit with seeds 0, 1, and 2; the reported main forecast averages the three predictions before evaluation. Seed tables also report the mean and sample standard deviation of individual-seed losses.

## Scoring and table creation

For each model and category, the summaries retain the observation count, sum of squared forecast errors, and sum of absolute forecast errors. RMSE is the square root of the squared-error sum divided by the count; MAE is the absolute-error sum divided by the count. Mean category losses average the category-specific RMSEs or MAEs equally. Pooled category losses combine error sums and counts first. Total retail is scored separately and is never added to the subsector pool.

Percentage reductions are computed from unrounded losses as 100 times one minus the ratio of the comparison loss to the reference loss. Positive values indicate lower forecast errors. Neural forecast averaging occurs before these calculations, not by averaging seed RMSEs.

Use the [table guide](../README.md#table-guide) to select an output. The four descriptive tables are generated from recorded study metadata rather than estimated errors. Their histories and computing environment describe the paper, not the machine or data on which a reader runs the commands.
