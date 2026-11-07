# Input data

Place the following files under the directory supplied to `--data-root`:

```text
data/
  raw/msrs/state_retail_yy.csv
  processed/information_vintages/
    state_factor_panel_v1.csv.gz
    industry_factor_panel_v1.csv.gz
    state_factor_panel_v2.csv.gz
    industry_factor_panel_v2.csv.gz
    state_factor_panel_v3.csv.gz
    industry_factor_panel_v3.csv.gz
```

The main, source-replacement, and appendix MRTS-lag comparisons require the target file and the two `v1` panels. The timing comparison requires all seven files. The distributed `config/inputs.json` contains their SHA-256 checksums. Obtain the same prepared snapshots from the data holder with the required permissions; raw vendor deliveries alone are not sufficient inputs to this package. The package provides no commercial-data acquisition credentials.

## Official target file

`state_retail_yy.csv` is the Census MSRS wide-format year-over-year growth file. Its identifier columns are `fips`, `stateabbr`, and `naics`; monthly columns have names `yyYYYYMM`. Values are percentage changes. The parser retains suppression code `S` as missing and checks state identifiers against the included public state crosswalk.

The evaluation universe is the 50 states and the District of Columbia, with total retail excluding nonstore retailers (`TOTAL`) and subsectors 441–448 and 451–453. National rows, if present, are not state evaluation observations. The target is the calendar-month change in the published year-over-year growth rate; both adjacent monthly observations must exist.

## Prepared indicator panels

Both types of panel contain `source`, `date`, `reference_month`, `concept`, `timing_variant`, and `delta_yoy`. State panels also contain `state`. Dates identify calendar months. `date` is the assumed information month; `reference_month` identifies the month measured by the indicator. `delta_yoy` is the month-to-month change in year-over-year growth, in percentage points. The state key is the two-letter abbreviation. Concept identifiers preserve the prepared source's category definitions and aggregate measures.

Within each panel, source, concept, information month, timing variant, and (for state panels) state uniquely identify an observation. Missing observations remain missing. The files may contain additional preparation metadata; estimation uses the fields above and, when present, the availability flag for the relevant information stage.

| Panels | Private transaction reference month | MSRS | MARTS | MRTS |
| --- | --- | --- | --- | --- |
| `v1` | t | t−1 | t−1 | t−2 |
| `v2` | t | t−1 | t−1 and t | t−2 |
| `v3` | t | t−1 | t−1 and t | t−2 and t |

Official national indicators distinguish `lag1_official` (MARTS), `lag2_official` (MRTS), and `current_official` in `timing_variant`. Later panels retain lagged indicators and add current national measures. Current state MSRS is never a predictor. Availability is imposed on revised study snapshots; these panels are not a complete archive of historical releases.

Transaction sources are identified as `fiserv` and `gamma`; the latter is the paper's designation for the replacement source. Official sources are `msrs`, `marts`, and `mrts`. The estimator retains only the transaction source appropriate to the comparison, along with official information. Commercial measures and their category mapping must already be represented in these prepared panels. Aggregate spending measures are formed before growth-rate transformations; unavailable calendar-month comparisons are not interpolated.

The source-replacement comparison removes category-specific electronics (443) indicators before constructing predictors, and excludes 443 as an evaluation target. Published aggregate predictors remain intact. It compares Fiserv, Gamma with its observed history, and official information alone over the same target observations.

See [Methods and scoring](METHODS.md) for the difference between the hypothetical April 2023 replacement and the April 2024 source change in the main and timing comparisons.
