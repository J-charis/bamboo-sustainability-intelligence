# Methodology - dataset audit and evaluation method

## Part A. Dataset audit (Zenodo record 7024827)
All findings below were checked directly in the downloaded files (SHA-256 verified against `SHA256SUMS.txt`).

### Provenance
- Title: Bamboo biomass; creator: Masisi, Bhoke (Bern University of Applied Sciences); published 2022-08-26; **CC BY 4.0**; DOI 10.5281/zenodo.7024827.
- Files: `RAWDATA.xlsx` (raw record), `analyses.csv` (data imported to R), `R_SCRIPT.R` (authors' analysis), `library.R` (package list).
- The record description states: 60 destructive sampling plots, **one bamboo culm randomly harvested per plot**; four stem subsamples at 0, 0.25, 0.5 and 0.75 of stem length; subsamples of up to 300 g oven-dried.

### True sampling unit and sample size
- The workbook has one sheet (`raw data`, A1:AA3646) with a 3-row merged header (cells A1:A3 ... M1:O2, P1:U2, V1:AA2) and **3,643 data rows over 60 plots** (14-211 rows per plot).
- Only **60 rows** have height, age and weights; each is `Tree_No = 1` of its plot. These are the harvested culms (DBH measured before harvest, length measured on the ground). The other 3,583 rows are DBH records of other culms in the plot: not labelled samples.
- Therefore the independent labelled observations are **60 plots = 60 culms**, not 3,643. There are no repeated measurements of the same culm and no duplicated rows or (plot, tree) pairs.
- Balance: 30 *B. vulgaris* / 30 *O. abyssinica*; 30 Intensive / 30 Extensive; districts Kyela 20, Mufindi 25, Rungwe 15; six ecosystem types. Species and ecosystem/district are strongly entangled (Rungwe: *B. vulgaris* only; riverbank plots: *B. vulgaris* only).

### Units and definitions (from workbook headers)
- Component fresh weights `Weight_stem/branch/Foliage`: **kg** (total for the culm component).
- Subsample fresh and dry weights: **g**.
- DBH, height and age: **units not stated** in the workbook or record. cm, m and years are assumed from magnitudes (DBH 3-10.3; height 4.2-18; age 0.5-5) and the 5.65 m plot radius in the script. This is an assumption to be confirmed with the authors.

### Target definition
- The authors' script computes dry above-ground biomass per component as fresh weight x (dry subsample / fresh subsample), averaging four stem parts, and sums components (`Total_AGB`). This was reimplemented and **matches the script's formula to rounding (max difference ~4e-15) on all 59 computable plots**.
- "Total fresh biomass" (sum of the three fresh component weights) is valid and kg-compatible; it correlates r = 0.97 with dry biomass. **Dry biomass was chosen as the target** because it is what carbon accounting relies on and the source documents the scaling method.
- Carbon: the script multiplies biomass by 0.5. No carbon content was measured. Treated as an assumption only.

### Problems found and how they were handled
| Issue | Evidence | Handling |
|---|---|---|
| Plot 43: branch and foliage **fresh subsample weights blank** in the workbook | `analyses.csv` contains 17.0 g and 160 g there; with branch dry = 67.4 g that implies a dry:fresh ratio of 3.96 (impossible) | Dry target not computable: **excluded, not imputed** (n = 59). The CSV values were not used. |
| Plot 58: foliage dry subsample = fresh subsample (180 g = 180 g) | ratio exactly 1.0 | Physically implausible; flagged, kept in primary analysis, excluded in a sensitivity run (results barely change) |
| Elevation differs between workbook and CSV for plots 8-11 | e.g. plot 8: 580 (xlsx) vs 524 (csv) | Elevation not used as a feature |
| Empty trailing column in `analyses.csv`; blank text in a numeric column (`Fol_grW`) | the authors' script deletes column 25 and coerces to numeric | Workbook is the primary source; CSV used only for district labels and cross-check |
| Merged multi-row header | merged cell ranges listed above | Header rows skipped; canonical names applied; layout validated by `data_loader.load_raw_workbook` |
| Outliers | 1.5xIQR screening flags a few large culms | Reported, **none removed** |

### What was not verified
- DBH/height/age units (assumed). Whether the 0.5 carbon fraction is appropriate for these species (not tested; not measured).
- How the authors treated plot 43 in their published results (not examined).


## Part B. Evaluation method
- **Observation-level rule:** the 3,643 workbook rows are never treated as 3,643 independent samples. Only the 60 harvested culms (59 usable) are modelling rows; `data/processed/all_60_culms_audit.csv` keeps all 60 with their quality flags, `biomass_modeling_table.csv` the 59 used.
- **Cross-validation:** 5-fold, repeated 20 times (seeds 42 to 61). Errors are computed only on culms hidden from the model. Reported figures are averages over repeats; the 95% intervals come from bootstrapping the 59 culms.
- **Model selection:** the simplest model whose mean squared error is not distinguishable from the best (paired bootstrap, 2,000 resamples). Hyper-parameters were fixed in advance (no tuning), so model comparison is not inflated by a tuning loop.
- **Limitation of reusing cross-validation:** the same cross-validated predictions are used to choose the model and to report its error. This is mildly optimistic and not an independent test. With 59 culms a separate hold-out set would be too small to be informative.
- **Robustness checks (`evaluate.py`):** leave-one-district-out (new-site transfer), repeating everything without plot 58, and linear regression with fewer inputs.
- **Not causal:** species, ecosystem type, district and management overlap strongly (for example Rungwe contains only *B. vulgaris*), so group differences are descriptive only.

## Part C. Verification of the refactor
The earlier project (a separate folder, left untouched) reported n = 59, Linear regression, MAE 2.12, RMSE 2.75, R² 0.71. The refactored `train.py` and `evaluate.py` were run from scratch and compared with the earlier saved results: **largest difference 0** over every cross-validation metric (including spreads and bootstrap intervals), 84 robustness numbers, the model selection and the prediction interval. The new `data/processed/biomass_modeling_table.csv` has identical model columns to the old one.

Discrepancy found and fixed during the refactor: reading the processed CSV with pandas' default fast float parser changed the target column in the last binary digit, which shifted random-forest and extra-trees results in the fourth decimal (for example RMSE 3.1280 instead of 3.1272). `data_loader.load_modeling_table` now reads with `float_precision="round_trip"`, which restores identical results. The selected model and conclusions were never affected.

## Part D. Final cross-validated results
| Model | MAE (kg) | MSE (kg²) | RMSE (kg) | R² |
|---|---|---|---|---|
| Baseline (mean) | 4.16 | 27.03 | 5.20 | -0.04 |
| Linear regression | 2.12 | 7.59 | 2.75 | 0.71 |
| Allometric log-log linear | 2.19 | 10.34 | 3.21 | 0.60 |
| Random forest | 2.54 | 9.81 | 3.13 | 0.62 |
| Extra trees | 2.37 | 8.70 | 2.95 | 0.67 |
