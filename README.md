# Bamboo Sustainability Intelligence
Machine-learning estimate of the **dry above-ground biomass of one bamboo culm** (kg) from its diameter, height, age and species, with a Streamlit dashboard. A first-year engineering mini project.

## 1. Problem statement
Measuring bamboo biomass normally means cutting, drying and weighing culms. Can quick, non-destructive measurements (DBH, height, age, species) predict the dry mass well enough - and how reliable is that estimate when only ~60 sampled culms exist?

## 2. Dataset and its limits
- **Source:** Masisi, B. (2022), *Bamboo biomass*, Zenodo, DOI 10.5281/zenodo.7024827, licence **CC BY 4.0**. Files are in `data/external/zenodo_7024827/` (unchanged; SHA-256 in `SHA256SUMS.txt`, details in `SOURCE.md`). To re-download: open https://zenodo.org/records/7024827 and save the files into that folder.
- **What the 3,643 rows really are:** 60 plots, **one harvested culm per plot**. Only those 60 rows have weights; the rest are diameters of other culms. So there are **60 observations, 59 usable** (plot 43 has missing subsample weights and is excluded, not guessed).
- **Limits:** two species, three districts, mostly 1-3-year-old culms; DBH/height/age units are not stated in the file (cm, m, years assumed); carbon was not measured. Full audit: `METHODOLOGY.md`.

## 3. Inputs and target
| | Name | Meaning |
|---|---|---|
| Inputs | `dbh_cm`, `height_m`, `age_years`, `species` | measured before harvesting (always in this order) |
| Target | `agb_dry_kg` | dry above-ground biomass of one culm = stem + branches + foliage, each = fresh weight x (dry / fresh) of its subsample (the dataset authors' method, reproduced) |

## 4. How the project works
```text
Research dataset (data/external)
      ↓
Data preparation (data_loader.py)  -> data/processed/biomass_modeling_table.csv (59 culms)
      ↓
Train candidate regression models (train.py)
      ↓
Cross-validation and error calculation (evaluate.py: MAE, MSE, RMSE, R²)
      ↓
Compare models and select one (train.py)
      ↓
Save trained model (models/biomass_model.joblib)
      ↓
Streamlit dashboard predicts biomass (app.py -> predict.py)
```

## 5. The models (all in `train.py`)
- **Baseline:** always predicts the average biomass. Anything useful must beat it.
- **Linear regression** (selected): fits a straight-line formula, biomass = a + b1·DBH + b2·height + b3·age + b4·species, choosing a and b's that make the squared errors smallest.
- **Allometric log-log regression:** the same idea on logarithms, i.e. biomass = a·DBH^b; a classic size-biomass form.
- **Random forest:** 300 decision trees on random samples of the culms; their answers are averaged.
- **Extra trees:** like a random forest with more random split points.

## 6. Training, errors and model selection
- **Cross-validation** (`train.py`): the 59 culms are split into 5 groups; a model trains on 4 and predicts the 5th, which it has never seen; repeated 5 times so every culm is predicted once. The whole thing is repeated 20 times with different splits. Each row is a separate plot, so no culm appears in both training and test.
- **Errors** (`evaluate.py`), where `actual` is the measured biomass and `predicted` is the hidden-culm prediction:
  - **MAE** = mean(|actual − predicted|): average error in kg.
  - **MSE** = mean((actual − predicted)²): squared kg; big mistakes count more.
  - **RMSE** = √MSE: back in kg.
  - **R²** = 1 − (sum of squared model errors / sum of squared errors of always guessing the average). 1 is perfect; 0 is no better than the average.
- **Selection rule** (fixed beforehand): choose the *simplest* model whose error cannot be statistically told apart from the best model's (bootstrap over culms). With so little data a fancy model is not preferred unless it is clearly better.
- **Honest limit:** the same cross-validation is used to compare models and to report their errors, so the reported numbers are slightly optimistic and are **not** an independent final test. 59 culms are too few to set aside a separate test set. The leave-one-district-out test below is a harder, more independent check.

## 7. Verified results (from the saved runs)
5-fold cross-validation x 20 repeats, 59 culms:

| Model | MAE (kg) | MSE (kg²) | RMSE (kg) | R² |
|---|---|---|---|---|
| Baseline (mean) | 4.16 | 27.03 | 5.20 | -0.04 |
| Linear regression | 2.12 | 7.59 | 2.75 | 0.71 |
| Allometric log-log linear | 2.19 | 10.34 | 3.21 | 0.60 |
| Random forest | 2.54 | 9.81 | 3.13 | 0.62 |
| Extra trees | 2.37 | 8.70 | 2.95 | 0.67 |

- **Selected: Linear regression** - CV **MAE 2.12 kg, RMSE 2.75 kg (95% interval 2.18-3.37), R² 0.71**; the average-guess baseline has RMSE 5.20 kg (interval 4.09-6.31). It beats the baseline reliably: **True**.
- **New districts** (train on two districts, test on the third):

| Model | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|
| Baseline (mean) | 4.79 | 5.85 | -0.32 |
| Linear regression | 2.93 | 3.61 | 0.50 |
| Allometric log-log linear | 3.15 | 4.15 | 0.34 |
| Random forest | 3.27 | 3.91 | 0.41 |
| Extra trees | 3.40 | 4.09 | 0.36 |

  The error on unseen sites is clearly larger (RMSE 3.61 kg for the selected model).
- **Which inputs matter:**

| Inputs used (linear regression) | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|
| DBH only | 2.54 | 3.21 | 0.60 |
| DBH + height | 2.45 | 3.21 | 0.60 |
| DBH + height + species | 2.09 | 2.73 | 0.71 |
| DBH + height + age + species (all) | 2.12 | 2.75 | 0.71 |

  DBH carries most of the signal; species helps; height and age add nothing measurable here.
- **Without the flagged plot 58**, the selected model's CV RMSE is 2.77 kg (almost unchanged).
- **These match the earlier reported results** (MAE 2.12, RMSE 2.75, R² 0.71, 59 culms). The refactored code reproduces the earlier run exactly (largest difference 0 across all metrics); see `METHODOLOGY.md`.

## 8. How the dashboard uses the model
`app.py` never trains. `predict.py` loads `models/biomass_model.joblib` once (it contains scaling + species encoding + the fitted model), checks the inputs are valid and inside the training range, puts them in the training order, and returns the estimate in kg. Invalid inputs or a missing model give an error message, never a made-up number. Sections: Overview, Biomass Predictor, Data Insights, About & Methodology. The optional carbon figure is `0.5 x dry biomass`, an assumption by the dataset authors, not a measurement.

## 9. Run it (Windows PowerShell, from this folder)
```powershell
py -3 -m venv .venv                      # once
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data_loader.py                    # (optional) rebuild data\processed from the raw Excel file
python train.py                          # train, cross-validate, select, save models\
python evaluate.py                       # error table + robustness checks -> models\evaluation_results.json
python -m pytest -q                      # tests
python -m streamlit run app.py           # dashboard at http://localhost:8501
```

## 10. Files
| File | Job |
|---|---|
| `data_loader.py` | settings; reads the Excel file; computes the target; writes `data/processed/` |
| `train.py` | **the whole training workflow** (models, cross-validation, selection, saving) |
| `evaluate.py` | **where all errors are calculated** + robustness checks |
| `predict.py` | loads the saved model and predicts |
| `app.py` | Streamlit dashboard |
| `tests/test_project.py` | essential tests |
| `METHODOLOGY.md` | dataset audit and evaluation method |
| `models/` | saved model, model card, cross-validated predictions, evaluation results |

## 11. Viva guide - where to look
- *How was the model trained?* `train.py`, read top to bottom (steps 1-8, comments on every model).
- *Where are MAE / RMSE / R² calculated?* `evaluate.py`, function `calculate_errors` (the first function).
- *Why not just test on the training data?* `cross_validate_predictions` in `train.py`: each culm is predicted by a model that never saw it.
- *How is the best model chosen?* `select_model` in `train.py`.
- *How does the app predict?* `predict_biomass` in `predict.py`, called from `run_prediction` in `app.py`.
- *Why only 59 culms out of 3,643 rows?* `METHODOLOGY.md`, section "True sampling unit".
- *What can't this model do?* It is not a carbon, sequestration or stand-level estimate; it works only for two species and the measured size ranges, and is less accurate at new sites.

## 12. Reference
Masisi, B. (2022) *Bamboo biomass*, Zenodo, doi:10.5281/zenodo.7024827 (CC BY 4.0). Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, JMLR 12:2825-2830.
