"""train.py - the complete model-training workflow, from the data table to the saved model.

Run:  python train.py

Steps (each one is a function or a numbered block below):
  1. Load the verified modelling table (59 culms)
  2. Pick the input features and the target
  3. Prepare the inputs (scaling numbers, encoding species)
  4. Define the candidate regression models
  5. Cross-validate every model (predict culms that were hidden during training)
  6. Calculate the errors (evaluate.py) and compare the models
  7. Choose the best model with a fixed rule
  8. Fit the chosen model on all 59 culms and save it with joblib
"""
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import data_loader as dl
import evaluate


# ===================================================================== step 3: preparing the inputs
def make_preprocessor(numeric_columns, scale=False, take_log=False):
    """Numbers can be log-transformed and/or scaled; species (text) becomes a 0/1 column."""
    number_steps = []
    if take_log:
        number_steps.append(("log", FunctionTransformer(np.log, feature_names_out="one-to-one")))
    if scale:
        number_steps.append(("scale", StandardScaler()))      # (value - mean) / spread, so no input dominates
    numbers = Pipeline(number_steps) if number_steps else "passthrough"
    return ColumnTransformer([
        ("numbers", numbers, numeric_columns),
        ("species", OneHotEncoder(drop="if_binary"), dl.CATEGORICAL_FEATURES),
    ])


# ===================================================================== step 4: candidate models
def linear_model(columns):
    """Linear regression using only the given input columns (used here and in the ablation check)."""
    numeric = [c for c in columns if c in dl.NUMERIC_FEATURES]
    parts = []
    if numeric:
        parts.append(("numbers", StandardScaler(), numeric))
    if "species" in columns:
        parts.append(("species", OneHotEncoder(drop="if_binary"), ["species"]))
    return Pipeline([("prepare", ColumnTransformer(parts)), ("model", LinearRegression())])


def build_models():
    """The five candidates, listed from simplest to most complex. No hyper-parameter tuning (only 59 culms)."""
    numbers = dl.NUMERIC_FEATURES
    return {
        # Baseline: always predicts the average biomass. Any useful model must beat this.
        "Baseline (mean)": Pipeline([("prepare", make_preprocessor(numbers)),
                                     ("model", DummyRegressor(strategy="mean"))]),

        # Linear regression: biomass = a + b1*DBH + b2*height + b3*age + b4*species. Draws the best straight-line fit.
        "Linear regression": Pipeline([("prepare", make_preprocessor(numbers, scale=True)),
                                       ("model", LinearRegression())]),

        # Allometric (log-log) regression: the same idea on logarithms, i.e. biomass = a * DBH^b ...
        # a classic form for tree/bamboo size-biomass relationships. Predictions are converted back with exp().
        "Allometric log-log linear": Pipeline([
            ("prepare", make_preprocessor(numbers, scale=True, take_log=True)),
            ("model", TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp))]),

        # Random forest: 300 decision trees, each trained on a random sample of the culms; their answers are averaged.
        "Random forest": Pipeline([("prepare", make_preprocessor(numbers)),
                                   ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                   random_state=dl.SEED))]),

        # Extra trees: like a random forest, but the split points are chosen more randomly.
        "Extra trees": Pipeline([("prepare", make_preprocessor(numbers)),
                                 ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                                               random_state=dl.SEED))]),
    }


# ===================================================================== step 5: cross-validation
def cross_validate_predictions(model, X, y, folds=dl.CV_FOLDS, repeats=dl.CV_REPEATS):
    """Predict every culm while it is hidden from the model.

    5-fold cross-validation: split the culms into 5 groups; train on 4, predict the 5th; do this 5 times
    so every culm gets one prediction made by a model that never saw it. We repeat the whole thing
    `repeats` times with different random splits. Returns an array of shape (repeats, number of culms).
    Each row of the table is a separate plot (one culm per plot), so no culm leaks between train and test.
    """
    predictions = np.empty((repeats, len(y)))
    for repeat in range(repeats):
        splitter = KFold(folds, shuffle=True, random_state=dl.SEED + repeat)
        for train_rows, test_rows in splitter.split(X):
            fitted = clone(model).fit(X.iloc[train_rows], y[train_rows])     # learn from 4 folds
            predictions[repeat, test_rows] = fitted.predict(X.iloc[test_rows])   # predict the hidden fold
    return np.clip(predictions, 0, None)                                     # biomass cannot be negative


# ===================================================================== step 7: choosing the best model
def select_model(model_names, squared_errors):
    """Selection rule (fixed before looking at results):
    choose the SIMPLEST model whose error cannot be statistically told apart from the best model's error.
    'Cannot be told apart' = the 95% bootstrap interval of the MSE difference includes 0."""
    candidates = [n for n in model_names if not n.startswith("Baseline")]
    best = min(candidates, key=lambda n: squared_errors[n].mean())
    chosen = next(n for n in candidates
                  if evaluate.paired_mse_difference(squared_errors[n], squared_errors[best])[1] <= 0)
    gap, low, high = evaluate.paired_mse_difference(squared_errors["Baseline (mean)"], squared_errors[chosen])
    return {"best_by_cv_mse": best, "selected": chosen,
            "baseline_minus_selected_mse": {"mean": gap, "ci95": [low, high]},
            "beats_baseline_reliably": bool(low > 0)}


# ===================================================================== the whole workflow
def run(save=True, repeats=dl.CV_REPEATS):
    # 1 + 2. data, inputs and target
    table = dl.load_modeling_table()
    dl.check_no_leakage(dl.FEATURES)
    X, y = table[dl.FEATURES], table[dl.TARGET].to_numpy()
    print(f"Training on {len(table)} culms; inputs: {dl.FEATURES}; target: {dl.TARGET} (kg)")

    # 5. cross-validate every candidate model
    models = build_models()
    predictions = {}
    for name, model in models.items():
        print("  cross-validating:", name)
        predictions[name] = cross_validate_predictions(model, X, y, repeats=repeats)

    # 6. errors + comparison (calculated in evaluate.py)
    results, squared_errors = evaluate.compare_models(y, predictions)
    print(pd.DataFrame(results).T[["MAE", "MSE", "RMSE", "R2"]].round(3).to_string())

    # 7. choose the model
    selection = select_model(list(models), squared_errors)
    chosen = selection["selected"]
    print("Selected model:", chosen, "| beats baseline reliably:", selection["beats_baseline_reliably"])

    # 8. fit the chosen model on ALL 59 culms (more data = better final model)
    final_model = clone(models[chosen]).fit(X, y)
    average_prediction = predictions[chosen].mean(axis=0)
    log_error = np.log(y / np.clip(average_prediction, 1e-6, None))          # y / prediction, on a log scale
    low, high = np.percentile(log_error, [10, 90])                          # -> approximate 80% range for new predictions
    card = {
        "selected_model": chosen, "selection": selection, "n_train": int(len(y)),
        "target": dl.TARGET, "target_unit": "kg dry mass per culm", "features": dl.FEATURES,
        "feature_units": {"dbh_cm": "cm (assumed)", "height_m": "m (assumed)", "age_years": "years (assumed)", "species": "category"},
        "training_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in dl.NUMERIC_FEATURES},
        "species": sorted(X["species"].unique().tolist()),
        "interval_log_ratio_q10_q90": [float(low), float(high)],
        "interval_note": "Empirical 80% range from cross-validated errors (y/yhat on a log scale). Approximate (n=59).",
        "cv": {"folds": dl.CV_FOLDS, "repeats": repeats, "seed": dl.SEED},
        "cv_results": results,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }

    if save:
        dl.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"pipeline": final_model, "card": card}, dl.MODEL_DIR / "biomass_model.joblib")
        (dl.MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        # every cross-validated prediction, so evaluate.py (and the dashboard) can recalculate the errors
        rows = []
        for repeat in range(repeats):
            part = table[["plot_no", "species", "district"]].copy()
            part.insert(0, "repeat", repeat)
            part["actual_kg"] = y
            for name in models:
                part[name] = predictions[name][repeat]
            rows.append(part)
        pd.concat(rows).to_csv(dl.MODEL_DIR / "cv_predictions.csv", index=False, encoding="utf-8")
        print("Saved models/biomass_model.joblib, model_card.json, cv_predictions.csv")
    return card


if __name__ == "__main__":
    run()
