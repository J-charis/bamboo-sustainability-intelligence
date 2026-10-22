"""evaluate.py - this is where ALL prediction errors are calculated.

Run:  python evaluate.py     (after python train.py)

Two ideas to remember:
  actual    = the real dry biomass (kg) measured by the researchers for each culm
  predicted = what the model guessed for that culm WHILE THAT CULM WAS HIDDEN from the model
              (cross-validation: the culm was in the "test fold", never in the training folds)
Errors are therefore measured on culms the model has not seen - not on its own training data.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import data_loader as dl

N_BOOTSTRAP = 2000


# ===================================================================== 1. THE FOUR ERROR METRICS
def calculate_errors(actual, predicted):
    """MAE, MSE, RMSE and R2 for one set of predictions."""
    mae = mean_absolute_error(actual, predicted)    # average size of the error, in kg
    mse = mean_squared_error(actual, predicted)     # average of the squared errors, in kg^2 (big mistakes count more)
    rmse = mse ** 0.5                               # square root of MSE, back in kg
    r2 = r2_score(actual, predicted)                # 1 = perfect, 0 = no better than always guessing the average
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def summarise_repeats(actual, predictions):
    """`predictions` has one row per cross-validation repeat (shape: repeats x culms).
    We calculate the errors for every repeat and report the average (and how much it varies)."""
    per_repeat = [calculate_errors(actual, row) for row in predictions]
    summary = {}
    for name in ["MAE", "MSE", "RMSE", "R2"]:
        values = np.array([errors[name] for errors in per_repeat])
        summary[name] = float(values.mean())
        summary[name + "_sd_across_repeats"] = float(values.std(ddof=1))
    return summary


# ===================================================================== 2. HOW SURE ARE WE? (small-sample honesty)
def rmse_interval(squared_errors, seed=dl.SEED):
    """95% bootstrap interval for RMSE: re-draw the 59 culms with replacement 2000 times."""
    rng = np.random.default_rng(seed)
    rmses = [np.sqrt(squared_errors[rng.integers(0, len(squared_errors), len(squared_errors))].mean())
             for _ in range(N_BOOTSTRAP)]
    return [float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))]


def paired_mse_difference(squared_errors_a, squared_errors_b, seed=dl.SEED):
    """Is model A's MSE really different from model B's? Returns (mean difference, low, high of 95% interval).
    If the interval contains 0, the two models cannot be told apart with this little data."""
    difference = squared_errors_a - squared_errors_b
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(difference), size=(N_BOOTSTRAP, len(difference)))
    boot_means = difference[picks].mean(axis=1)
    return float(difference.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ===================================================================== 3. COMPARE ALL MODELS
def compare_models(actual, predictions_by_model):
    """predictions_by_model: {model name: array (repeats x culms)}.
    Returns the metrics table (a dict) and each model's squared error per culm (used for model selection)."""
    results, squared_errors = {}, {}
    for name, predictions in predictions_by_model.items():
        results[name] = summarise_repeats(actual, predictions)
        squared_errors[name] = ((predictions - actual) ** 2).mean(axis=0)   # per culm, averaged over repeats
        results[name]["RMSE_ci95_bootstrap_over_plots"] = rmse_interval(squared_errors[name])
    return results, squared_errors


def load_cv_predictions(path=None):
    """Read the cross-validation predictions saved by train.py -> (actual kg, {model: repeats x culms})."""
    table = pd.read_csv(path or dl.MODEL_DIR / "cv_predictions.csv", encoding="utf-8")
    table = table.sort_values(["repeat", "plot_no"])
    n_repeats, n_culms = table["repeat"].nunique(), table["plot_no"].nunique()
    actual = table["actual_kg"].to_numpy()[:n_culms]
    skip = ["repeat", "plot_no", "species", "district", "actual_kg"]
    models = {c: table[c].to_numpy().reshape(n_repeats, n_culms) for c in table.columns if c not in skip}
    return actual, models


# ===================================================================== 4. ROBUSTNESS CHECKS
# (these re-train models, so they import train.py inside the functions)
def leave_one_district_out(table):
    """Train on two districts, test on the third. Tests how well the model works at NEW sites."""
    import train
    from sklearn.base import clone
    X, y, groups = table[dl.FEATURES], table[dl.TARGET].to_numpy(), table["district"].to_numpy()
    results = {}
    for name, model in train.build_models().items():
        predicted = np.empty(len(y))
        for district in np.unique(groups):
            test = groups == district
            predicted[test] = clone(model).fit(X[~test], y[~test]).predict(X[test])
        predicted = np.clip(predicted, 0, None)                         # biomass can't be negative
        results[name] = calculate_errors(y, predicted)
        results[name]["per_district_RMSE"] = {
            d: float(np.sqrt(np.mean((y[groups == d] - predicted[groups == d]) ** 2))) for d in np.unique(groups)}
    return results


def without_flagged_culm():
    """Repeat the whole comparison without plot 58 (its foliage dry weight equals its fresh weight)."""
    import train
    table = dl.load_modeling_table(exclude_flagged=True)
    X, y = table[dl.FEATURES], table[dl.TARGET].to_numpy()
    return {name: summarise_repeats(y, train.cross_validate_predictions(model, X, y))
            for name, model in train.build_models().items()}


def which_inputs_matter(table):
    """Linear regression with fewer and fewer inputs. If the error does not get worse, the input was not needed."""
    import train
    X, y = table[dl.FEATURES], table[dl.TARGET].to_numpy()
    input_sets = {"DBH only": ["dbh_cm"],
                  "DBH + height": ["dbh_cm", "height_m"],
                  "DBH + height + species": ["dbh_cm", "height_m", "species"],
                  "DBH + height + age + species (all)": dl.FEATURES}
    return {label: summarise_repeats(y, train.cross_validate_predictions(train.linear_model(cols), X, y))
            for label, cols in input_sets.items()}


# ===================================================================== 5. RUN EVERYTHING
def main():
    actual, predictions = load_cv_predictions()
    results, _ = compare_models(actual, predictions)
    print("Cross-validated errors (held-out culms, averaged over repeats):")
    print(pd.DataFrame(results).T[["MAE", "MSE", "RMSE", "R2"]].round(3).to_string())

    table = dl.load_modeling_table()
    print("\nRobustness checks (re-training models; takes about a minute)...")
    robustness = {"logo": leave_one_district_out(table),
                  "sensitivity": without_flagged_culm(),
                  "ablation": which_inputs_matter(table)}
    print("New-district test (leave-one-district-out):")
    print(pd.DataFrame(robustness["logo"]).T[["MAE", "RMSE", "R2"]].round(3).to_string())

    output = {"n_culms": len(actual), "cv": results, **robustness}
    (dl.MODEL_DIR / "evaluation_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nSaved models/evaluation_results.json")


if __name__ == "__main__":
    main()
