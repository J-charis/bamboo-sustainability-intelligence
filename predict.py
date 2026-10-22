"""predict.py - use the SAVED model to estimate biomass. Nothing is re-trained here.

    from predict import predict_biomass
    result = predict_biomass(dbh_cm=5.5, height_m=9.5, age_years=2, species="Bambusa vulgaris")
    result["dry_biomass_kg"]      # the estimate, in kg of dry mass for one culm

The saved file (models/biomass_model.joblib) contains the whole fitted pipeline: scaling + species encoding +
Linear Regression. We only have to hand it the 4 inputs in the SAME ORDER as in training (dl.FEATURES).
"""
import math

import joblib
import pandas as pd

import data_loader as dl

MODEL_FILE = dl.MODEL_DIR / "biomass_model.joblib"


class ModelNotAvailableError(RuntimeError):
    """The saved model is missing or unreadable. Fix: run  python train.py"""


class InvalidInputError(ValueError):
    """An input is missing, not a number, an unknown species, or outside the range the model was trained on."""
    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(problems))


def load_model(path=MODEL_FILE):
    """Step 1: load the saved model (and its 'card' with ranges, species list, etc.)."""
    if not path.exists():
        raise ModelNotAvailableError(f"Model file not found: {path}. Run: python train.py")
    try:
        bundle = joblib.load(path)
        assert "pipeline" in bundle and "card" in bundle
    except Exception as error:
        raise ModelNotAvailableError(f"Model file could not be loaded ({type(error).__name__}). Run: python train.py") from error
    return bundle


def check_inputs(dbh_cm, height_m, age_years, species, card):
    """Step 2: make sure the measurements are valid numbers inside the training range. Returns clean values."""
    problems, values = [], {}
    labels = {"dbh_cm": ("DBH", dbh_cm, "cm"), "height_m": ("Height", height_m, "m"), "age_years": ("Age", age_years, "years")}
    for key, (label, raw_value, unit) in labels.items():
        try:
            number = float(raw_value)
            if not math.isfinite(number):
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"{label} is missing or not a number.")
            continue
        low, high = card["training_ranges"][key]
        if not low <= number <= high:
            problems.append(f"{label} = {number:g} {unit} is outside the range the model was trained on ({low:g}-{high:g} {unit}).")
        values[key] = number
    if species not in card["species"]:
        problems.append(f"Species '{species}' is not supported (supported: {', '.join(card['species'])}).")
    values["species"] = species
    if problems:
        raise InvalidInputError(problems)
    return values


def predict_biomass(dbh_cm, height_m, age_years, species, bundle=None):
    """Steps 3-5: arrange the inputs in training order, predict, return kg of dry biomass for one culm."""
    bundle = bundle or load_model()
    card = bundle["card"]
    values = check_inputs(dbh_cm, height_m, age_years, species, card)
    row = pd.DataFrame([values])[dl.FEATURES]              # same columns, same order as in training
    raw_estimate = float(bundle["pipeline"].predict(row)[0])
    estimate = max(raw_estimate, 0.0)                      # a culm cannot weigh less than 0 kg
    low_factor, high_factor = card["interval_log_ratio_q10_q90"]
    interval = (estimate * math.exp(low_factor), estimate * math.exp(high_factor)) if estimate > 0 else None
    return {"dry_biomass_kg": estimate, "unit": "kg dry mass per culm", "interval_80_kg": interval,
            "clipped_to_zero": raw_estimate < 0, "inputs": values, "model": card["selected_model"], "is_estimate": True}


def carbon_estimate_kg(dry_biomass_kg, fraction=dl.CARBON_FRACTION):
    """OPTIONAL and only an assumption: carbon = 0.5 x dry biomass. The dataset did not measure carbon."""
    if dry_biomass_kg < 0 or not 0 < fraction < 1:
        raise ValueError("dry biomass must be >= 0 and the carbon fraction between 0 and 1")
    return dry_biomass_kg * fraction


if __name__ == "__main__":
    print(predict_biomass(5.5, 9.5, 2, "Bambusa vulgaris"))
