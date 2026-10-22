"""Essential tests for the Bamboo Sustainability Intelligence project.  Run:  python -m pytest -q"""
import hashlib
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_loader as dl
import evaluate
import predict
import train


@pytest.fixture(scope="module")
def culms():
    return dl.build_culm_table(dl.load_raw_workbook())


@pytest.fixture(scope="module")
def usable(culms):
    return culms[culms["usable_for_target"]].reset_index(drop=True)


@pytest.fixture(scope="module")
def bundle():
    return predict.load_model()          # the real saved model


# ---------------------------------------------------------------- 1. data
def test_original_zenodo_files_are_unchanged():
    for line in (dl.RAW_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256((dl.RAW_DIR / name).read_bytes()).hexdigest() == digest, name

def test_only_60_of_3643_rows_are_labelled_culms(culms):
    raw = dl.load_raw_workbook()
    assert len(raw) == 3643 and raw["plot_no"].nunique() == 60
    assert len(culms) == 60 and culms["plot_no"].is_unique

def test_wrong_workbook_layout_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "RAWDATA.xlsx"
    pd.DataFrame(np.zeros((5, 27))).to_excel(bad, header=False, index=False)
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path)
    with pytest.raises(ValueError):
        dl.load_raw_workbook()

def test_dry_biomass_formula_matches_hand_calculation(culms):
    plot1 = culms[culms["plot_no"] == 1].iloc[0]
    stem = 14.7 * np.mean([165 / 310, 181.7 / 270, 154.1 / 230, 87.6 / 150])    # kg x (dry/fresh)
    expected = stem + 7.1 * (49.9 / 110) + 3.83 * (68.8 / 160)
    assert plot1[dl.TARGET] == pytest.approx(expected, rel=1e-9)
    assert 0 < plot1[dl.TARGET] < plot1["fresh_total_kg"]                       # dry mass < fresh mass

def test_incomplete_plot_is_excluded_not_imputed_and_odd_plot_is_flagged(culms, usable):
    assert np.isnan(culms[culms["plot_no"] == 43][dl.TARGET].iloc[0]) and 43 not in set(usable["plot_no"])
    assert culms[culms["plot_no"] == 58]["flag_dry_ge_fresh"].iloc[0] and 58 in set(usable["plot_no"])
    assert len(usable) == 59 and not usable[dl.FEATURES + [dl.TARGET]].isna().any().any()

def test_processed_file_matches_a_fresh_rebuild(usable):
    saved = dl.load_modeling_table()
    assert len(saved) == 59 and "eastings" not in saved.columns            # coordinates are not shipped
    assert np.array_equal(saved[dl.TARGET].to_numpy(), usable[dl.TARGET].to_numpy())

def test_leakage_guard():
    dl.check_no_leakage(dl.FEATURES)
    for leaked in ["agb_dry_kg", "wt_stem_kg", "stem_dry_kg", "dry_s0_g", "ratio_br", "plot_no", "fresh_total_kg"]:
        with pytest.raises(ValueError):
            dl.check_no_leakage(dl.FEATURES + [leaked])


# ---------------------------------------------------------------- 2. errors (evaluate.py)
def test_error_metrics_against_hand_calculation():
    actual, guess = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    e = evaluate.calculate_errors(actual, guess)
    assert e["MAE"] == pytest.approx(2 / 3) and e["MSE"] == pytest.approx(4 / 3)
    assert e["RMSE"] == pytest.approx((4 / 3) ** 0.5) and e["R2"] == pytest.approx(-1.0)     # 1 - 4/2
    assert evaluate.calculate_errors(actual, actual)["R2"] == 1.0

def test_cross_validation_never_predicts_a_row_it_trained_on():
    from sklearn.base import BaseEstimator, RegressorMixin
    class Memoriser(BaseEstimator, RegressorMixin):          # answers NaN if it has seen the row in training
        def fit(self, X, y): self.seen_ = {tuple(r) for r in X.to_numpy()}; return self
        def predict(self, X): return np.array([np.nan if tuple(r) in self.seen_ else 1.0 for r in X.to_numpy()])
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)[::-1]})
    out = train.cross_validate_predictions(Memoriser(), X, np.arange(40.0), repeats=3)
    assert out.shape == (3, 40) and np.isfinite(out).all()

def test_model_selection_rule():
    rng = np.random.default_rng(0); base = rng.uniform(1, 2, 50)
    noise = rng.normal(size=50); noise -= noise.mean()
    tie = {"Baseline (mean)": base + 5, "Linear regression": base, "Random forest": base + 0.3 * noise}
    assert train.select_model(list(tie), tie)["selected"] == "Linear regression"           # indistinguishable -> simpler
    clear = {"Baseline (mean)": base + 5, "Linear regression": base + 1, "Random forest": base}
    assert train.select_model(list(clear), clear)["selected"] == "Random forest"           # clearly better -> complex
    useless = {"Baseline (mean)": base, "Linear regression": base}
    assert train.select_model(list(useless), useless)["beats_baseline_reliably"] is False


# ---------------------------------------------------------------- 3. training and the reported results
def test_training_runs_and_beats_baseline_without_touching_saved_files():
    before = (dl.MODEL_DIR / "biomass_model.joblib").stat().st_mtime
    card = train.run(save=False, repeats=2)
    cv = card["cv_results"]
    assert list(cv) == ["Baseline (mean)", "Linear regression", "Allometric log-log linear", "Random forest", "Extra trees"]
    assert card["selected_model"] != "Baseline (mean)" and cv[card["selected_model"]]["RMSE"] < cv["Baseline (mean)"]["RMSE"]
    assert (dl.MODEL_DIR / "biomass_model.joblib").stat().st_mtime == before

def test_saved_results_match_the_reported_ones():
    card = pd.read_json(dl.MODEL_DIR / "model_card.json", typ="series")
    best = card["cv_results"][card["selected_model"]]
    assert card["n_train"] == 59 and card["selected_model"] == "Linear regression"
    assert round(best["MAE"], 2) == 2.12 and round(best["RMSE"], 2) == 2.75 and round(best["R2"], 2) == 0.71

def test_evaluate_recalculates_the_same_numbers_from_saved_predictions():
    actual, predictions = evaluate.load_cv_predictions()
    results, _ = evaluate.compare_models(actual, predictions)
    card = pd.read_json(dl.MODEL_DIR / "model_card.json", typ="series")["cv_results"]
    for name in card:
        for metric in ("MAE", "RMSE", "R2"):
            assert results[name][metric] == pytest.approx(card[name][metric], abs=1e-9)


# ---------------------------------------------------------------- 4. prediction
def test_saved_model_loads_and_expects_the_training_feature_order(bundle):
    assert list(bundle["pipeline"].feature_names_in_) == dl.FEATURES

def test_valid_prediction_has_units_and_sensible_behaviour(bundle):
    r = predict.predict_biomass(5.5, 9.5, 2, "Bambusa vulgaris", bundle)
    assert r["unit"] == "kg dry mass per culm" and r["is_estimate"] and r["dry_biomass_kg"] > 0
    low, high = r["interval_80_kg"]; assert low < r["dry_biomass_kg"] < high
    thin = predict.predict_biomass(4, 9.5, 2, "Bambusa vulgaris", bundle)["dry_biomass_kg"]
    thick = predict.predict_biomass(9, 9.5, 2, "Bambusa vulgaris", bundle)["dry_biomass_kg"]
    assert thick > thin                                                   # thicker culms weigh more

@pytest.mark.parametrize("args", [(50, 9, 2, "Bambusa vulgaris"), (5, 100, 2, "Bambusa vulgaris"), (5, 9, 40, "Bambusa vulgaris"),
                                  (-1, 9, 2, "Bambusa vulgaris"), (None, 9, 2, "Bambusa vulgaris"), (float("nan"), 9, 2, "Bambusa vulgaris"),
                                  ("abc", 9, 2, "Bambusa vulgaris"), (5, 9, 2, "Giant timber bamboo")])
def test_invalid_inputs_are_rejected(bundle, args):
    with pytest.raises(predict.InvalidInputError) as error:
        predict.predict_biomass(*args, bundle=bundle)
    assert error.value.problems

def test_missing_or_corrupt_model_never_gives_a_number(tmp_path):
    with pytest.raises(predict.ModelNotAvailableError):
        predict.load_model(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.joblib"; bad.write_bytes(b"not a model")
    with pytest.raises(predict.ModelNotAvailableError):
        predict.load_model(bad)

def test_model_survives_save_and_load(bundle, tmp_path):
    joblib.dump(bundle, tmp_path / "copy.joblib"); copy = predict.load_model(tmp_path / "copy.joblib")
    a = predict.predict_biomass(6, 10, 2, "Oxytenanthera abyssinica", bundle)["dry_biomass_kg"]
    assert predict.predict_biomass(6, 10, 2, "Oxytenanthera abyssinica", copy)["dry_biomass_kg"] == pytest.approx(a)

def test_carbon_is_a_labelled_assumption_with_checks():
    assert predict.carbon_estimate_kg(10) == pytest.approx(5.0) and predict.carbon_estimate_kg(10, 0.47) == pytest.approx(4.7)
    for bad in (lambda: predict.carbon_estimate_kg(-1), lambda: predict.carbon_estimate_kg(1, 0), lambda: predict.carbon_estimate_kg(1, 1.5)):
        with pytest.raises(ValueError):
            bad()


# ---------------------------------------------------------------- 5. dashboard
def start_app(page=None):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_data.clear(); st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90); at.run()
    if page:
        at.radio(key="page").set_value(page); at.run()
    return at

def test_dashboard_has_four_sections_and_shows_the_verified_kpis():
    at = start_app()
    assert list(at.radio(key="page").options) == ["Overview", "Biomass Predictor", "Data Insights", "About & Methodology"]
    assert [m.value for m in at.metric][:4] == ["59", "Linear regression", "2.12 kg", "0.71"]
    for page in at.radio(key="page").options:
        at.radio(key="page").set_value(page); at.run()
        assert not at.exception and not at.error, page

def test_dashboard_prediction_reset_and_input_refusal():
    at = start_app("Biomass Predictor")
    at.button(key="predict_btn").click(); at.run()
    text = " ".join(m.value for m in at.markdown)
    assert "kg per culm" in text and "not a physical measurement" in text
    expected = predict.predict_biomass(5.5, 9.5, 2.0, at.selectbox(key="species").value)["dry_biomass_kg"]
    assert f"{expected:.2f} <span" in text                                  # the page shows what predict.py returns
    at.number_input(key="dbh").set_value(7.0); at.run()
    assert "Inputs changed" in " ".join(m.value for m in at.markdown)
    at.button(key="reset_btn").click(); at.run()
    assert at.number_input(key="dbh").value == 5.5
    at.number_input(key="dbh").set_value(50.0); at.button(key="predict_btn").click(); at.run()
    assert at.error and "kg per culm" not in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------- 6. project hygiene
def test_no_references_to_removed_modules_and_no_training_in_the_app():
    for path in ROOT.glob("*.py"):
        code = path.read_text(encoding="utf-8")
        assert "bamboo_sustainability" not in code and "bamboo_color" not in code and "biomass_cli" not in code, path.name
    app_code = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "import train" not in app_code and ".fit(" not in app_code     # the dashboard never trains

def test_project_files_needed_to_run_exist():
    for name in ["app.py", "train.py", "evaluate.py", "predict.py", "data_loader.py", "requirements.txt", "README.md", ".gitignore",
                 "METHODOLOGY.md", "models/biomass_model.joblib", "data/processed/biomass_modeling_table.csv",
                 "data/external/zenodo_7024827/RAWDATA.xlsx", "data/external/zenodo_7024827/SOURCE.md"]:
        assert (ROOT / name).exists(), name
