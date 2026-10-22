"""data_loader.py - settings + loading and preparing the bamboo biomass dataset.

Source: Masisi, B. (2022) "Bamboo biomass", Zenodo, DOI 10.5281/zenodo.7024827 (CC BY 4.0).

IMPORTANT FACT ABOUT THE DATA
The workbook has 3,643 rows, but only 60 of them are harvested culms with weights
(one culm per plot). The other rows are just diameters of other culms in the same plots.
So we have 60 real observations, and 59 of them can be used (plot 43 is incomplete).

Run `python data_loader.py` to rebuild the two files in data/processed/.
"""
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------ settings
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "external" / "zenodo_7024827"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

SEED = 42                      # fixed random seed -> same results every run
CV_FOLDS = 5                   # 5-fold cross-validation ...
CV_REPEATS = 20                # ... repeated 20 times with different shuffles

FEATURES = ["dbh_cm", "height_m", "age_years", "species"]       # model inputs (order matters!)
NUMERIC_FEATURES = ["dbh_cm", "height_m", "age_years"]
CATEGORICAL_FEATURES = ["species"]
TARGET = "agb_dry_kg"          # dry above-ground biomass of ONE culm, in kg

# Carbon is NOT measured in this dataset. The dataset authors assumed 0.5 x dry mass.
CARBON_FRACTION = 0.5

# Column names of RAWDATA.xlsx (the sheet has a 3-row merged header, so we name columns ourselves)
RAW_COLUMNS = [
    "plot_no", "eastings", "northings", "elevation_m", "village", "eco_type", "mgt_option",
    "tree_no", "species", "dbh", "height", "age",
    "wt_stem_kg", "wt_branch_kg", "wt_foliage_kg",                                   # whole component, fresh, kg
    "fresh_s0_g", "fresh_s25_g", "fresh_s50_g", "fresh_s75_g", "fresh_br_g", "fresh_fol_g",   # subsamples, fresh, g
    "dry_s0_g", "dry_s25_g", "dry_s50_g", "dry_s75_g", "dry_br_g", "dry_fol_g",               # subsamples, dried, g
]
# dry/fresh ratio of each subsample: name -> (fresh column, dry column)
SUBSAMPLES = {
    "s0": ("fresh_s0_g", "dry_s0_g"), "s25": ("fresh_s25_g", "dry_s25_g"),
    "s50": ("fresh_s50_g", "dry_s50_g"), "s75": ("fresh_s75_g", "dry_s75_g"),
    "br": ("fresh_br_g", "dry_br_g"), "fol": ("fresh_fol_g", "dry_fol_g"),
}


# ------------------------------------------------------------------ loading raw files
def load_raw_workbook():
    """Read RAWDATA.xlsx exactly as published and give the columns readable names."""
    path = RAW_DIR / "RAWDATA.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README.md ('Dataset') for the Zenodo download link.")
    top = pd.read_excel(path, header=None, nrows=3)
    if top.shape[1] != len(RAW_COLUMNS) or top.iloc[2, 12] != "Weight_stem":
        raise ValueError("RAWDATA.xlsx does not have the layout that was audited.")
    raw = pd.read_excel(path, header=None, skiprows=3)      # skip the 3 header rows
    raw.columns = RAW_COLUMNS
    return raw


def load_districts():
    """District of each plot (only the authors' analyses.csv has it)."""
    csv = pd.read_csv(RAW_DIR / "analyses.csv", encoding="utf-8-sig")
    districts = csv[["Plot_No", "District"]].drop_duplicates()
    return districts.rename(columns={"Plot_No": "plot_no", "District": "district"})


# ------------------------------------------------------------------ preparing the dataset
def build_culm_table(raw):
    """Keep the 60 harvested culms and compute dry biomass (kg) with the authors' method:

        dry mass of a component = fresh weight of the component x (dry / fresh) of its subsample
        stem uses the average ratio of its 4 subsamples (0, 25, 50 and 75 % of stem length)
        dry biomass of the culm = stem + branches + foliage
    """
    for col in RAW_COLUMNS:                                  # make sure numbers are numbers
        if col not in ("village", "eco_type", "mgt_option", "species"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    culms = raw[raw["wt_stem_kg"].notna()].copy()            # only rows with weights = harvested culms
    assert culms["plot_no"].is_unique and (culms["tree_no"] == 1).all()

    for name, (fresh, dry) in SUBSAMPLES.items():
        culms["ratio_" + name] = culms[dry] / culms[fresh]

    stem_ratio = culms[["ratio_s0", "ratio_s25", "ratio_s50", "ratio_s75"]].mean(axis=1, skipna=False)
    culms["stem_dry_kg"] = culms["wt_stem_kg"] * stem_ratio
    culms["branch_dry_kg"] = culms["wt_branch_kg"] * culms["ratio_br"]
    culms["foliage_dry_kg"] = culms["wt_foliage_kg"] * culms["ratio_fol"]
    parts = ["stem_dry_kg", "branch_dry_kg", "foliage_dry_kg"]
    culms[TARGET] = culms[parts].sum(axis=1, min_count=3)    # stays empty (NaN) if any part is missing
    culms["fresh_total_kg"] = culms[["wt_stem_kg", "wt_branch_kg", "wt_foliage_kg"]].sum(axis=1)

    # data-quality flags (rows are NOT deleted or "fixed" silently)
    ratios = ["ratio_" + name for name in SUBSAMPLES]
    culms["flag_missing_subsample"] = culms[ratios].isna().any(axis=1)      # plot 43
    culms["flag_dry_ge_fresh"] = (culms[ratios] >= 1.0).any(axis=1)         # plot 58: dry mass can't equal fresh mass
    culms["usable_for_target"] = culms[TARGET].notna()

    culms = culms.rename(columns={"dbh": "dbh_cm", "height": "height_m", "age": "age_years"})
    culms["species"] = culms["species"].astype(str).str.strip()
    culms = culms.merge(load_districts(), on="plot_no", how="left")
    culms = culms.drop(columns=["eastings", "northings", "elevation_m", "tree_no"])   # coordinates not needed; elevation disagrees between files
    return culms.reset_index(drop=True)


def prepare_dataset():
    """Raw files -> (all 60 culms, 59 usable culms). Writes both to data/processed/."""
    all_culms = build_culm_table(load_raw_workbook())
    usable = all_culms[all_culms["usable_for_target"]].reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_culms.to_csv(PROCESSED_DIR / "all_60_culms_audit.csv", index=False, encoding="utf-8")
    usable.to_csv(PROCESSED_DIR / "biomass_modeling_table.csv", index=False, encoding="utf-8")
    return all_culms, usable


def load_modeling_table(exclude_flagged=False):
    """The table used for modelling (59 culms). Builds it from the raw files if needed."""
    path = PROCESSED_DIR / "biomass_modeling_table.csv"
    # float_precision="round_trip" reads the numbers back EXACTLY as saved (the default fast parser can be
    # off by one unit in the last digit, which slightly changes random-forest results)
    table = pd.read_csv(path, encoding="utf-8", float_precision="round_trip") if path.exists() else prepare_dataset()[1]
    if exclude_flagged:                                      # used for the sensitivity check (drops plot 58)
        table = table[~table["flag_dry_ge_fresh"]].reset_index(drop=True)
    return table


def check_no_leakage(feature_names):
    """Stop if a model input contains (a part of) the thing we are predicting."""
    forbidden = {TARGET, "stem_dry_kg", "branch_dry_kg", "foliage_dry_kg", "fresh_total_kg",
                 "wt_stem_kg", "wt_branch_kg", "wt_foliage_kg", "plot_no"}
    bad = [f for f in feature_names if f in forbidden or f.startswith(("fresh_", "dry_", "ratio_"))]
    if bad:
        raise ValueError(f"Target leakage / identifier used as input: {bad}")


if __name__ == "__main__":
    everything, usable = prepare_dataset()
    print(f"harvested culms: {len(everything)}   usable for modelling: {len(usable)}")
    print("saved to", PROCESSED_DIR)
