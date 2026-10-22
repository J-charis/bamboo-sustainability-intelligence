"""app.py - Bamboo Sustainability Intelligence (Streamlit dashboard).

Run:  python -m streamlit run app.py

This file only DISPLAYS things. It never trains a model:
  - predictions come from predict.py (loads the saved model)
  - tables/metrics come from data/processed/ and models/ (created by train.py and evaluate.py)
Sections: Overview | Biomass Predictor | Data Insights | About & Methodology
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import data_loader as dl
from predict import InvalidInputError, ModelNotAvailableError, carbon_estimate_kg, load_model, predict_biomass

FOREST, SAGE, INK = "#1f4d2e", "#8fae8b", "#2b3036"
SPECIES_STYLE = {"Bambusa vulgaris": (FOREST, "o"), "Oxytenanthera abyssinica": (SAGE, "^")}
NAV = ["Overview", "Biomass Predictor", "Data Insights", "About & Methodology"]
DEFAULTS = {"dbh": 5.5, "height": 9.5, "age": 2.0}

st.set_page_config(page_title="Bamboo Sustainability Intelligence", page_icon="🌿", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{ font-family: "Inter","Segoe UI",system-ui,sans-serif; color:{INK}; }}
.stApp {{ background:#fbfcfa; }}
.block-container {{ padding-top:2.2rem; max-width:1180px; }}
h1,h2,h3,h4 {{ color:{FOREST}; letter-spacing:-.01em; }}
#MainMenu, footer {{ visibility:hidden; }}
[data-testid="stSidebar"] {{ background:#f1f5ef; border-right:1px solid #e1e8dd; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid #e3e8e0; border-radius:12px; padding:14px 16px; box-shadow:0 1px 4px rgba(31,77,46,.06); }}
[data-testid="stMetricLabel"] p {{ color:#5d685c; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }}
[data-testid="stMetricValue"] {{ color:{FOREST}; }}
.card {{ background:#fff; border:1px solid #e3e8e0; border-radius:12px; padding:16px 18px; box-shadow:0 1px 4px rgba(31,77,46,.06); height:100%; }}
.chip {{ display:inline-block; background:#e9f0e6; color:{FOREST}; border-radius:999px; padding:2px 10px; font-size:.75rem; font-weight:600; margin-bottom:6px; }}
.chip.sec {{ background:#f3f3ef; color:#6a6f66; }}
.note {{ background:#f4f8f1; border-left:4px solid {SAGE}; border-radius:6px; padding:10px 14px; font-size:.9rem; }}
.hero-num {{ font-size:3rem; font-weight:700; color:{FOREST}; line-height:1.1; }}
.sub {{ color:#5d685c; margin-top:-.6rem; }}
div.stButton > button[kind="primary"] {{ background:{FOREST}; border:0; border-radius:8px; font-weight:600; padding:.55rem 1.4rem; }}
div.stButton > button[kind="primary"]:hover {{ background:#173b23; }}
</style>""", unsafe_allow_html=True)


# ====================================================================== loading saved files
@st.cache_data
def read_csv(path):
    p = pathlib.Path(path)
    return pd.read_csv(p, encoding="utf-8") if p.exists() else None

@st.cache_data
def read_json(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_resource
def get_model():
    return load_model()

table = read_csv(str(dl.PROCESSED_DIR / "biomass_modeling_table.csv"))
card = read_json(str(dl.MODEL_DIR / "model_card.json"))
cv_predictions = read_csv(str(dl.MODEL_DIR / "cv_predictions.csv"))
robust = read_json(str(dl.MODEL_DIR / "evaluation_results.json"))    # optional: made by evaluate.py
try:
    bundle, load_error = get_model(), None
except ModelNotAvailableError as e:
    bundle, load_error = None, str(e)

if table is None or card is None:
    st.error("Data or the trained model are missing. Run `python data_loader.py`, then `python train.py` in the project folder, and reload.")
    st.stop()

cv, SEL, N = card["cv_results"], card["selected_model"], card["n_train"]
RG, SPECIES = card["training_ranges"], card["species"]
CV_TEXT = f"{card['cv']['folds']}-fold cross-validation × {card['cv']['repeats']} repeats"
NEW_SITE = robust["logo"][SEL] if robust else None
# one cross-validated prediction per culm (average over the repeats) for the actual-vs-predicted charts
oof = None if cv_predictions is None else cv_predictions.groupby(["plot_no", "species", "district"], as_index=False).agg(
    actual_kg=("actual_kg", "first"), predicted_kg=(SEL, "mean"))


# ====================================================================== charts (matplotlib, one minimal style)
def new_chart(w=4.6, h=3.3):
    fig, ax = plt.subplots(figsize=(w, h))
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color("#b8c2b4")
    ax.tick_params(colors=INK, labelsize=8); ax.grid(color="#e3e8e0", linewidth=.8); ax.set_axisbelow(True)
    ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
    return fig, ax

def finish(fig, ax, title=None, legend=True):
    if title: ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=10)
    if legend: ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def species_scatter(ax, df, x, y):
    for sp, g in df.groupby("species"):
        color, marker = SPECIES_STYLE[sp]
        ax.scatter(g[x], g[y], s=30, color=color, marker=marker, alpha=.85, edgecolor="white", linewidth=.5, label=sp)

def chart_scatter(x, xlabel, title=None, **size):
    fig, ax = new_chart(**size); species_scatter(ax, table, x, dl.TARGET)
    ax.set(xlabel=xlabel, ylabel="Dry biomass per culm (kg)"); finish(fig, ax, title)

def chart_histogram(column, xlabel, title=None, by_species=False, **size):
    fig, ax = new_chart(**size)
    if by_species:
        bins = np.histogram_bin_edges(table[column], bins=10)
        for sp, g in table.groupby("species"):
            ax.hist(g[column], bins=bins, alpha=.75, color=SPECIES_STYLE[sp][0], label=f"{sp} (n={len(g)})", edgecolor="white")
    else:
        ax.hist(table[column], bins=12, color=FOREST, edgecolor="white")
    ax.set(xlabel=xlabel, ylabel="Culms"); finish(fig, ax, title, legend=by_species)

def chart_actual_vs_predicted(title=None, **size):
    fig, ax = new_chart(**size); species_scatter(ax, oof, "actual_kg", "predicted_kg")
    top = max(oof["actual_kg"].max(), oof["predicted_kg"].max()) * 1.05
    ax.plot([0, top], [0, top], color=INK, lw=1, ls="--", label="perfect prediction")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Measured dry biomass (kg)", ylabel="Cross-validated prediction (kg)"); finish(fig, ax, title)

def chart_residuals(**size):
    fig, ax = new_chart(**size); oof["residual_kg"] = oof["actual_kg"] - oof["predicted_kg"]
    species_scatter(ax, oof, "predicted_kg", "residual_kg"); ax.axhline(0, color=INK, lw=1)
    ax.set(xlabel="Predicted dry biomass (kg)", ylabel="Measured - predicted (kg)"); finish(fig, ax, "Residuals")

def chart_model_comparison(**size):
    fig, ax = new_chart(**size); names = list(cv)
    rmse = [cv[n]["RMSE"] for n in names]; ci = [cv[n]["RMSE_ci95_bootstrap_over_plots"] for n in names]
    colors = ["#c9d1c5" if n.startswith("Baseline") else (FOREST if n == SEL else SAGE) for n in names]
    ax.barh(names, rmse, xerr=[[r - c[0] for r, c in zip(rmse, ci)], [c[1] - r for r, c in zip(rmse, ci)]], color=colors, capsize=3, ecolor="#6b756a")
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("CV RMSE (kg) · bars: 95% interval")
    finish(fig, ax, "Model comparison (lower is better)", legend=False)

def chart_boxes(group, **size):
    fig, ax = new_chart(**size); groups = [(k, g[dl.TARGET].to_numpy()) for k, g in table.groupby(group)]
    box = ax.boxplot([v for _, v in groups], tick_labels=[f"{str(k).replace(' ', chr(10), 1)}\n(n={len(v)})" for k, v in groups],
                     patch_artist=True, medianprops={"color": INK})
    for b in box["boxes"]: b.set(facecolor=SAGE, edgecolor=FOREST, alpha=.8)
    ax.set_ylabel("Dry biomass per culm (kg)"); ax.tick_params(axis="x", labelsize=7); finish(fig, ax, legend=False)


# ====================================================================== small layout helpers
def full_width(fn, *a, **k):
    """Full-width widgets on both old (use_container_width) and new (width='stretch') Streamlit."""
    try: return fn(*a, width="stretch", **k)
    except TypeError: return fn(*a, use_container_width=True, **k)

def show_table(*a, **k): return full_width(st.dataframe, *a, **k)
def card_box(html): st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)
def note(html): st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)

def header(title, subtitle):
    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:.2rem">
<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="{FOREST}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21V8"/><path d="M12 8c0-3 2-5 5-5 0 3-2 5-5 5z"/><path d="M12 13c0-2.5-1.8-4.3-4.5-4.3 0 2.5 1.8 4.3 4.5 4.3z"/><path d="M12 17.5c0-2-1.5-3.5-3.7-3.5 0 2 1.5 3.5 3.7 3.5z"/></svg>
<h1 style="margin:0">{title}</h1></div><p class="sub">{subtitle}</p>""", unsafe_allow_html=True)


# ====================================================================== sidebar
st.sidebar.markdown(f"<div style='font-weight:700;color:{FOREST};font-size:1.05rem'>🌿 Bamboo Sustainability<br>Intelligence</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Section", NAV, key="page", label_visibility="collapsed")
st.sidebar.caption("Research prototype · estimates, not certified forestry measurements.")


# ====================================================================== 1. OVERVIEW
def overview():
    header("Bamboo Sustainability Intelligence", "Data-driven insights for sustainable bamboo resource management")
    k = st.columns(4)
    k[0].metric("Usable observations", N, help="Destructively sampled culms (one per plot). The workbook has 3,643 rows but only 60 carry biomass weights; 59 have a computable dry-mass target.")
    k[1].metric("Selected model", SEL, help="Simplest model statistically indistinguishable from the best in cross-validation.")
    k[2].metric("Cross-validated MAE", f"{cv[SEL]['MAE']:.2f} kg", help="Average absolute error in dry biomass per culm.")
    k[3].metric("Cross-validated R²", f"{cv[SEL]['R2']:.2f}", help="Share of biomass variation explained on held-out culms.")
    extra = f" On a district the model had not seen, RMSE rose to {NEW_SITE['RMSE']:.1f} kg (R² {NEW_SITE['R2']:.2f})." if NEW_SITE else ""
    st.caption(f"Evaluation: {CV_TEXT} on the same {N} culms (RMSE {cv[SEL]['RMSE']:.2f} kg). These are research estimates of expected error, not a guarantee for new sites or species.{extra}")
    c = st.columns(3)
    with c[0]: chart_scatter("dbh_cm", "DBH (cm)", "DBH vs dry biomass", w=4.2, h=3.2)
    with c[1]: chart_histogram(dl.TARGET, "Dry biomass per culm (kg)", "Biomass distribution", w=4.2, h=3.2)
    with c[2]:
        if oof is not None: chart_actual_vs_predicted("Actual vs predicted (cross-validated)", w=4.2, h=3.2)
        else: st.info("Cross-validated predictions not found. Run `python train.py`.")
    st.markdown("#### Why biomass estimation matters")
    s = st.columns(3)
    with s[0]: card_box('<span class="chip">SDG 15 · Life on Land</span><br>Non-destructive biomass estimates can support bamboo resource assessment without felling culms.')
    with s[1]: card_box('<span class="chip">SDG 12 · Responsible Consumption &amp; Production</span><br>Knowing how much biomass is standing helps plan harvest and use more sustainably.')
    with s[2]: card_box('<span class="chip sec">SDG 13 · Climate Action (secondary)</span><br>Biomass is one input to carbon accounting. This model does <b>not</b> measure carbon sequestration or emissions reductions.')


# ====================================================================== 2. BIOMASS PREDICTOR
def reset_inputs():
    st.session_state.update(species=SPECIES[0], dbh=DEFAULTS["dbh"], height=DEFAULTS["height"], age=DEFAULTS["age"], pred=None, pred_errors=None)

def run_prediction():
    values = (st.session_state["dbh"], st.session_state["height"], st.session_state["age"], st.session_state["species"])
    try:
        st.session_state["pred"] = {"inputs": values, "result": predict_biomass(*values, bundle=bundle)}   # <- predict.py
        st.session_state["pred_errors"] = None
    except InvalidInputError as e:
        st.session_state["pred"], st.session_state["pred_errors"] = None, e.problems

def predictor():
    header("Biomass Predictor", "Estimate the dry above-ground biomass of one bamboo culm")
    if bundle is None:
        st.error(f"The prediction model could not be loaded, so no estimate can be shown. {load_error}"); return
    st.session_state.setdefault("species", SPECIES[0])
    for key, value in DEFAULTS.items(): st.session_state.setdefault(key, value)
    st.session_state.setdefault("pred", None); st.session_state.setdefault("pred_errors", None)
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Measurements")
        st.selectbox("Species", SPECIES, key="species")
        st.number_input("DBH — diameter at breast height (cm)", min_value=0.0, step=0.1, key="dbh", help=f"Model trained on {RG['dbh_cm'][0]:g}–{RG['dbh_cm'][1]:g} cm")
        st.number_input("Culm height / length (m)", min_value=0.0, step=0.1, key="height", help=f"Model trained on {RG['height_m'][0]:g}–{RG['height_m'][1]:g} m")
        st.number_input("Culm age (years)", min_value=0.0, step=0.25, key="age", help=f"Model trained on {RG['age_years'][0]:g}–{RG['age_years'][1]:g} years")
        st.caption(f"Accepted ranges · DBH {RG['dbh_cm'][0]:g}–{RG['dbh_cm'][1]:g} cm · height {RG['height_m'][0]:g}–{RG['height_m'][1]:g} m · age {RG['age_years'][0]:g}–{RG['age_years'][1]:g} years. Units are assumed (the source workbook does not state them).")
        b1, b2, _ = st.columns([2, 1.2, 2])
        b1.button("Predict Biomass", type="primary", on_click=run_prediction, key="predict_btn")
        b2.button("Reset", on_click=reset_inputs, key="reset_btn")
    with right:
        st.markdown("##### Estimate")
        errors, pred = st.session_state["pred_errors"], st.session_state["pred"]
        current = (st.session_state["dbh"], st.session_state["height"], st.session_state["age"], st.session_state["species"])
        if errors:
            for problem in errors: st.error(problem)
            st.caption("No estimate is shown for inputs outside what the model was trained on; extrapolation would be unreliable.")
        elif pred is None:
            note("Enter the measurements and press <b>Predict Biomass</b>.")
        elif pred["inputs"] != current:
            note("Inputs changed. Press <b>Predict Biomass</b> to update the estimate.")
        else:
            r = pred["result"]; est = r["dry_biomass_kg"]; v = r["inputs"]
            st.markdown(f'<div class="card"><div style="color:#5d685c;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em">Predicted dry above-ground biomass</div>'
                        f'<div class="hero-num">{est:.2f} <span style="font-size:1.2rem">kg per culm</span></div></div>', unsafe_allow_html=True)
            if r["interval_80_kg"]:
                low, high = r["interval_80_kg"]; st.write(f"Approximate 80% range from cross-validated errors: **{low:.1f} – {high:.1f} kg**")
            if r["clipped_to_zero"]: st.warning("The linear model extrapolated below 0 kg; shown as 0.")
            peers = table[table["species"] == v["species"]][dl.TARGET]
            st.markdown(f"**In plain English:** for a *{v['species']}* culm with DBH {v['dbh_cm']:g} cm, height {v['height_m']:g} m and age {v['age_years']:g} years, the model estimates about **{est:.1f} kg** of dry mass. "
                        f"Culm diameter drives most of this estimate; height and age had little influence in this dataset. "
                        f"For comparison, the {len(peers)} sampled {v['species']} culms had a median of {peers.median():.1f} kg.")
            note(f"<b>Limits.</b> This is a statistical estimate, not a physical measurement. In testing, estimates were typically off by about {cv[SEL]['MAE']:.1f} kg"
                 + (f" and by more (RMSE {NEW_SITE['RMSE']:.1f} kg) on a district not used for training" if NEW_SITE else "") + f". Based on only {N} culms of two species.")
            with st.expander("Optional: illustrative carbon estimate (assumption-based)"):
                st.metric("Illustrative carbon", f"{carbon_estimate_kg(est):.2f} kg C")
                st.caption(f"= {dl.CARBON_FRACTION} × dry biomass. The dataset has no measured carbon content; 0.5 is the source authors' assumed fraction. This is not a measured carbon value, a sequestration rate, or a stand-level figure.")


# ====================================================================== 3. DATA INSIGHTS
LABELS = {"dbh_cm": "DBH (cm)", "height_m": "Height (m)", "age_years": "Age (years)", dl.TARGET: "Dry biomass (kg)"}

def insights():
    header("Data Insights", "Explore the dataset and how well the model performs")
    note(f"<b>Small sample.</b> Only {N} culms (two species, three districts) support this analysis. Patterns are descriptive; they do not show cause and effect.")
    tab1, tab2, tab3 = st.tabs(["Dataset", "Relationships", "Model performance"])
    with tab1:
        a = st.columns(4)
        a[0].metric("Culms used", N); a[1].metric("Species", table["species"].nunique()); a[2].metric("Districts", table["district"].nunique())
        a[3].metric("Missing values", int(table[dl.FEATURES + [dl.TARGET]].isna().sum().sum()), help="In the modelling table. Plot 43 was excluded rather than imputed.")
        left, right = st.columns(2)
        with left:
            var = st.selectbox("Distribution of", list(LABELS), key="dist_var", format_func=LABELS.get)
            chart_histogram(var, LABELS[var], by_species=True, w=4.8, h=3.2)
        with right:
            st.markdown("**Coverage (culms per district × species)**")
            show_table(table.groupby(["district", "species"]).size().unstack(fill_value=0))
            st.caption("Species and district overlap only partly (Rungwe contains *B. vulgaris* only), so group differences cannot be separated from site effects.")
        with st.expander("Data preview"):
            cols = ["plot_no", "district", "eco_type", "mgt_option", "species", "dbh_cm", "height_m", "age_years", "fresh_total_kg", dl.TARGET]
            show_table(table[cols], hide_index=True)
    with tab2:
        left, right = st.columns(2)
        with left:
            x = st.selectbox("Compare biomass with", ["dbh_cm", "height_m", "age_years"], key="xvar", format_func=LABELS.get)
            chart_scatter(x, LABELS[x], w=4.8, h=3.4)
            st.caption(f"Spearman correlation with dry biomass: {table[[x, dl.TARGET]].corr(method='spearman').iloc[0, 1]:.2f}. A correlation is not evidence of cause.")
        with right:
            group = st.selectbox("Group biomass by", ["species", "mgt_option", "eco_type", "district"], key="grp",
                                 format_func={"species": "Species", "mgt_option": "Management option", "eco_type": "Ecosystem type", "district": "District"}.get)
            chart_boxes(group, w=4.8, h=3.4)
            st.caption("Groups are small and overlap; management was not randomised. Differences are exploratory only.")
    with tab3:
        left, right = st.columns([5, 6], gap="large")
        with left:
            chart_model_comparison(w=4.8, h=3.3)
            show_table(pd.DataFrame(cv).T[["MAE", "MSE", "RMSE", "R2"]].round(2))
        with right:
            if oof is not None: chart_actual_vs_predicted(f"Actual vs predicted · {SEL}", w=4.8, h=3.6)
        if oof is not None:
            with st.expander("Residuals (measured − predicted)"): chart_residuals(w=6.5, h=3.2)
        verdict = "reliably beats" if card["selection"]["beats_baseline_reliably"] else "does NOT reliably beat"
        st.markdown(f"**What this means.** Evaluated with {CV_TEXT}, **{SEL}** {verdict} a model that always guesses the average biomass "
                    f"(RMSE {cv[SEL]['RMSE']:.2f} kg vs {cv['Baseline (mean)']['RMSE']:.2f} kg). "
                    + (f"On a district held out entirely it was less accurate (RMSE {NEW_SITE['RMSE']:.2f} kg), so expect larger errors at new sites. " if NEW_SITE else "")
                    + "With a sample this small the uncertainty is wide.")
        if robust:
            with st.expander("Robustness checks"):
                st.write("New-district test (leave-one-district-out)"); show_table(pd.DataFrame(robust["logo"]).T[["MAE", "RMSE", "R2"]].round(2))
                st.write("Which inputs matter (linear regression with fewer inputs)"); show_table(pd.DataFrame(robust["ablation"]).T[["MAE", "RMSE", "R2"]].round(2))


# ====================================================================== 4. ABOUT & METHODOLOGY
def about():
    header("About & Methodology", "Data sources, method, limits and SDG alignment")
    st.markdown(f"""
**Objective.** Estimate the dry above-ground biomass of a single bamboo culm from quick, non-destructive measurements, and show honestly how reliable that estimate is.

**Dataset.** Masisi, B. (2022). *Bamboo biomass*. Zenodo, DOI 10.5281/zenodo.7024827, licence **CC BY 4.0**. 60 plots, one destructively sampled culm per plot; *{SPECIES[0]}* and *{SPECIES[1]}*; three districts.

**Preprocessing.** Target = stem, branch and foliage fresh weight × the dry:fresh ratio of its subsamples (the authors' documented method, reproduced). Plot 43 excluded (missing subsample weights, not imputed); plot 58 flagged (dry weight = fresh weight) and tested in a sensitivity run. No duplicates or outliers were removed. DBH, height and age units are assumed (cm, m, years) because the workbook does not state them. Inputs: species, DBH, height, age.

**Model and evaluation.** Five models were compared (mean baseline, linear, log-log linear, random forest, extra trees) using {CV_TEXT}. The simplest model statistically indistinguishable from the best was selected: **{SEL}**. A leave-one-district-out test checks transfer to new sites.

**Limitations.** {N} culms; two species; three districts; mostly young (1–3 year) culms; assumed units; per-culm estimates only (no stand or hectare estimate); species, ecosystem and district are entangled, so group differences are not causal; carbon content was not measured.
""")
    s = st.columns(3)
    with s[0]: card_box('<span class="chip">SDG 15</span><br><b>Life on Land.</b> Supports non-destructive assessment of bamboo resources.')
    with s[1]: card_box('<span class="chip">SDG 12</span><br><b>Responsible Consumption &amp; Production.</b> Informs planning of harvest and use.')
    with s[2]: card_box('<span class="chip sec">SDG 13 (secondary)</span><br><b>Climate Action.</b> Potential input to carbon accounting only; no sequestration or emissions claims are made.')
    st.markdown("")
    note("<b>Disclaimer.</b> Predictions are research estimates from a student project, not certified forestry measurements, and must not be used for carbon credits, regulatory reporting or stand inventories.")
    st.markdown("**More detail:** `README.md` (how to run, how it works) and `METHODOLOGY.md` (dataset audit and evaluation method).")
    st.caption("Reference: Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, JMLR 12:2825–2830.")


{NAV[0]: overview, NAV[1]: predictor, NAV[2]: insights, NAV[3]: about}[page]()
