"""
================================================================================
Predicting EBITDA Growth in UK Mid-Market Private Equity-Owned Companies
Analysis pipeline

Adrian Igharo (250369782)
MSc Artificial Intelligence with Business Strategy, Aston University
EP4DIS MSc Dissertation, September 2026
================================================================================

PURPOSE
    Reproduces every result, table and figure reported in Chapter 4 from the
    raw FAME export.

INPUT
    FAME_export.xlsx    Companies House financial data via FAME (Bureau van
                        Dijk), retrieved 11 September 2026. Search criteria:
                          1. Active companies, not in receivership or dormant
                          2. England, Scotland, Wales or Northern Ireland
                          3. Turnover >= GBP 10.2m, 2022
                          4. EBITDA reported, 2022
                          5. Ultimate owner type = Private equity firm,
                             minimum ownership path 50.01%
                        Returns 564 companies. Columns extracted: turnover,
                        EBITDA and employees for 2019-2024; SIC code;
                        incorporation date; registered number.

OUTPUT
    analytical_sample.csv   Cleaned dataset, n = 341
    table_4_1 ... 4_4       Descriptives, model performance, OLS, robustness
    fig4_1 ... fig4_4.png   Figures as reported in Chapter 4

DEPENDENCIES
    python 3.12  pandas 2.x  numpy  scikit-learn 1.8  xgboost 3.4
    shap 0.52  statsmodels  scipy  matplotlib  openpyxl

REPRODUCIBILITY
    All stochastic components use RANDOM_STATE = 42. Cross-validation folds
    are fixed. Re-running reproduces reported figures exactly.
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shap
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONSTANTS

SOURCE_FILE   = "FAME_export.xlsx"
RANDOM_STATE  = 42
N_SEEDS       = 20            # repeated CV fold assignments
BASELINE_YEAR = 2022          # T: predictors measured at or before this year
OUTCOME_YEAR  = 2024          # outcome measured to this year
PRE_YEAR      = 2019          # pre-period base (primary specification)
PRE_YEAR_ALT  = 2020          # pre-period base (robustness; Covid-affected)

TURNOVER_FLOOR_K   = 10_200   # GBP thousands. Companies Act 2006 small-company
                              # threshold; below this, abbreviated accounts may
                              # be filed and EBITDA cannot be derived.
TURNOVER_CEILING_K = 300_000  # GBP thousands. Approximates the upper bound of
                              # KPMG's GBP 10-300m mid-market definition.
MARGIN_CEILING     = 0.60     # Above this, reported turnover does not capture
                              # trading income (holding entities).
FILING_ERROR_BOUNDS = (0.05, 20)    # implausible turnover movement 2022-24
WINSOR             = (0.05, 0.95)   # outcome
WINSOR_PREDICTORS  = (0.01, 0.99)   # predictors

# Structural suffixes stripped when matching company name stems, so that
# "Deuce Topco", "Deuce Holdco" and "Deuce Midco" resolve to one group.
STRUCTURAL_TERMS = (
    r"\b(TOPCO|BIDCO|MIDCO|HOLDCO|FINCO|PARENTCO|HOLDINGS?|"
    r"ACQUISITIONS?|GROUP|UK|LIMITED|LTD|PLC|LLP)\b"
)
VEHICLE_TERMS = r"\b(TOPCO|BIDCO|MIDCO|HOLDCO|FINCO|PARENTCO)\b"

TURNOVER = "Turnover th GBP"
EBITDA   = "EBITDA th GBP"
EMPLOYEE = "Number of employees"

FEATURES = [
    "log_turnover", "firm_age", "log_employees", "rev_per_emp",
    "margin_2022", "rev_cagr_pre", "ebitda_cagr_pre", "margin_trend_pre",
]

FEATURE_LABELS = {
    "log_turnover":     "Log turnover",
    "firm_age":         "Firm age",
    "log_employees":    "Log employees",
    "rev_per_emp":      "Revenue per employee",
    "margin_2022":      "EBITDA margin 2022",
    "rev_cagr_pre":     "Revenue CAGR 2019\u201322",
    "ebitda_cagr_pre":  "EBITDA CAGR 2019\u201322",
    "margin_trend_pre": "Margin trend 2019\u201322",
}

BLUE, RED, GREY = "#2F5D8C", "#A23B3B", "#B4B4B4"


def col(base: str, year: int) -> str:
    """FAME column name for a variable in a given year."""
    return f"{base} {year}"


# ------------------------------------------------------------------- LOADING

def load_fame_export(path: str = SOURCE_FILE) -> pd.DataFrame:
    """Read the FAME export and coerce financial columns to numeric.

    FAME writes 'n.a.' for missing values and exports all financial fields as
    text. The 'Results' sheet carries the data; 'Search summary' records the
    search criteria and is retained in the submitted file for audit purposes.
    """
    df = (pd.read_excel(path, sheet_name="Results", header=0)
            .iloc[:, 1:]                       # drop FAME's row-index column
            .replace("n.a.", np.nan))

    for c in df.columns:
        if any(k in c for k in (TURNOVER, EBITDA, EMPLOYEE)):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Date of incorporation"] = pd.to_datetime(
        df["Date of incorporation"], errors="coerce")
    return df


# ------------------------------------------------------------------ CLEANING

def deduplicate_by_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse holding-company stacks sharing identical baseline turnover.

    UK buyouts are executed through layered vehicles (Topco / Midco / Bidco),
    each filing consolidated accounts. Turnover consolidates unchanged up the
    stack while EBITDA differs by holding-company costs, so identical turnover
    reliably identifies one economic group.

    Where a group is found, retain the entity with the most complete EBITDA
    history, preferring operating companies over named acquisition vehicles
    and older incorporations to break remaining ties.
    """
    t = col(TURNOVER, BASELINE_YEAR)
    df = df.copy()
    df["_completeness"] = df[[col(EBITDA, y)
                              for y in range(PRE_YEAR, OUTCOME_YEAR + 1)]].notna().sum(axis=1)
    df["_is_vehicle"] = df["Company name"].str.upper().str.contains(
        VEHICLE_TERMS, regex=True, na=False)

    ranked = df[df[t].notna()].sort_values(
        ["_completeness", "_is_vehicle", "Date of incorporation"],
        ascending=[False, True, True])
    keep = ranked.drop_duplicates(t, keep="first").index

    return df.loc[sorted(set(keep) | set(df[df[t].isna()].index))]


def deduplicate_by_name_stem(df: pd.DataFrame) -> pd.DataFrame:
    """Second deduplication pass for stacks whose turnover differs marginally.

    Matches on the first two words of the company name after removing
    structural suffixes, catching cases such as "Salisbury Topco" and
    "Salisbury Bidco" that report slightly different consolidated turnover.
    """
    df = df.copy()
    df["_stem"] = (df["Company name"].str.upper()
                     .str.replace(r"[^A-Z ]", "", regex=True)
                     .str.replace(STRUCTURAL_TERMS, "", regex=True)
                     .str.split().str[:2].str.join(" ").str.strip())
    df["_completeness"] = df[[col(EBITDA, y)
                              for y in range(PRE_YEAR, OUTCOME_YEAR + 1)]].notna().sum(axis=1)
    df["_is_vehicle"] = df["Company name"].str.upper().str.contains(
        VEHICLE_TERMS, regex=True, na=False)

    # Stems of three characters or fewer are too short to match reliably and
    # are passed through unchanged.
    long_stem = df[df["_stem"].str.len() > 3]
    kept = (long_stem.sort_values(["_completeness", "_is_vehicle"],
                                  ascending=[False, True])
                     .drop_duplicates("_stem", keep="first"))
    return pd.concat([kept, df[df["_stem"].str.len() <= 3]])


def build_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the screens and return the sample with an attrition log."""
    t, e = col(TURNOVER, BASELINE_YEAR), col(EBITDA, BASELINE_YEAR)
    log = [("FAME extract", len(df))]

    # Filing-error screen. A turnover movement of this magnitude over two
    # years reflects a restatement, a change of accounting reference date or
    # a cessation of trading rather than a business outcome. Examples include
    # a company reporting turnover of GBP 26.1m in 2022 and GBP 3,000 in 2024.
    ratio = df[col(TURNOVER, OUTCOME_YEAR)] / df[t]
    df = df[~((ratio < FILING_ERROR_BOUNDS[0]) |
              (ratio > FILING_ERROR_BOUNDS[1])).fillna(False)]
    log.append(("Filing-error screen", len(df)))

    df = deduplicate_by_turnover(df)
    log.append(("Deduplicate group stacks (turnover)", len(df)))

    df = df[(df[t] >= TURNOVER_FLOOR_K) & (df[t] <= TURNOVER_CEILING_K)]
    log.append(("Turnover GBP 10.2m-300m, 2022", len(df)))

    df = df[df[e].notna() & df[col(EBITDA, OUTCOME_YEAR)].notna()]
    log.append(("EBITDA present 2022 and 2024", len(df)))

    # Percentage growth is undefined from a non-positive base.
    df = df[df[e] > 0]
    log.append(("Baseline EBITDA positive", len(df)))

    df = deduplicate_by_name_stem(df)
    log.append(("Deduplicate group stacks (name stem)", len(df)))

    margin = df[e] / df[t]
    df = df[(margin > 0) & (margin <= MARGIN_CEILING)]
    log.append(("Baseline EBITDA margin 0-60%", len(df)))

    attrition = pd.DataFrame(log, columns=["Screen", "n"])
    attrition["Change"] = attrition["n"].diff().fillna(0).astype(int)
    return df, attrition


# --------------------------------------------------------- FEATURE BUILDING

def build_features(df: pd.DataFrame, pre_year: int = PRE_YEAR) -> pd.DataFrame:
    """Construct outcomes and predictors.

    All predictors are measured at or before BASELINE_YEAR, so that no
    information from the outcome window enters the feature set.
    """
    T, O, n = BASELINE_YEAR, OUTCOME_YEAR, BASELINE_YEAR - pre_year
    d = pd.DataFrame(index=df.index)

    d["company"] = df["Company name"]
    d["reg_no"]  = df["Registered number"]
    d["sic_div"] = (df["Primary UK SIC (2007) code"].astype(str)
                      .str.zfill(5).str[:2])

    # --- outcomes -----------------------------------------------------------
    growth = (df[col(EBITDA, O)] - df[col(EBITDA, T)]) / df[col(EBITDA, T)]
    d["ebitda_growth_raw"] = growth
    d["y_growth"] = growth.clip(*growth.quantile(WINSOR))

    d["margin_2022"] = df[col(EBITDA, T)] / df[col(TURNOVER, T)]
    d["y_margin_pp"] = (df[col(EBITDA, O)] / df[col(TURNOVER, O)]
                        - d["margin_2022"]) * 100

    # --- predictors: scale and maturity ------------------------------------
    d["log_turnover"]  = np.log(df[col(TURNOVER, T)])
    d["firm_age"]      = T - df["Date of incorporation"].dt.year
    d["log_employees"] = np.log(df[col(EMPLOYEE, T)])

    # --- predictors: operational efficiency --------------------------------
    d["rev_per_emp"] = df[col(TURNOVER, T)] / df[col(EMPLOYEE, T)]

    # --- predictors: pre-period trajectory ---------------------------------
    d["rev_cagr_pre"] = (df[col(TURNOVER, T)] / df[col(TURNOVER, pre_year)]) ** (1 / n) - 1
    d["ebitda_cagr_pre"] = np.where(
        df[col(EBITDA, pre_year)] > 0,
        (df[col(EBITDA, T)] / df[col(EBITDA, pre_year)]) ** (1 / n) - 1,
        np.nan)
    d["margin_trend_pre"] = (d["margin_2022"]
                             - df[col(EBITDA, pre_year)] / df[col(TURNOVER, pre_year)])

    # Winsorise predictors against filing anomalies.
    for c in FEATURES:
        d[c] = d[c].clip(*d[c].quantile(WINSOR_PREDICTORS))

    # Exclude implausible margin swings (filing errors, not business outcomes).
    return d[d["y_margin_pp"].abs() < 50]


# --------------------------------------------------------------- MODELLING

def make_pipeline(model, scale: bool = False) -> Pipeline:
    """Impute within fold, optionally scale, then fit.

    Imputation sits inside the pipeline so it is fitted on training folds
    only, preventing leakage from validation data.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_specifications() -> dict:
    """Four models: two linear, two ensemble.

    Hyperparameters are set conservatively given n = 341. Shallow trees and
    a low learning rate for XGBoost, and a minimum leaf size of eight for the
    random forest, constrain model complexity relative to sample size.
    """
    return {
        "OLS": make_pipeline(LinearRegression(), scale=True),
        "LASSO": make_pipeline(
            LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000), scale=True),
        "Random forest": make_pipeline(RandomForestRegressor(
            n_estimators=500, min_samples_leaf=8,
            random_state=RANDOM_STATE, n_jobs=-1)),
        "XGBoost": make_pipeline(XGBRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2,
            random_state=RANDOM_STATE, n_jobs=-1)),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series,
                    n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Cross-validated performance averaged over repeated fold assignments.

    A single five-fold split gives an estimate that depends materially on how
    observations happen to fall across folds. With n = 335 the variation is
    large enough to change conclusions: in this sample the random forest
    ranges from 0.049 to 0.167 across seeds. Performance is therefore
    reported as the mean and standard deviation over N_SEEDS repetitions,
    each with a different fold assignment.

    R-squared is computed on out-of-sample predictions, so a model failing to
    beat the sample mean returns a negative value.
    """
    naive = np.full(len(y), y.mean())
    rows = [{"Model": "Mean predictor",
             "RMSE": np.sqrt(mean_squared_error(y, naive)),
             "MAE": mean_absolute_error(y, naive),
             "R2 mean": 0.0, "R2 SD": 0.0}]

    for name, model in model_specifications().items():
        scores, rmses, maes = [], [], []
        for seed in range(n_seeds):
            cv = KFold(5, shuffle=True, random_state=seed)
            pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            scores.append(r2_score(y, pred))
            rmses.append(np.sqrt(mean_squared_error(y, pred)))
            maes.append(mean_absolute_error(y, pred))
        rows.append({"Model": name,
                     "RMSE": np.mean(rmses), "MAE": np.mean(maes),
                     "R2 mean": np.mean(scores), "R2 SD": np.std(scores)})
    return pd.DataFrame(rows).round(3)


def ols_table(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """OLS on standardised predictors, for coefficient interpretation.

    Standardisation means coefficients express the change in the outcome
    associated with a one standard deviation increase in the predictor,
    making magnitudes comparable across variables on different scales.
    """
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns, index=X.index)
    standardised = (imputed - imputed.mean()) / imputed.std()

    fit = sm.OLS(y, sm.add_constant(standardised)).fit()
    table = pd.DataFrame({
        "Coefficient": fit.params.round(3),
        "SE": fit.bse.round(3),
        "p": fit.pvalues.round(3),
    })
    stats_ = {"r2": fit.rsquared, "adj_r2": fit.rsquared_adj, "f_pvalue": fit.f_pvalue}
    return table, stats_


def shap_importance(X: pd.DataFrame, y: pd.Series):
    """SHAP values for the random forest, the best-performing specification.

    Returns both the fitted values matrix and mean absolute importance.
    SHAP decomposes each prediction into per-feature contributions, giving
    directional interpretation that permutation importance does not.
    """
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns, index=X.index)
    forest = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=8,
        random_state=RANDOM_STATE, n_jobs=-1).fit(imputed, y)
    values = shap.TreeExplainer(forest).shap_values(imputed)
    importance = pd.Series(np.abs(values).mean(axis=0),
                           index=X.columns).sort_values(ascending=False)
    return values, imputed, importance


def robustness_pre_period(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare pre-Covid (2019) and Covid-affected (2020) pre-period bases.

    A company whose earnings collapsed in 2020 and recovered by 2022 registers
    as high-growth when it has merely returned to trend. This test establishes
    which findings survive the removal of that distortion.
    """
    rows = []
    for label, year in [("Covid base (2020-22)", PRE_YEAR_ALT),
                        ("Pre-Covid base (2019-22)", PRE_YEAR)]:
        d = build_features(raw, pre_year=year)
        for feature in ["ebitda_cagr_pre", "margin_trend_pre",
                        "rev_cagr_pre", "margin_2022"]:
            pair = d[[feature, "y_margin_pp"]].dropna()
            rho, p = stats.spearmanr(pair[feature], pair["y_margin_pp"])
            rows.append({"Specification": label,
                         "Predictor": FEATURE_LABELS[feature],
                         "rho": round(rho, 3), "p": round(p, 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- FIGURES

def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.axisbelow": True, "figure.dpi": 200,
    })


def figure_model_performance(performance: pd.DataFrame) -> None:
    """Figure 4.1 - out-of-sample R-squared by model."""
    perf = performance[performance["Model"] != "Mean predictor"]
    order = ["OLS", "LASSO", "XGBoost", "Random forest"]
    perf = perf.set_index("Model").loc[order]

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    colours = [BLUE if m == "Random forest" else GREY for m in order]
    bars = ax.bar([m.replace(" ", "\n") for m in order],
                  perf["R2 mean"], color=colours, width=0.6)
    ax.axhline(0, color="#333", lw=0.8)
    for bar, value in zip(bars, perf["R2 mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                f"{value:.3f}", ha="center", fontsize=8.5)
    ax.errorbar(range(len(order)), perf["R2 mean"], yerr=perf["R2 SD"],
                fmt="none", ecolor="#333", capsize=3, lw=0.9)
    ax.set_ylabel("Out-of-sample R\u00b2")
    ax.set_ylim(0, (perf["R2 mean"] + perf["R2 SD"]).max() * 1.3)
    ax.set_title("Figure 4.1  Out-of-sample performance, EBITDA growth (mean \u00b1 SD, 20 CV seeds)",
                 loc="left", fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig("fig4_1.png", bbox_inches="tight")
    plt.close(fig)


def figure_shap_importance(importance: pd.Series) -> None:
    """Figure 4.2 - mean absolute SHAP value by feature."""
    ordered = importance.sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh([FEATURE_LABELS[i] for i in ordered.index],
            ordered.values, color=BLUE, height=0.65)
    for i, value in enumerate(ordered.values):
        ax.text(value + 0.004, i, f"{value:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_xlim(0, ordered.max() * 1.18)
    ax.set_title("Figure 4.2  Feature importance, random forest (EBITDA growth)",
                 loc="left", fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig("fig4_2.png", bbox_inches="tight")
    plt.close(fig)


def figure_mean_reversion(d: pd.DataFrame) -> None:
    """Figure 4.3 - subsequent EBITDA growth by quartile of prior growth."""
    quartile = pd.qcut(d["ebitda_cagr_pre"], 4,
                       labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])
    medians = d.groupby(quartile, observed=True)["y_growth"].median() * 100

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bars = ax.bar(medians.index.astype(str), medians.values,
                  color=[BLUE if v > 0 else RED for v in medians.values], width=0.6)
    ax.axhline(0, color="#333", lw=0.8)
    for bar, value in zip(bars, medians.values):
        offset = 2 if value > 0 else -4.5
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                f"{value:+.1f}%", ha="center", fontsize=8.5)
    ax.set_ylabel("Median EBITDA growth 2022\u201324 (%)")
    ax.set_xlabel("Quartile of EBITDA CAGR 2019\u201322")
    ax.set_ylim(medians.min() - 8, medians.max() * 1.15)
    ax.set_title("Figure 4.3  Mean reversion in earnings",
                 loc="left", fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig("fig4_3.png", bbox_inches="tight")
    plt.close(fig)


def figure_shap_beeswarm(values, imputed: pd.DataFrame) -> None:
    """Figure 4.4 - SHAP value distribution, showing direction of effect."""
    plt.figure(figsize=(6.2, 3.6))
    shap.summary_plot(values, imputed,
                      feature_names=[FEATURE_LABELS[f] for f in imputed.columns],
                      show=False, plot_size=None, max_display=len(imputed.columns))
    plt.title("Figure 4.4  SHAP value distribution by feature",
              loc="left", fontsize=10, pad=10)
    plt.tight_layout()
    plt.savefig("fig4_4.png", bbox_inches="tight", dpi=200)
    plt.close()


# --------------------------------------------------------------------- MAIN

def main() -> None:
    configure_plot_style()
    header = lambda s: print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")

    raw = load_fame_export()
    sample, attrition = build_sample(raw)
    d = build_features(sample, pre_year=PRE_YEAR)
    d.to_csv("analytical_sample.csv", index=False)

    header("SAMPLE ATTRITION (Table 3.1)")
    print(attrition.to_string(index=False))
    print(f"\nFinal analytical sample: {len(d)}  "
          f"(retention {len(d) / len(raw) * 100:.1f}%)")

    header("TABLE 4.1  DESCRIPTIVE STATISTICS")
    descriptives = d[FEATURES + ["y_growth", "y_margin_pp"]].describe(
        percentiles=[.25, .5, .75]).T[["count", "mean", "std", "25%", "50%", "75%"]]
    print(descriptives.round(3).to_string())

    X = d[FEATURES]
    for outcome, label in [("y_growth", "EBITDA growth (winsorised)"),
                           ("y_margin_pp", "EBITDA margin change (pp)")]:
        header(f"TABLE 4.2  MODEL PERFORMANCE - {label}")
        print(evaluate_models(X, d[outcome]).to_string(index=False))

        header(f"TABLE 4.3  OLS STANDARDISED - {label}")
        table, fit_stats = ols_table(X, d[outcome])
        print(table.to_string())
        print(f"\nR2 = {fit_stats['r2']:.3f}   "
              f"adjusted R2 = {fit_stats['adj_r2']:.3f}   "
              f"F p-value = {fit_stats['f_pvalue']:.4f}")

    header("TABLE 4.4  ROBUSTNESS - PRE-PERIOD BASE YEAR")
    print(robustness_pre_period(sample).to_string(index=False))

    header("SHAP FEATURE IMPORTANCE")
    values, imputed, importance = shap_importance(X, d["y_growth"])
    for feature, value in importance.items():
        print(f"  {FEATURE_LABELS[feature]:26s} {value:.4f}")

    header("FIGURES")
    figure_model_performance(evaluate_models(X, d["y_growth"]))
    figure_shap_importance(importance)
    figure_mean_reversion(d)
    figure_shap_beeswarm(values, imputed)
    print("  fig4_1.png  Out-of-sample predictive performance")
    print("  fig4_2.png  Feature importance")
    print("  fig4_3.png  Mean reversion in earnings")
    print("  fig4_4.png  SHAP value distribution")


if __name__ == "__main__":
    main()
