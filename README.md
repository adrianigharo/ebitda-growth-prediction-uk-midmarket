# Predicting EBITDA Growth in UK Mid-Market Private Equity-Owned Companies

**A Machine Learning Approach**

Adrian Igharo · MSc Artificial Intelligence with Business Strategy · Aston University · September 2026

This repository contains the code and data supporting my MSc dissertation. It reproduces every result, table and figure reported in Chapter 4.

---

## What the study does

Private equity value creation in the UK mid-market has shifted from financial engineering toward operational improvement, which places the critical judgement earlier in the transaction: which businesses have the characteristics that make earnings growth achievable? For privately held companies that judgement rests largely on statutory filings.

This study asks how much of the variation in post-baseline EBITDA performance can actually be predicted from those filings, using a sample of 337 UK companies whose ultimate owner is a private equity firm holding at least 50.01 per cent.

**Headline results**

| | |
|---|---|
| Best model | Random forest, mean out-of-sample R² of 0.134; the only model to beat a naive benchmark under a corrected significance test (p = 0.04) |
| Linear benchmark | OLS at 0.083; the random forest wins in 19 of 20 fold assignments, but the gap is not statistically significant (corrected p = 0.42) |
| Dominant signal | Mean reversion in earnings |
| Counter-signal | Revenue momentum enters positively once reversion is controlled for |
| Asymmetry | Reversion is over six times steeper below the median; the random forest learns the same kink |
| Screening value | Out of sample, the top predicted fifth grew 68.1 per cent against 2.4 per cent for the bottom fifth |
| Ownership | Against 6,448 family-owned companies: the same patterns, but PE companies grew faster and were far more predictable |

Companies in the lowest quartile of pre-period EBITDA growth achieved median subsequent growth of 55.5 per cent, against negative 6.3 per cent for the highest quartile. Screening processes that treat strong recent earnings as a quality signal are, on this evidence, selecting adversely. Both patterns match the established evidence on listed firms: profitability mean-reverts (Fama and French, 2000), and sales growth persists while earnings growth does not (Chan, Karceski and Lakonishok, 2003).

Roughly seven-eighths of the variation remains unexplained by data available in filed accounts, and predictive performance disappears at a one-year horizon. The direction of the relationships is robust to the base year, the horizon, hyperparameter tuning and sector; the accuracy of the models is not.

---

## Files

| File | Description |
|---|---|
| `Appendix_A_Analysis_Notebook.ipynb` | The full analysis with outputs and figures rendered inline. GitHub displays this directly—no download needed. |
| `Appendix_B_Analysis_Pipeline.py` | The same analysis as a documented module, structured for reproducibility. |
| `FAME_export.xlsx` | Raw extract from FAME (Bureau van Dijk). 564 companies. The `Search summary` sheet records the search criteria, FAME data update and export date. |
| `analytical_sample.csv` | Cleaned analytical sample of 337 companies, produced by the pipeline. |
| `FAME_export_family.xlsx` | Comparison group: 9,542 family-owned companies from the same FAME search, with family ownership in place of private equity. |

---

## Running the analysis

```bash
pip install pandas numpy scikit-learn xgboost shap statsmodels scipy matplotlib openpyxl
python Appendix_B_Analysis_Pipeline.py
```

Both FAME exports must sit in the same directory. The script writes `analytical_sample.csv` and the five figures, and prints every table reported in Chapter 4, including the robustness checks in Table 4.5. A full run takes roughly fifteen minutes on a standard laptop, most of it in the nested tuning. The notebook contains the same analysis, with the robustness checks in its Section 9 and the family comparison in Section 11.

Tested on Python 3.12 with scikit-learn 1.8, xgboost 3.4 and shap 0.52.

---

## Method in brief

**Sample.** UK companies with a private equity ultimate owner at 50.01 per cent or above, turnover between £10.2 million and £300 million in the 2022 baseline year. The £10.2 million floor is the Companies Act 2006 threshold below which abbreviated accounts may be filed, making EBITDA underivable.

**Screens.** Eight, reducing 564 companies to 337. The substantive ones are a filing-error screen, deduplication of holding-company stacks, the mid-market turnover bounds, a margin plausibility check, and a requirement that the secondary outcome be computable so both outcomes share a common sample. UK buyouts are executed through layered vehicles, so a single trading business commonly appears as a Topco, Midco and Bidco, each filing consolidated accounts with identical turnover—retaining all three would violate the independence assumption the models rest on.

**Design.** Predictors measured at or before 2022; outcome measured to 2024. Pre-period trend variables use a 2019 base in the primary specification, with a 2020 base retained for robustness. The 2020 base is contaminated by the pandemic: a company whose earnings collapsed in 2020 and recovered by 2022 registers as high-growth when it has merely returned to trend.

**Comparison group.** The same FAME search with the ultimate owner set to named individuals or families. Fourteen companies appearing in both extracts are removed and the same eight screens applied, leaving 6,448. Differences are tested in a pooled regression with interaction terms and the outcome winsorised across both groups; model performance is compared on twenty random family samples of 337.

**Models.** OLS, LASSO, random forest and gradient boosting. Performance is averaged over twenty five-fold cross-validation fold assignments rather than a single split, because at this sample size a single split is unstable—the random forest ranges from 0.073 to 0.164 across seeds.

**Interpretation.** SHAP values, which give directional attribution rather than magnitude alone; a piecewise regression testing whether reversion is asymmetric; partial dependence showing the shape the random forest learns; and out-of-sample quintiles measuring screening value.

**Robustness.** Model differences are tested with the corrected resampled t-test of Nadeau and Bengio (2003), since repeated fold assignments reuse the same companies and a conventional test overstates confidence. Further checks tune hyperparameters by nested cross-validation, shorten the outcome to one year, and add sector indicators. The full robustness run adds roughly ten minutes.

---

## Scope and limitations

FAME records current ownership but carries no acquisition date, so companies sit at unobserved points in their hold periods. The baseline is a fixed calendar year, not a transaction date, and the study measures EBITDA growth among private equity-owned companies over a defined period rather than post-acquisition growth relative to a deal event.

The study was originally designed to test whether firm-level AI adoption signals improve on this baseline. That test could not be conducted: no accessible dataset links AI adoption signals to EBITDA and ownership classification at firm level for UK private companies. Job posting data is published in national aggregate; the ONS holds firm-level adoption data but neither EBITDA nor ownership flags; FAME holds EBITDA and ownership but no adoption measure.

Survivorship bias is present and runs in a known direction—FAME returns only companies still filing. The 2022–24 window also carries a strong negative period effect from inflation, energy costs and interest rates.

The ownership comparison shows associations, not effects: private equity funds choose their targets, so differences between the groups may reflect selection as much as ownership.

---

## Data licensing

The FAME extract is provided for academic assessment purposes. FAME is a Bureau van Dijk product accessed under Aston University licence; onward use is subject to those terms.

---

## Citation

> Igharo, A. (2026) *Predicting EBITDA Growth in UK Mid-Market Private Equity-Owned Companies: A Machine Learning Approach*. MSc dissertation, Aston University.
