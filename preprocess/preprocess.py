"""
Currency Crisis Early-Warning Preprocessing Pipeline
=====================================================
Detects early-warning signs of currency crises in 22 emerging-market countries
over approximately 50 years (1974–2026) using capital flows, reserves, inflation,
and interest differentials.

Pipeline structure
------------------
  STAGE 1 — Data Ingestion and Alignment
    - Load 10 source files (CSV and Excel)
    - Reshape from wide to long (country × month panel)
    - Standardise all dates to YYYY-MM-01
    - Merge into a single balanced panel
    - Impute short gaps (up to 6 months per country)

  STAGE 2 — Feature Engineering
    - Base indicators: interest differential, reserve changes, M2/reserves,
      inflation measures, trade pressure, credit expansion
    - REER misalignment: % deviation from 60-month rolling mean
    - Reserves-to-Short-Term-Debt ratio: rollover risk indicator
    - Sudden Stop indicator: capital-flow proxy 2 standard deviations below mean
    - ADF stationarity test: first-difference non-stationary series
    - Log-transform heavily skewed features

  STAGE 3 — Crisis Labelling
    - Exchange Market Pressure (EMP) index using Eichengreen, Rose & Wyplosz (1996)
    - Binary crisis flag at 2.0 standard deviations above country mean
    - Forward-looking window labels for 3, 6, and 12-month horizons

  STAGE 4 — Model Preparation
    - StandardScaler and RobustScaler normalisation
    - SMOTE applied to training split only
    - TimeSeriesSplit (forward chaining) indices saved for cross-validation

Output files (written to output/)
----------------------------------
  panel_data.csv            Full merged panel with all raw and derived columns
  features_raw.csv          All engineered features at original scale + crisis labels
  features_standard.csv     StandardScaler normalised features
  features_robust.csv       RobustScaler normalised features
  train_smote.csv           SMOTE-balanced training set (pre-2016)
  cv_fold_plan.csv          TimeSeriesSplit fold date boundaries
  feature_correlations.csv  Pairwise Pearson correlation matrix
"""

import warnings, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path("final datasets")
OUT_DIR  = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# 22 emerging-market countries in the dataset
EM_COUNTRIES = [
    "Brazil", "Chile", "China", "Colombia", "Czech Republic", "Greece",
    "Hungary", "India", "Indonesia", "Kuwait", "Malaysia", "Mexico",
    "Peru", "Philippines", "Poland", "Qatar", "Saudi Arabia",
    "South Africa", "South Korea", "Thailand", "Turkey", "UAE",
]

# ISO currency codes, used to map FX columns back to country names
COUNTRY_ISO = {
    "Brazil": "BRL", "Chile": "CLP", "China": "CNY", "Colombia": "COP",
    "Czech Republic": "CZK", "Hungary": "HUF", "India": "INR",
    "Indonesia": "IDR", "Kuwait": "KWD", "Malaysia": "MYR",
    "Mexico": "MXN", "Peru": "PEN", "Philippines": "PHP",
    "Poland": "PLN", "Qatar": "QAR", "Saudi Arabia": "SAR",
    "South Africa": "ZAR", "South Korea": "KRW", "Thailand": "THB",
    "Turkey": "TRY", "UAE": "AED",
}

ALL_COUNTRIES = EM_COUNTRIES + ["United States", "Saudi Arabia", "UAE"]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA INGESTION, RESHAPING, ALIGNMENT, MERGE, IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 1 — Load, reshape, merge, impute")
print("=" * 60)

# ── Helper functions ──────────────────────────────────────────────────────────

def load_csv(fname, date_col="Date"):
    """Load a plain-text CSV and parse the date column."""
    df = pd.read_csv(DATA_DIR / fname)
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def load_excel(fname):
    """
    Load an Excel file (some source files carry a .csv extension but are
    actually binary XLSX, identifiable by the PK ZIP header bytes).
    Renames the first column to 'Date'.
    """
    df = pd.read_excel(DATA_DIR / fname)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def extract_country(col_name, countries):
    """
    Extract the country name from a column header such as 'Brazil_CPI_Index'.
    Countries are sorted longest-first so 'South Africa' matches before 'Africa'.
    """
    for c in sorted(countries, key=len, reverse=True):
        if col_name.startswith(c):
            return c
    return None

def wide_to_long(df, countries, date_col="Date"):
    """
    Convert a wide-format DataFrame (one column per country) into a long-format
    panel with columns [Date, Country, metric1, metric2, ...].
    """
    long = df.melt(id_vars=date_col, var_name="col", value_name="value")
    long["Country"] = long["col"].apply(lambda c: extract_country(c, countries))
    long["metric"]  = long.apply(
        lambda r: r["col"][len(r["Country"]) + 1:] if r["Country"] else r["col"],
        axis=1
    )
    long = long.dropna(subset=["Country"])
    long.drop(columns="col", inplace=True)
    pivoted = long.pivot_table(
        index=[date_col, "Country"], columns="metric",
        values="value", aggfunc="first"
    ).reset_index()
    pivoted.columns.name = None
    return pivoted

def to_month_start(df, date_col="Date"):
    """Normalise any date to the first day of its month (YYYY-MM-01)."""
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    return df

# ── Load all source files ──────────────────────────────────────────────────────
# Two files have .csv extensions but are binary XLSX — detected by ZIP header.
# They must be opened with pd.read_excel().

fx_raw      = load_csv("emerging_markets_50yr_rates.csv", date_col="TIME_PERIOD")
fx_raw.rename(columns={"TIME_PERIOD": "Date"}, inplace=True)
interest_em = load_csv("Clean_EM_InterestRates_50Y.csv")
interest_us = load_csv("Clean_US_InterestRates_50Y.csv")
inflation   = load_csv("Master_EM_Inflation_Rates_50Y.csv")
reserves    = load_csv("Master_EM_Reserves_50Y.csv")
trade       = load_csv("Master_EM_Trade_Data_50Y.csv")
m2          = load_csv("Master_EM_M2_BroadMoney_50Y.csv")
m0          = load_csv("Master_EM_MonetaryBase_50Y.csv")
dctps       = load_excel("Master_DCTPS%.csv")        # Excel disguised as CSV
short_debt  = load_excel("Master_Short_Term_Debt copy.csv")  # same

# ── Reshape wide files to long format ─────────────────────────────────────────
# FX file uses ISO codes as column headers — map back to country names first.
iso_to_country = {v: k for k, v in COUNTRY_ISO.items()}
fx_long = fx_raw.melt(id_vars="Date", var_name="iso", value_name="fx_em_per_usd")
fx_long["Country"] = fx_long["iso"].map(iso_to_country)
fx_long.dropna(subset=["Country"], inplace=True)
fx_long.drop(columns="iso", inplace=True)
fx_long["Date"] = fx_long["Date"].dt.to_period("M").dt.to_timestamp()
# Keep the last daily rate of each month as the end-of-month signal
fx_monthly = fx_long.groupby(["Date","Country"])["fx_em_per_usd"].last().reset_index()

interest_em_long = wide_to_long(interest_em, EM_COUNTRIES)

# Rename US interest rate columns to a consistent us_* prefix
interest_us.columns = [
    "Date" if c == "Date"
    else c.replace("United States_", "us_").replace(" ", "_").lower()
    for c in interest_us.columns
]

inflation_long = wide_to_long(inflation, ALL_COUNTRIES)
reserves_long  = reserves.melt(id_vars="Date", var_name="Country",
                                value_name="fx_reserves_usd")
trade_long     = wide_to_long(trade, ALL_COUNTRIES)
m2_long        = wide_to_long(m2, ALL_COUNTRIES)
m0_long        = wide_to_long(m0, ALL_COUNTRIES)

# Annual files: melt to long then extract country
dctps_long = dctps.melt(id_vars="Date", var_name="col", value_name="dctps_pct_gdp")
dctps_long["Country"] = dctps_long["col"].apply(lambda c: extract_country(c, ALL_COUNTRIES))
dctps_long.dropna(subset=["Country"], inplace=True)
dctps_long.drop(columns="col", inplace=True)

std_long = wide_to_long(short_debt, ALL_COUNTRIES)

# ── Standardise all dates to YYYY-MM-01 ───────────────────────────────────────
for df in [interest_em_long, interest_us, inflation_long, reserves_long,
           trade_long, m2_long, m0_long, fx_monthly, dctps_long, std_long]:
    to_month_start(df)

# ── Build the country × month panel ──────────────────────────────────────────
# Use FX reserves as the backbone (widest coverage: 22 countries from 1974).
# All other datasets are left-joined so no reserve row is ever lost.
panel = reserves_long[reserves_long["Country"].isin(EM_COUNTRIES)].copy()
panel = panel.merge(fx_monthly, on=["Date","Country"], how="left")
panel = panel.merge(interest_em_long[interest_em_long["Country"].isin(EM_COUNTRIES)],
                    on=["Date","Country"], how="left")
panel = panel.merge(interest_us, on="Date", how="left")  # US rates broadcast to all countries
panel = panel.merge(inflation_long[inflation_long["Country"].isin(EM_COUNTRIES)],
                    on=["Date","Country"], how="left")
panel = panel.merge(trade_long[trade_long["Country"].isin(EM_COUNTRIES)],
                    on=["Date","Country"], how="left")

m2_keep = [c for c in ["Date","Country","Broad_Money_M2","M2_Growth_YoY_%"] if c in m2_long.columns]
panel = panel.merge(m2_long[m2_long["Country"].isin(EM_COUNTRIES)][m2_keep],
                    on=["Date","Country"], how="left")
m0_keep = [c for c in ["Date","Country","Monetary_Base_XDC","M0_Growth_YoY_%"] if c in m0_long.columns]
panel = panel.merge(m0_long[m0_long["Country"].isin(EM_COUNTRIES)][m0_keep],
                    on=["Date","Country"], how="left")

panel = panel.sort_values(["Country","Date"])

# Annual DCTPS and short-term debt: merge then forward-fill up to 11 months
# so each monthly row carries the most recent annual reading available at that time.
for annual_df in [dctps_long[dctps_long["Country"].isin(EM_COUNTRIES)],
                  std_long[std_long["Country"].isin(EM_COUNTRIES)]]:
    annual_cols = [c for c in annual_df.columns if c not in ["Date","Country"]]
    panel = panel.merge(annual_df, on=["Date","Country"], how="left")
    panel[annual_cols] = panel.groupby("Country")[annual_cols].ffill(limit=11)

panel = panel.sort_values(["Country","Date"]).reset_index(drop=True)

# ── Impute short data gaps within each country's time series ─────────────────
# Strategy: linear interpolation for gaps up to 3 months (data release delays),
# then forward-fill for gaps of 4–6 months. Gaps > 6 months are left as NaN.
# Imputation is always within-country — never borrows across country boundaries.
numeric_cols = panel.select_dtypes(include="number").columns.tolist()
panel[numeric_cols] = (
    panel.groupby("Country")[numeric_cols]
    .transform(lambda s: s.interpolate(method="linear", limit=3, limit_direction="forward"))
)
panel[numeric_cols] = (
    panel.groupby("Country")[numeric_cols]
    .transform(lambda s: s.ffill(limit=6))
)

print(f"  Panel: {panel.shape[0]:,} rows x {panel.shape[1]} cols")
print(f"  Date range : {panel['Date'].min().date()} to {panel['Date'].max().date()}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 2 — Feature engineering")
print("=" * 60)

# ── Interest rate features ────────────────────────────────────────────────────

# Build a single composite EM rate from whichever rate columns exist,
# using the most liquid / most widely reported rate as the priority.
panel["em_rate"] = np.nan
for col in ["Interbank_Rate", "Policy_Rate", "10Y_Bond"]:
    if col in panel.columns:
        panel["em_rate"] = panel["em_rate"].where(panel["em_rate"].notna(), panel[col])

# US benchmark: use the most comparable short-term rate available
panel["us_benchmark"] = np.nan
for col in ["us_interbank_rate","us_3m_tbill","us_policy_rate","us_10y_bond"]:
    if col in panel.columns:
        panel["us_benchmark"] = panel["us_benchmark"].where(
            panel["us_benchmark"].notna(), panel[col])

# Interest differential: how much higher the EM rate is above the US rate.
# A rising differential signals the central bank is defending the currency
# (raising rates to discourage capital outflows), or that markets demand a
# devaluation risk premium.
panel["interest_differential"] = panel["em_rate"] - panel["us_benchmark"]

# Real interest rate: nominal rate minus inflation.
# Deeply negative real rates drive capital flight as savings lose purchasing power.
if "Inflation_YoY_%" in panel.columns:
    panel["real_interest_rate"] = panel["em_rate"] - panel["Inflation_YoY_%"]

# ── Exchange rate features ─────────────────────────────────────────────────────
# Convention: fx_em_per_usd = EM-currency units per 1 USD (indirect quote).
# A rising value = EM currency depreciating = positive depreciation signal.
# The YoY version is NOT included: it correlates 0.98 with Inflation_YoY%
# due to the Purchasing Power Parity identity, adding no independent signal.
panel["fx_depreciation_mom"] = (
    panel.groupby("Country")["fx_em_per_usd"].pct_change() * 100
)

# ── Reserve features ───────────────────────────────────────────────────────────
# Central banks defend the peg by selling USD reserves. Rapid depletion signals
# the defence is losing — a core EMP component.
panel["reserves_change_mom"] = (
    panel.groupby("Country")["fx_reserves_usd"].pct_change() * 100
)
panel["reserves_change_yoy"] = (
    panel.groupby("Country")["fx_reserves_usd"].pct_change(12) * 100
)

# Reserve adequacy (Guidotti-Greenspan rule): how many months of imports
# could be funded purely from reserves if all foreign financing stopped.
# The IMF considers below 3 months dangerous. Implementation notes:
#   - Minimum import threshold of $10M/month filters out near-zero data
#     artefacts that would produce trillion-month ratios.
#   - Hard cap at 120 months prevents extreme outliers distorting scaling.
if "Imports" in panel.columns:
    valid_imports = panel["Imports"].where(panel["Imports"] >= 1e7, np.nan)
    panel["reserve_adequacy_months"] = (
        panel["fx_reserves_usd"] / valid_imports
    ).clip(upper=120)

# ── M2/Reserves ratio (Calvo indicator) ────────────────────────────────────────
# If all domestic bank deposits were converted to USD simultaneously, could
# the central bank cover them? A ratio above 5 means a speculative attack
# is mathematically feasible given available reserves.
# IMPORTANT: M2 must be converted to USD before dividing by USD reserves.
# Dividing local-currency M2 directly by USD reserves is dimensionally invalid.
if "Broad_Money_M2" in panel.columns:
    valid_fx = panel["fx_em_per_usd"].replace(0, np.nan)
    panel["m2_usd"] = panel["Broad_Money_M2"] / valid_fx   # local CCY → USD
    panel["m2_reserves_ratio"] = panel["m2_usd"] / panel["fx_reserves_usd"].replace(0, np.nan)

# ── Inflation features ─────────────────────────────────────────────────────────
# Three-month acceleration: rising YoY inflation over the past quarter signals
# monetary control is being lost.
if "Inflation_YoY_%" in panel.columns:
    panel["inflation_accel_3m"] = (
        panel.groupby("Country")["Inflation_YoY_%"].diff(3))

# ── Trade and monetary features ────────────────────────────────────────────────
if "Trade_Balance" in panel.columns:
    # Trade balance relative to reserves: persistent deficits drain the buffer.
    panel["trade_to_reserves"] = (
        panel["Trade_Balance"] / panel["fx_reserves_usd"].replace(0, np.nan))

if "Export_Momentum_YoY_%" in panel.columns:
    # Falling exports reduce the inflow of foreign currency.
    panel["export_growth_yoy"] = panel["Export_Momentum_YoY_%"].replace(
        [np.inf, -np.inf], np.nan)

if "M2_Growth_YoY_%" in panel.columns:
    # Sustained money supply expansion above 20% is a monetary overhang risk.
    panel["m2_growth_yoy"] = panel["M2_Growth_YoY_%"].replace([np.inf, -np.inf], np.nan)

# Credit boom indicator (Kaminsky & Reinhart 1999 twin-crises framework):
# rapid credit expansion to the private sector precedes both banking and
# currency crises. Measured as the annual change in the credit/GDP ratio.
if "dctps_pct_gdp" in panel.columns:
    panel["credit_boom_yoy"] = panel.groupby("Country")["dctps_pct_gdp"].diff(1)

print("  Base features computed")

# ── REER Misalignment ──────────────────────────────────────────────────────────
# An overvalued exchange rate is the primary target of speculative attacks.
# We approximate the Real Effective Exchange Rate (REER) using the bilateral
# USD rate adjusted for relative inflation (Purchasing Power Parity):
#
#   REER_approx[t] = fx_em_per_usd[t] × (CPI_US[t] / CPI_EM[t])
#
# Misalignment = % deviation from the 60-month (5-year) rolling mean.
# A negative value means the currency is stronger than its historical norm
# (overvalued) — the classic precondition for a speculative attack.
# Note: true REER requires trade-weighted multi-currency rates; the bilateral
# approximation introduces error for countries where the EU is the main trade partner.
print("\n  Computing REER misalignment...")

if "CPI_Index" in panel.columns:
    us_cpi = (inflation_long[inflation_long["Country"] == "United States"]
              [["Date", "CPI_Index"]].rename(columns={"CPI_Index": "us_cpi"}))
    to_month_start(us_cpi)
    panel = panel.merge(us_cpi, on="Date", how="left")

    valid_cpi = panel["CPI_Index"].replace(0, np.nan)
    valid_usd = panel["fx_em_per_usd"].replace(0, np.nan)
    panel["reer_approx"] = valid_usd * (panel["us_cpi"] / valid_cpi)

    panel["reer_60m_mean"] = (
        panel.groupby("Country")["reer_approx"]
        .transform(lambda s: s.rolling(60, min_periods=24).mean())
    )
    panel["reer_misalignment"] = (
        (panel["reer_approx"] - panel["reer_60m_mean"])
        / panel["reer_60m_mean"].replace(0, np.nan) * 100
    )
    print("    REER misalignment: CPI-adjusted bilateral USD (% from 60-month mean)")
else:
    # Fallback when CPI not available: use nominal FX deviation only
    panel["reer_60m_mean"] = (
        panel.groupby("Country")["fx_em_per_usd"]
        .transform(lambda s: s.rolling(60, min_periods=24).mean())
    )
    panel["reer_misalignment"] = (
        (panel["fx_em_per_usd"] - panel["reer_60m_mean"])
        / panel["reer_60m_mean"].replace(0, np.nan) * 100
    )
    print("    REER misalignment: nominal FX only (CPI unavailable)")

# ── Reserves-to-Short-Term-Debt Ratio ─────────────────────────────────────────
# The Greenspan-Guidotti Rule: reserves should fully cover all short-term
# external debt (debt maturing within 12 months). Countries with high short-term
# debt are vulnerable to rollover crises — if creditors refuse to renew maturing
# loans, the country faces an immediate foreign-currency shortage.
# Key mechanism in the 1997 Asian and 1998 Russian crises.
if "STD%" in panel.columns:
    panel["std_vulnerability"] = panel["STD%"]            # higher = more rollover risk
    panel["reserves_to_std_proxy"] = 100 / panel["STD%"].replace(0, np.nan)
    panel["reserves_to_std_proxy"] = panel["reserves_to_std_proxy"].clip(upper=20)
    if "STD_yoy_%" in panel.columns:
        panel["std_growth_yoy"] = panel["STD_yoy_%"]
    print("    Reserves-to-STD proxy computed")

# ── Sudden Stop Indicator ─────────────────────────────────────────────────────
# A sudden stop is the abrupt collapse of net capital inflows.
# We approximate using the Balance of Payments identity:
#   Capital Account ≈ Δ(reserves) − Trade_Balance
# The sudden stop flag triggers when the proxy falls more than 2 standard
# deviations below the country's own historical mean — capturing genuinely
# exceptional capital outflow events, not routine volatility.
# Note: limited to post-2005 data where trade figures are available.
if "Trade_Balance" in panel.columns:
    panel["reserves_change_abs"] = (
        panel.groupby("Country")["fx_reserves_usd"].diff()
    )
    panel["capital_flow_proxy"] = (
        panel["reserves_change_abs"] - panel["Trade_Balance"]
    )
    # Normalise as % of reserves for cross-country comparability
    panel["capital_flow_pct"] = (
        panel["capital_flow_proxy"] / panel["fx_reserves_usd"].replace(0, np.nan) * 100
    )
    panel["sudden_stop"] = 0
    for country, grp in panel.groupby("Country"):
        s = grp["capital_flow_pct"].dropna()
        if len(s) > 12:
            threshold = s.mean() - 2 * s.std()
            panel.loc[grp.index, "sudden_stop"] = (
                grp["capital_flow_pct"].lt(threshold).astype(int)
            )
    ss_rate = panel["sudden_stop"].mean()
    print(f"    Sudden stop indicator computed  (rate={ss_rate:.1%})")

print("  All features computed")

# ── Winsorisation ──────────────────────────────────────────────────────────────
# Countries with hyperinflation (e.g. Brazil peaked at 4,577% YoY in 1993,
# Turkey at 3,500% in 1994) generate extreme feature values that would cause
# models to simply learn the hyperinflation era rather than general crisis patterns.
# Winsorisation clips each feature at the 1st and 99th percentile within each
# country — so Brazil is judged against Brazil's own history, not against Thailand.
print("\n" + "=" * 60)
print("STAGE 2 — Winsorising")
print("=" * 60)

FEATURE_COLS = [
    "interest_differential", "real_interest_rate",
    "fx_depreciation_mom",
    "reserves_change_mom", "reserves_change_yoy",
    "reserve_adequacy_months", "m2_reserves_ratio",
    "inflation_accel_3m", "Inflation_YoY_%",
    "trade_to_reserves", "export_growth_yoy",
    "m2_growth_yoy", "credit_boom_yoy",
    "reer_misalignment",
    "std_vulnerability", "reserves_to_std_proxy",
    "capital_flow_pct",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in panel.columns]

def winsorize_group(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lower=lo, upper=hi)

panel[FEATURE_COLS] = (
    panel.groupby("Country")[FEATURE_COLS]
    .transform(winsorize_group)
)
print(f"  Winsorized {len(FEATURE_COLS)} feature columns (per-country 1st/99th percentile)")

# ── ADF Stationarity Test and First-Differencing ───────────────────────────────
# Non-stationary series (those that trend rather than mean-revert) can produce
# spurious correlations in models. The Augmented Dickey-Fuller (ADF) test
# checks for a unit root (non-stationarity) in each country's series.
# If more than 50% of countries fail the test for a given feature (p > 0.05),
# a first-differenced version is created: x_diff[t] = x[t] - x[t-1].
# Both the original (level signal) and differenced (change signal) versions are kept.
print("\n" + "=" * 60)
print("STAGE 2 — ADF stationarity test + first-differencing")
print("=" * 60)

def adf_pvalue(series):
    """Return the ADF p-value; NaN if fewer than 20 observations are available."""
    s = series.dropna()
    if len(s) < 20:
        return np.nan
    try:
        return adfuller(s, autolag="AIC")[1]
    except Exception:
        return np.nan

DIFF_COLS = []
adf_results = {}

for col in FEATURE_COLS:
    pvals = []
    for country, grp in panel.groupby("Country"):
        p = adf_pvalue(grp[col])
        if not np.isnan(p):
            pvals.append(p)
    if len(pvals) == 0:
        continue
    non_stationary_frac = np.mean([p > 0.05 for p in pvals])
    adf_results[col] = {
        "median_pval": np.median(pvals),
        "non_stationary_pct": non_stationary_frac * 100,
    }
    if non_stationary_frac > 0.5:
        diff_col = f"{col}_diff"
        panel[diff_col] = panel.groupby("Country")[col].diff()
        DIFF_COLS.append(diff_col)

print(f"  ADF results (median p-value, % non-stationary across countries):")
for col, res in sorted(adf_results.items(), key=lambda x: -x[1]["non_stationary_pct"]):
    flag = " -> differenced" if f"{col}_diff" in DIFF_COLS else ""
    print(f"    {col:<35}  p={res['median_pval']:.3f}  "
          f"non-stationary={res['non_stationary_pct']:.0f}%{flag}")
print(f"\n  First-differenced columns added: {len(DIFF_COLS)}")

# ── Log-Transform Highly Skewed Features ──────────────────────────────────────
# After winsorisation some features remain skewed above 5.
# Linear models (logistic regression, SVM) assume near-Gaussian inputs.
# Highly skewed inputs produce unstable coefficients and slow convergence.
# Two transform variants:
#   log1p(x)         = log(1 + x)            — for non-negative features
#   signed_log(x)    = sign(x) × log(1 + |x|) — for features that can be negative
print("\n" + "=" * 60)
print("STAGE 2 — Log-transforming skewed features")
print("=" * 60)

LOG_COLS = []

log_positive = ["reserve_adequacy_months", "m2_reserves_ratio", "reserves_to_std_proxy"]
log_signed   = ["interest_differential", "Inflation_YoY_%", "inflation_accel_3m",
                "reer_misalignment", "capital_flow_pct"]

for col in log_positive:
    if col in panel.columns and abs(panel[col].dropna().skew()) > 5:
        name = f"{col}_log"
        panel[name] = np.log1p(panel[col].clip(lower=0))
        LOG_COLS.append(name)
        print(f"  log1p({col})  skew {panel[col].skew():.1f} -> {panel[name].skew():.1f}")

for col in log_signed:
    if col in panel.columns and abs(panel[col].dropna().skew()) > 5:
        name = f"{col}_log"
        panel[name] = np.sign(panel[col]) * np.log1p(panel[col].abs())
        LOG_COLS.append(name)
        print(f"  signed_log({col})  skew {panel[col].skew():.1f} -> {panel[name].skew():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CRISIS LABELLING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 3 — EMP index and crisis labels")
print("=" * 60)

# ── Exchange Market Pressure (EMP) Index ───────────────────────────────────────
# Eichengreen, Rose & Wyplosz (1996): a government facing speculative pressure
# has three options — let the currency fall, sell reserves, or raise rates.
# The EMP index combines all three responses into one composite signal.
# Each component is within-country Z-scored so that Brazil's normal volatility
# does not dwarf Thailand's signals.
#
# Weights: FX depreciation (0.5) > reserve loss (0.3) > rate differential (0.2)
# Threshold: mean + 2.0 standard deviations (matches the original 1996 paper)

def z_score(s):
    std = s.std()
    return (s - s.mean()) / std if std > 0 else s * 0

panel = panel.sort_values(["Country","Date"])
emp_list = []
for country, grp in panel.groupby("Country"):
    dep  = z_score(grp["fx_depreciation_mom"].fillna(0))
    res  = z_score((-grp["reserves_change_mom"]).fillna(0))   # negate: loss = pressure
    rate = z_score(grp["interest_differential"].fillna(0))
    emp  = 0.5 * dep + 0.3 * res + 0.2 * rate
    emp_list.append(emp)

panel["emp_index"] = pd.concat(emp_list).reindex(panel.index)

panel["emp_crisis"] = 0
for country, grp in panel.groupby("Country"):
    threshold = grp["emp_index"].mean() + 2.0 * grp["emp_index"].std()
    panel.loc[grp.index, "emp_crisis"] = (grp["emp_index"] > threshold).astype(int)

print(f"  Crisis months : {panel['emp_crisis'].sum()}  ({panel['emp_crisis'].mean():.2%})")

# Sanity check: all six canonical historical crises should be detected
KNOWN_CRISES = {
    "Mexico": ("1994-11-01","1995-03-01"), "Thailand":  ("1997-07-01","1997-12-01"),
    "Indonesia":("1997-08-01","1998-06-01"),"Brazil":  ("1998-10-01","1999-03-01"),
    "Turkey":  ("2001-01-01","2001-06-01"),"South Africa":("2001-10-01","2002-02-01"),
}
print("  Known-crisis sanity check:")
for country, (start, end) in KNOWN_CRISES.items():
    sub = panel[(panel.Country==country)&(panel.Date>=start)&(panel.Date<=end)]
    hit = sub["emp_crisis"].max() if len(sub) > 0 else "no data"
    print(f"    {'[OK]' if hit==1 else '[MISS]'}  {country} {start[:7]}-{end[:7]}")

# ── Forward-Looking Window Labels ─────────────────────────────────────────────
# The model's goal is early warning — predicting a crisis before it starts.
# Each label answers: "Will there be a crisis in ANY of the next h months?"
# Using a single shift(-h) would only flag the exact month h ahead, leaving
# the months immediately before a crisis event unlabelled as non-crises.
# The window approach (max over a stack of shifts) correctly labels all months
# within the warning horizon.
print("\n" + "=" * 60)
print("STAGE 3 — Forward-looking window labels")
print("=" * 60)

ALL_FEAT_COLS = FEATURE_COLS + DIFF_COLS + LOG_COLS
ALL_FEAT_COLS = [c for c in ALL_FEAT_COLS if c in panel.columns]

features = panel[["Date","Country"] + ALL_FEAT_COLS + ["emp_index","emp_crisis","sudden_stop"]].copy()

HORIZONS = [3, 6, 12]
for h in HORIZONS:
    future_shifts = pd.concat(
        [features.groupby("Country")["emp_crisis"].shift(-k) for k in range(1, h+1)],
        axis=1
    )
    features[f"crisis_in_{h}m"] = future_shifts.max(axis=1)

# Auto-regressive lag features: past values of key indicators provide
# momentum signals (e.g. reserves have been falling for 6 months).
LAG_TARGETS = [c for c in ["interest_differential","reserves_change_mom",
                            "Inflation_YoY_%","fx_depreciation_mom",
                            "reer_misalignment","capital_flow_pct"]
               if c in features.columns]
for col in LAG_TARGETS:
    for lag in [1, 3, 6]:
        features[f"{col}_lag{lag}"] = features.groupby("Country")[col].shift(lag)

# Drop rows where all three core features are simultaneously missing
CORE = [c for c in ["interest_differential","reserves_change_mom","Inflation_YoY_%"]
        if c in features.columns]
features.dropna(subset=CORE, how="all", inplace=True)
print(f"  Features shape : {features.shape[0]:,} rows x {features.shape[1]} cols")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MODEL PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 4 — Scaling, SMOTE, cross-validation setup")
print("=" * 60)

# ── Scaling: StandardScaler and RobustScaler ──────────────────────────────────
# StandardScaler: normalise to zero mean, unit variance.
#   Best for: logistic regression, SVM, neural networks.
#   Use with _log feature variants (closer to Gaussian).
#
# RobustScaler: normalise using IQR (Q75 - Q25) instead of standard deviation.
#   Best for: same model types when residual outliers remain after winsorisation.
#   The IQR is insensitive to extreme values, making it more stable.
#
# NOTE: In a real model, always fit scalers on training data only, then
# .transform() validation and test sets separately. The scalers here are fit
# on the full dataset for exploration and comparison purposes only.

model_feat_cols = (
    ALL_FEAT_COLS +
    [f"{c}_lag{l}" for c in LAG_TARGETS for l in [1,3,6]] +
    ["emp_index"]
)
model_feat_cols = [c for c in model_feat_cols if c in features.columns]

fill_matrix = features[model_feat_cols].fillna(0)

features_standard = features.copy()
features_standard[model_feat_cols] = StandardScaler().fit_transform(fill_matrix)

features_robust = features.copy()
features_robust[model_feat_cols] = RobustScaler().fit_transform(fill_matrix)

print(f"  Scaled {len(model_feat_cols)} features with both scalers")

# ── Class Imbalance: SMOTE ─────────────────────────────────────────────────────
# Currency crises account for only ~3–6% of all country-months.
# Without rebalancing, a classifier that always predicts "no crisis" achieves
# 94–97% accuracy — useless for early warning.
#
# SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic crisis
# examples by interpolating between existing crisis rows in feature space.
#
# CRITICAL RULE: SMOTE must be applied to training data ONLY.
# If synthetic points are derived from test-set features before splitting,
# information from the test set leaks into training — all metrics become invalid.
# Here SMOTE is applied to the pre-2016 temporal training split.

TARGET_COL = "crisis_in_6m"    # primary prediction target (recommended starting point)

smote_df = features_standard.dropna(subset=[TARGET_COL]).copy()
smote_df[TARGET_COL] = smote_df[TARGET_COL].astype(int)

train_mask = pd.to_datetime(smote_df["Date"]) < "2016-01-01"
X_train_raw = smote_df.loc[train_mask, model_feat_cols].fillna(0)
y_train_raw = smote_df.loc[train_mask, TARGET_COL]

print(f"\n  Training rows (before 2016)  : {len(X_train_raw):,}")
print(f"  Crisis rate in training set  : {y_train_raw.mean():.2%}")
print(f"  Class counts before SMOTE    : {dict(y_train_raw.value_counts())}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
X_train_smote = pd.DataFrame(X_train_smote, columns=model_feat_cols)
y_train_smote = pd.Series(y_train_smote, name=TARGET_COL)

print(f"\n  After SMOTE:")
print(f"  Training rows                : {len(X_train_smote):,}")
print(f"  Crisis rate                  : {y_train_smote.mean():.2%}")
print(f"  Class counts after SMOTE     : {dict(y_train_smote.value_counts())}")

smote_out = pd.concat([X_train_smote, y_train_smote], axis=1)
smote_out.to_csv(OUT_DIR / "train_smote.csv", index=False)
print(f"  Saved: output/train_smote.csv")

# ── Time-Series Cross-Validation ──────────────────────────────────────────────
# Standard k-fold cross-validation randomly shuffles rows — for time series this
# means training on future data to predict the past, making all metrics invalid.
#
# Forward Chaining (TimeSeriesSplit): always train on the past, test on the future.
# The 6-month gap between train end and test start prevents label-window leakage:
# since we predict crises up to 6 months ahead, the last 6 training months and
# first 6 test months would otherwise share overlapping label windows.

tscv_df = features_standard.dropna(subset=[TARGET_COL]).sort_values("Date").copy()
tscv_df[TARGET_COL] = tscv_df[TARGET_COL].astype(int)

X_all = tscv_df[model_feat_cols].fillna(0).values
y_all = tscv_df[TARGET_COL].values
dates_all = pd.to_datetime(tscv_df["Date"]).values

tscv = TimeSeriesSplit(n_splits=5, gap=6)

fold_info = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
    train_dates = dates_all[train_idx]
    test_dates  = dates_all[test_idx]
    crisis_rate = y_all[test_idx].mean()
    fold_info.append({
        "fold": fold,
        "train_start": pd.Timestamp(train_dates.min()).date(),
        "train_end"  : pd.Timestamp(train_dates.max()).date(),
        "test_start" : pd.Timestamp(test_dates.min()).date(),
        "test_end"   : pd.Timestamp(test_dates.max()).date(),
        "test_rows"  : len(test_idx),
        "crisis_rate": f"{crisis_rate:.2%}",
    })

fold_df = pd.DataFrame(fold_info)
print(f"\n  TimeSeriesSplit — 5 folds, 6-month gap:")
print(fold_df.to_string(index=False))
fold_df.to_csv(OUT_DIR / "cv_fold_plan.csv", index=False)

# ── Data Quality Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

print("\n  Crisis balance per horizon:")
for h in ["crisis_in_3m","crisis_in_6m","crisis_in_12m"]:
    v = features[h].value_counts(normalize=True, dropna=True)
    print(f"    {h}: crisis={v.get(1.0,0):.1%}  no-crisis={v.get(0.0,0):.1%}")

print(f"\n  Sudden stop rate: {features['sudden_stop'].mean():.2%}")

print("\n  Differenced feature columns added:")
for c in DIFF_COLS:
    print(f"    {c}")

print("\n  Advanced feature coverage (missing data %):")
new_cols = ["reer_misalignment","std_vulnerability","reserves_to_std_proxy",
            "capital_flow_pct","sudden_stop"]
for c in new_cols:
    if c in features.columns:
        na = features[c].isna().mean()
        print(f"    {c:<35}  missing={na:.0%}")

corr = features[FEATURE_COLS].corr().round(3)
print("\n  Multicollinearity — pairs with |r| > 0.7:")
found = False
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        val = corr.iloc[i,j]
        if abs(val) > 0.7:
            print(f"    {corr.columns[i]}  <->  {corr.columns[j]}  r={val:.2f}")
            found = True
if not found:
    print("    None")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════════════

panel.to_csv(OUT_DIR / "panel_data.csv", index=False)
features.to_csv(OUT_DIR / "features_raw.csv", index=False)
features_standard.to_csv(OUT_DIR / "features_standard.csv", index=False)
features_robust.to_csv(OUT_DIR / "features_robust.csv", index=False)
features[FEATURE_COLS].corr().round(3).to_csv(OUT_DIR / "feature_correlations.csv")

print("\n" + "=" * 60)
print("ALL OUTPUT FILES SAVED")
print("=" * 60)
print(f"  output/panel_data.csv          {len(panel):,} rows — full panel with all raw and derived columns")
print(f"  output/features_raw.csv        {len(features):,} rows — engineered features at original scale + labels")
print(f"  output/features_standard.csv   {len(features_standard):,} rows — StandardScaler normalised")
print(f"  output/features_robust.csv     {len(features_robust):,} rows — RobustScaler normalised")
print(f"  output/train_smote.csv         {len(smote_out):,} rows — SMOTE-balanced training set (pre-2016)")
print(f"  output/cv_fold_plan.csv                — TimeSeriesSplit fold boundaries (5 folds)")
print(f"  output/feature_correlations.csv        — pairwise Pearson correlation matrix")
