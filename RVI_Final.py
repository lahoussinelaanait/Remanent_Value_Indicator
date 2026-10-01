# -*- coding: utf-8 -*-
"""
RVI replication code for:
"The Remanent Value Indicator (RVI): An Empirical Instrument for Detecting
the Realization of Remanent Value in Cognitive Capitalism (Second Version)"

Final correction for Qualcomm / Oracle:
- standard deviation uses population convention ddof=0
- wide Qualcomm / Oracle decomposition table bug fixed
- no 10-consecutive-years filter
- no redundant documented purge
- strict 5/5 per 5-year window
- positive mean margin per 5-year window
- economic consistency filter at 60%
- ASML / TSMC / Huawei reclassified to INDUS
- Toyota / Mercedes diagnostic included
"""

import os
import numpy as np
import pandas as pd

from google.colab import drive

# ============================================================================
# 1. MOUNT DRIVE
# ============================================================================
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")

# ============================================================================
# 2. PATHS
# ============================================================================
RAW_PANEL_PATH = "/content/drive/MyDrive/verification/Panel_Work_Final_RVI_SCOREBOARD.csv"
PURIFIED_PANEL_PATH = "/content/drive/MyDrive/verification/RVI_SCOREBOARD_DEFINITIF_FINAL_EUR.csv"

OUTPUT_DIR = "/content/drive/MyDrive/rvi4_revised_replication"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 3. PARAMETERS
# ============================================================================

# Four disjoint 5-year periods used in rvi4_revised
PERIODS = {
    "2005-2009": (2005, 2009),
    "2010-2014": (2010, 2014),
    "2015-2019": (2015, 2019),
    "2020-2024": (2020, 2024),
}

PERIOD_ORDER = list(PERIODS.keys())

# Sector order used in the article tables
SECTOR_ORDER = ["PHARMA", "TECH", "INDUS", "AUTO"]

# Block mapping: 2-cycle vs 1-cycle
BLOCK_MAP = {
    "PHARMA": "2-cycle",
    "TECH": "2-cycle",
    "INDUS": "1-cycle",
    "AUTO": "1-cycle",
}

# RVI parameters
RD0_BILLION = 1.0

# IMPORTANT:
# If rnd_24 is expressed in MILLION euros, use 1e-3 to convert to BILLION euros.
# If rnd_24 is expressed in EURO, use 1e-9.
# If rnd_24 is already in BILLION euros, use 1.0.
RD_UNIT_TO_BILLION = 1e-3

# Standard deviation convention for RVI.
# ddof=0 -> population standard deviation.
# ddof=1 -> sample standard deviation.
# For Qualcomm / Oracle alignment with rvi4_revised, use ddof=0.
STD_DDOF = 0

# Economic consistency filter:
# exclude firm-year observations where abs(margin) > 0.60 or abs(R&D intensity) > 0.60
MAX_ABS_MARGIN = 0.60
MAX_ABS_RND_INTENSITY = 0.60

# Strict 5-out-of-5 rule per disjoint window
REQUIRE_STRICT_5_OUT_OF_5 = True

# Profitability baseline: average operating margin strictly positive in the window
REQUIRE_POSITIVE_MEAN_MARGIN = True

# Temporal depth filter: DISABLED
# The 5/5 rule and positive mean margin per window are sufficient.
MIN_CONSECUTIVE_YEARS = None

# Documented purge: DISABLED
# The purified whitelist RVI_SCOREBOARD_DEFINITIF_FINAL_EUR.csv is treated as final.
APPLY_DOCUMENTED_PURGE = False

# Top 100 global cohort
TOP_N_GLOBAL = 100

# Firms used in the Qualcomm / Oracle diagnostic table
DIAGNOSTIC_FIRMS = ["QUALCOMM", "ORACLE"]

# Reclassification acted in rvi4_revised:
# ASML Holding, Taiwan Semiconductor (TSMC), Huawei Technologies -> INDUS
RECLASSIFY_TO_INDUS_PATTERNS = [
    "ASML",
    "TAIWAN SEMICONDUCTOR",
    "TSMC",
    "HUAWEI",
]

# Optional documented purge parameters.
# Kept here only if APPLY_DOCUMENTED_PURGE is manually set to True.
TIRES_ICB4 = {"Tires"}
PURGE_COMPANY_EXACT = {"37"}
PURGE_NACE_EXACT = {"7010"}

# Expected columns
COMPANY_COL = "company"
YEAR_COL = "year"
SECTOR_COL = "Sector_Final"

NETSALES_COL = "netsales_24"
RND_COL = "rnd_24"
PROFIT_COL = "profit_24"

# Optional columns used only for purge
ICB4_COL = "icb4_name"
NACE_COL = "nace_rev2_new"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

# ============================================================================
# 4. HELPERS
# ============================================================================

def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_upper(x) -> str:
    return clean_str(x).upper()


def clean_nace(x) -> str:
    s = clean_str(x)
    if "." in s:
        s = s.split(".")[0]
    return s.strip()


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def format_presence(n: int, N: int) -> str:
    rate = safe_rate(n, N)
    if pd.isna(rate):
        return f"{n}/{N} = NA"
    return f"{n}/{N} = {rate:.3f}"


def contains_any(series: pd.Series, patterns) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for p in patterns:
        mask |= series.str.contains(str(p).upper(), na=False, regex=False)
    return mask


def std_rvi(x: pd.Series) -> float:
    """
    Standard deviation used in the RVI formula.
    Default: population standard deviation, ddof=0.
    """
    return x.std(ddof=STD_DDOF)


def save_csv_and_tex(df_out: pd.DataFrame, name: str, float_format: str = "{:.4f}"):
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    tex_path = os.path.join(OUTPUT_DIR, f"{name}.tex")
    txt_path = os.path.join(OUTPUT_DIR, f"{name}.txt")

    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df_out.to_latex(index=False, float_format=float_format.format))
    except Exception as exc:
        print(f"Could not write LaTeX for {name}: {exc}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df_out.to_string(index=False))

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {txt_path}")


# ============================================================================
# 5. LOAD RAW PANEL AND PURIFIED WHITELIST
# ============================================================================

print("=" * 110)
print("LOADING PANELS")
print("=" * 110)

raw = pd.read_csv(RAW_PANEL_PATH, low_memory=False)

# Robustness: some versions use fiscal_year instead of year
if YEAR_COL not in raw.columns and "fiscal_year" in raw.columns:
    raw = raw.rename(columns={"fiscal_year": YEAR_COL})

required_cols = [
    COMPANY_COL,
    YEAR_COL,
    SECTOR_COL,
    NETSALES_COL,
    RND_COL,
    PROFIT_COL,
]

missing = [c for c in required_cols if c not in raw.columns]
if missing:
    raise KeyError(f"Missing required columns in raw panel: {missing}")

purified = pd.read_csv(PURIFIED_PANEL_PATH, usecols=[COMPANY_COL])
keep = set(purified[COMPANY_COL].dropna().map(clean_str).unique())

raw[COMPANY_COL] = raw[COMPANY_COL].map(clean_str)

before_firms = raw[COMPANY_COL].nunique()
df = raw[raw[COMPANY_COL].isin(keep)].copy()
after_firms = df[COMPANY_COL].nunique()

print(f"Raw panel rows: {len(raw)}")
print(f"Raw firms: {before_firms}")
print(f"Purified whitelist firms: {len(keep)}")
print(f"After purification: {after_firms} firms, {len(df)} rows")

# ============================================================================
# 6. NUMERIC CONVERSION AND BASIC FILTERS
# ============================================================================

for col in [YEAR_COL, NETSALES_COL, RND_COL, PROFIT_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "emp" in df.columns:
    df["emp"] = pd.to_numeric(df["emp"], errors="coerce")

# Restrict to the full article span
df = df[(df[YEAR_COL] >= 2005) & (df[YEAR_COL] <= 2024)].copy()

# Drop missing essential variables
df = df.dropna(
    subset=[
        COMPANY_COL,
        YEAR_COL,
        SECTOR_COL,
        NETSALES_COL,
        RND_COL,
        PROFIT_COL,
    ]
).copy()

# Revenue must be strictly positive to compute margin and R&D intensity
df = df[df[NETSALES_COL] > 0].copy()

# R&D can be zero, but not negative in this standardized Scoreboard construction.
df = df[df[RND_COL] >= 0].copy()

print(f"After basic numeric filters: {df[COMPANY_COL].nunique()} firms, {len(df)} rows")

# ============================================================================
# 7. COMPUTE MARGIN AND R&D INTENSITY
# ============================================================================

df["margin"] = df[PROFIT_COL] / df[NETSALES_COL]
df["rnd_intensity"] = df[RND_COL] / df[NETSALES_COL]

# ============================================================================
# 8. ECONOMIC CONSISTENCY FILTER
# ============================================================================

consistency_mask = (
    (df["margin"].abs() <= MAX_ABS_MARGIN)
    & (df["rnd_intensity"].abs() <= MAX_ABS_RND_INTENSITY)
)

dropped_consistency = df[~consistency_mask].copy()
df = df[consistency_mask].copy()

print(f"Dropped by economic consistency filter: {len(dropped_consistency)} firm-year rows")
print(f"After consistency filter: {df[COMPANY_COL].nunique()} firms, {len(df)} rows")

# ============================================================================
# 9. OPTIONAL DOCUMENTED PURGE
# ============================================================================

if APPLY_DOCUMENTED_PURGE:
    df["_company_upper"] = df[COMPANY_COL].map(clean_upper)

    purge_masks = []

    # Company exact purge, e.g. "37"
    if PURGE_COMPANY_EXACT:
        purge_masks.append(
            df["_company_upper"].isin({clean_upper(x) for x in PURGE_COMPANY_EXACT})
        )

    # ICB4 tires purge
    if ICB4_COL in df.columns and TIRES_ICB4:
        df["_icb4_clean"] = df[ICB4_COL].map(clean_str)
        purge_masks.append(df["_icb4_clean"].isin(TIRES_ICB4))

    # NACE 7010 holdings purge
    if NACE_COL in df.columns and PURGE_NACE_EXACT:
        df["_nace_clean"] = df[NACE_COL].map(clean_nace)
        purge_masks.append(df["_nace_clean"].isin(PURGE_NACE_EXACT))

    if purge_masks:
        purge_mask = np.logical_or.reduce([m.to_numpy() for m in purge_masks])
        purged = df[purge_mask].copy()
        df = df[~purge_mask].copy()

        print(f"Documented purge removed: {purged[COMPANY_COL].nunique()} firms, {len(purged)} rows")

        if len(purged) > 0:
            purge_summary = (
                purged.groupby([SECTOR_COL, COMPANY_COL], as_index=False)
                .size()
                .rename(columns={"size": "n_obs"})
                .sort_values([SECTOR_COL, COMPANY_COL])
            )
            save_csv_and_tex(purge_summary, "documented_purged_firms")

    df = df.drop(
        columns=[c for c in ["_company_upper", "_icb4_clean", "_nace_clean"] if c in df.columns],
        errors="ignore"
    )

else:
    print("Documented purge disabled: purified whitelist is treated as final.")

# ============================================================================
# 10. SECTOR CLEANING AND RECLASSIFICATION
# ============================================================================

df["Sector_Clean"] = df[SECTOR_COL].map(clean_upper)
df["_company_upper"] = df[COMPANY_COL].map(clean_upper)

# Reclassify ASML, TSMC, Huawei to INDUS, as stated in rvi4_revised
reclass_mask = contains_any(df["_company_upper"], RECLASSIFY_TO_INDUS_PATTERNS)
df.loc[reclass_mask, "Sector_Clean"] = "INDUS"

# Keep only the four article sectors
df = df[df["Sector_Clean"].isin(SECTOR_ORDER)].copy()

df["Block"] = df["Sector_Clean"].map(BLOCK_MAP)
df = df[df["Block"].notna()].copy()

print("=" * 110)
print("ANNUAL PANEL AFTER PURIFICATION, FILTERS AND RECLASSIFICATION")
print("=" * 110)

annual_agg = {
    "n_firms": (COMPANY_COL, "nunique"),
    "n_obs": (COMPANY_COL, "size"),
    "first_year": (YEAR_COL, "min"),
    "last_year": (YEAR_COL, "max"),
    "mean_margin": ("margin", "mean"),
    "mean_rnd_intensity": ("rnd_intensity", "mean"),
}

if "emp" in df.columns:
    annual_agg["mean_emp"] = ("emp", "mean")

annual_summary = (
    df.groupby(["Sector_Clean", "Block"], as_index=False, observed=True)
    .agg(**annual_agg)
    .sort_values(["Block", "Sector_Clean"])
)

print(annual_summary.to_string(index=False))

save_csv_and_tex(annual_summary, "Table0_annual_panel_summary")

# Save annual working panel
annual_panel_path = os.path.join(OUTPUT_DIR, "panel_rvi4_revised_annual_2005_2024.csv")
df.to_csv(annual_panel_path, index=False, encoding="utf-8-sig")
print(f"Saved annual working panel: {annual_panel_path}")

# ============================================================================
# 11. TEMPORAL DEPTH FILTER: DISABLED
# ============================================================================
# We do NOT require 10 consecutive years over 2005-2024.
# The RVI is computed on disjoint 5-year windows with:
#   - strict 5-out-of-5 completeness per window;
#   - strictly positive average operating margin per window.
#
# This is sufficient for the RVI construction and avoids over-filtering
# firms with intermittent reporting gaps, especially in AUTO.

print(
    "Temporal depth filter disabled: "
    "using strict 5/5 and positive mean margin per 5-year window."
)

# ============================================================================
# 12. FIRM-PERIOD RVI COMPUTATION
# ============================================================================

print("\n" + "=" * 110)
print("COMPUTING FIRM-PERIOD RVI")
print("=" * 110)
print(f"Standard deviation convention: ddof={STD_DDOF}")

period_records = []

for period_label, (y0, y1) in PERIODS.items():
    sub = df[df[YEAR_COL].between(y0, y1)].copy()

    if sub.empty:
        print(f"{period_label}: no observations in window")
        continue

    agg_spec = {
        "n_years": (YEAR_COL, "nunique"),
        "first_year": (YEAR_COL, "min"),
        "last_year": (YEAR_COL, "max"),
        "mbar": ("margin", "mean"),
        "rbar": ("rnd_intensity", "mean"),
        "sigma_m": ("margin", std_rvi),
        "sigma_r": ("rnd_intensity", std_rvi),
        "rd_median_million": (RND_COL, "median"),
        "netsales_median_million": (NETSALES_COL, "median"),
        "profit_median_million": (PROFIT_COL, "median"),
    }

    if "emp" in sub.columns:
        agg_spec["mean_emp"] = ("emp", "mean")

    stats = (
        sub.groupby(
            [COMPANY_COL, "Sector_Clean", "Block"],
            as_index=False,
            observed=True
        )
        .agg(**agg_spec)
    )

    if "mean_emp" not in stats.columns:
        stats["mean_emp"] = np.nan

    # Strict 5-out-of-5 rule
    if REQUIRE_STRICT_5_OUT_OF_5:
        stats = stats[stats["n_years"] == 5].copy()

    # Positive average operating margin over the window
    if REQUIRE_POSITIVE_MEAN_MARGIN:
        stats = stats[stats["mbar"] > 0].copy()

    if stats.empty:
        print(f"{period_label}: 0 firms retained after strict filters")
        continue

    # Convert median R&D to billion euros
    stats["rd_median_billion"] = stats["rd_median_million"] * RD_UNIT_TO_BILLION

    # Scale bonus
    stats["scale_bonus"] = 1.0 + np.log1p(
        stats["rd_median_billion"] / RD0_BILLION
    )

    # Creative effort term A
    stats["A"] = (
        stats["rbar"]
        * stats["scale_bonus"]
        / (1.0 + stats["sigma_r"])
    )

    # RVI
    stats["RVI"] = (
        stats["mbar"]
        * stats["A"]
        / (1.0 + stats["sigma_m"])
    )

    stats["Period"] = period_label
    stats["PeriodStart"] = y0
    stats["PeriodEnd"] = y1

    period_records.append(stats)

    print(
        f"{period_label}: {stats[COMPANY_COL].nunique()} firms retained "
        f"after strict 5/5 and positive margin filters"
    )

if not period_records:
    raise RuntimeError("No firm-period observations survived the filters.")

firm_period = pd.concat(period_records, ignore_index=True)

firm_period["Period"] = pd.Categorical(
    firm_period["Period"],
    categories=PERIOD_ORDER,
    ordered=True,
)

firm_period["Sector_Clean"] = pd.Categorical(
    firm_period["Sector_Clean"],
    categories=SECTOR_ORDER,
    ordered=True,
)

firm_period["Block"] = pd.Categorical(
    firm_period["Block"],
    categories=["2-cycle", "1-cycle"],
    ordered=True,
)

firm_period["_company_upper"] = firm_period[COMPANY_COL].map(clean_upper)

print("\nFirm-period RVI summary:")
print(
    firm_period.groupby(
        ["Period", "Sector_Clean"],
        observed=True
    ).size().to_string()
)

# Save firm-period working panel: this is the main panel to deposit on GitHub
firm_period_path = os.path.join(
    OUTPUT_DIR,
    "panel_rvi4_revised_firm_period.csv"
)
firm_period.to_csv(firm_period_path, index=False, encoding="utf-8-sig")
print(f"\nSaved firm-period working panel: {firm_period_path}")

# ============================================================================
# 13. TABLE 1 — PANEL COMPOSITION BY PERIOD AND SECTOR
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 1 — PANEL COMPOSITION BY PERIOD AND SECTOR")
print("=" * 110)

composition = (
    firm_period.groupby(["Period", "Sector_Clean"], observed=True)[COMPANY_COL]
    .nunique()
    .unstack(fill_value=0)
    .reindex(index=PERIOD_ORDER, columns=SECTOR_ORDER, fill_value=0)
    .astype(int)
)

composition["Total"] = composition.sum(axis=1)
table1 = composition.reset_index()

print(table1.to_string(index=False))
save_csv_and_tex(table1, "Table1_panel_composition_by_period")

# ============================================================================
# 14. TABLE 2 — TOP 100 GLOBAL AND PRESENCE RATE BY SECTOR
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 2 — TOP 100 GLOBAL: PRESENCE RATE BY SECTOR")
print("=" * 110)

presence_rows = []
top100_frames = []

for period_label in PERIOD_ORDER:
    fp = firm_period[firm_period["Period"] == period_label].copy()

    if fp.empty:
        continue

    top100 = fp.sort_values("RVI", ascending=False).head(TOP_N_GLOBAL).copy()
    top100_frames.append(top100)

    totals = fp["Sector_Clean"].value_counts().reindex(SECTOR_ORDER, fill_value=0)
    counts = top100["Sector_Clean"].value_counts().reindex(SECTOR_ORDER, fill_value=0)

    row = {"Period": period_label}

    for sector in SECTOR_ORDER:
        n = int(counts.get(sector, 0))
        N = int(totals.get(sector, 0))
        rate = safe_rate(n, N)

        row[f"{sector}_n"] = n
        row[f"{sector}_N"] = N
        row[f"{sector}_rate"] = rate
        row[f"{sector}_formatted"] = format_presence(n, N)

    presence_rows.append(row)

presence_table_raw = pd.DataFrame(presence_rows)

# Formatted table like in the article
presence_formatted = presence_table_raw[
    ["Period"] + [f"{s}_formatted" for s in SECTOR_ORDER]
].copy()
presence_formatted.columns = ["Period"] + SECTOR_ORDER

print(presence_formatted.to_string(index=False))
save_csv_and_tex(presence_formatted, "Table2_top100_presence_rate_formatted")

# Numeric presence table
presence_numeric = presence_table_raw[
    ["Period"] + [f"{s}_rate" for s in SECTOR_ORDER]
].copy()
presence_numeric.columns = ["Period"] + SECTOR_ORDER

print("\nNumeric presence rates:")
print(presence_numeric.to_string(index=False))
save_csv_and_tex(presence_numeric, "Table2_top100_presence_rate_numeric")

# Save full Top 100 by period
if top100_frames:
    top100_all = pd.concat(top100_frames, ignore_index=True)
    top100_path = os.path.join(OUTPUT_DIR, "top100_global_by_period.csv")
    top100_all.to_csv(top100_path, index=False, encoding="utf-8-sig")
    print(f"Saved Top 100 global by period: {top100_path}")

# ============================================================================
# 15. EQUAL-SIZED COHORT COMPARISON: PHARMA VS TECH
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 3 — EQUAL-SIZED COHORT COMPARISON: PHARMA VS TECH")
print("=" * 110)

cohort_rows = []

for period_label in PERIOD_ORDER:
    fp = firm_period[firm_period["Period"] == period_label].copy()
    if fp.empty:
        continue

    top100 = fp.sort_values("RVI", ascending=False).head(TOP_N_GLOBAL)

    N_pharma = fp[fp["Sector_Clean"] == "PHARMA"][COMPANY_COL].nunique()
    N_tech = fp[fp["Sector_Clean"] == "TECH"][COMPANY_COL].nunique()

    n_pharma = top100[top100["Sector_Clean"] == "PHARMA"][COMPANY_COL].nunique()
    n_tech = top100[top100["Sector_Clean"] == "TECH"][COMPANY_COL].nunique()

    rate_pharma = safe_rate(n_pharma, N_pharma)
    rate_tech = safe_rate(n_tech, N_tech)

    # Rescale PHARMA presence rate to the total TECH population size
    n_equiv_pharma = rate_pharma * N_tech if not pd.isna(rate_pharma) else np.nan

    if pd.isna(n_equiv_pharma):
        dominant = "TECH"
    else:
        dominant = "PHARMA" if n_equiv_pharma > n_tech else "TECH"

    cohort_rows.append({
        "Period": period_label,
        "n_P/N_P": rate_pharma,
        "n_T/N_T": rate_tech,
        "n_equiv_P": n_equiv_pharma,
        "n_T": n_tech,
        "Dominant": dominant,
    })

cohort_table = pd.DataFrame(cohort_rows)

print(cohort_table.to_string(index=False))
save_csv_and_tex(cohort_table, "Table3_equal_cohort_pharma_vs_tech")

# ============================================================================
# 16. TOP-1 RVI BY SECTOR AND PERIOD
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 4 — TOP-1 RVI BY SECTOR AND PERIOD")
print("=" * 110)

top1_rows = []

for period_label in PERIOD_ORDER:
    fp = firm_period[firm_period["Period"] == period_label].copy()
    if fp.empty:
        continue

    for sector in SECTOR_ORDER:
        sec = fp[fp["Sector_Clean"] == sector]

        if sec.empty:
            top1_rows.append({
                "Period": period_label,
                "Sector": sector,
                "Top1_Firm": "",
                "Top1_RVI": np.nan,
                "Label": "",
            })
        else:
            best = sec.sort_values("RVI", ascending=False).iloc[0]
            top1_rows.append({
                "Period": period_label,
                "Sector": sector,
                "Top1_Firm": best[COMPANY_COL],
                "Top1_RVI": best["RVI"],
                "Label": f"{best[COMPANY_COL]} ({best['RVI']:.4f})",
            })

top1_long = pd.DataFrame(top1_rows)

top1_wide = (
    top1_long.pivot(index="Period", columns="Sector", values="Label")
    .reindex(index=PERIOD_ORDER, columns=SECTOR_ORDER)
    .reset_index()
)

# Global #1 by period
global_top1 = (
    firm_period.sort_values(["Period", "RVI"], ascending=[True, False])
    .groupby("Period", as_index=False, observed=True)
    .head(1)
    .sort_values("Period")
)

global_top1["Global_Top1_Label"] = (
    global_top1[COMPANY_COL]
    + " ("
    + global_top1["RVI"].map(lambda x: f"{x:.4f}")
    + ") — "
    + global_top1["Sector_Clean"].astype(str)
)

top1_wide = top1_wide.merge(
    global_top1[["Period", "Global_Top1_Label"]],
    on="Period",
    how="left"
)

print(top1_wide.to_string(index=False))
save_csv_and_tex(top1_wide, "Table4_top1_RVI_by_sector_period")

# Also save numeric Top-1 table
top1_numeric = (
    top1_long.pivot(index="Period", columns="Sector", values="Top1_RVI")
    .reindex(index=PERIOD_ORDER, columns=SECTOR_ORDER)
    .reset_index()
)

save_csv_and_tex(top1_numeric, "Table4_top1_RVI_numeric")

# ============================================================================
# 17. TWO BLOCS: 2-CYCLE VS 1-CYCLE PRESENCE RATE
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 5 — TWO BLOCS: 2-CYCLE VS 1-CYCLE PRESENCE RATE")
print("=" * 110)

bloc_rows = []

for period_label in PERIOD_ORDER:
    fp = firm_period[firm_period["Period"] == period_label].copy()
    if fp.empty:
        continue

    top100 = fp.sort_values("RVI", ascending=False).head(TOP_N_GLOBAL)

    for bloc in ["2-cycle", "1-cycle"]:
        bloc_firms = fp[fp["Block"] == bloc]
        N_bloc = bloc_firms[COMPANY_COL].nunique()
        n_top_bloc = top100[top100["Block"] == bloc][COMPANY_COL].nunique()

        # Sector-level rows inside the bloc
        for sector in [s for s, b in BLOCK_MAP.items() if b == bloc]:
            n_top = top100[top100["Sector_Clean"] == sector][COMPANY_COL].nunique()
            N_sector = fp[fp["Sector_Clean"] == sector][COMPANY_COL].nunique()

            bloc_rows.append({
                "Period": period_label,
                "Bloc": bloc,
                "Sector": sector,
                "Firms_in_Top100": n_top,
                "Total_Firms": N_sector,
                "Presence_rate": safe_rate(n_top, N_sector),
                "Formatted": format_presence(n_top, N_sector),
            })

        # Bloc aggregate
        bloc_rows.append({
            "Period": period_label,
            "Bloc": bloc,
            "Sector": f"{bloc} AGGREGATE",
            "Firms_in_Top100": n_top_bloc,
            "Total_Firms": N_bloc,
            "Presence_rate": safe_rate(n_top_bloc, N_bloc),
            "Formatted": format_presence(n_top_bloc, N_bloc),
        })

bloc_table = pd.DataFrame(bloc_rows)

print(bloc_table.to_string(index=False))
save_csv_and_tex(bloc_table, "Table5_two_blocs_presence_rate")

# Article-style snapshot for 2020-2024
bloc_2020_2024 = bloc_table[bloc_table["Period"] == "2020-2024"].copy()

print("\nArticle-style bloc table for 2020-2024:")
print(bloc_2020_2024.to_string(index=False))
save_csv_and_tex(bloc_2020_2024, "Table5_two_blocs_2020_2024")

# ============================================================================
# 18. CASE STUDIES: TOP-1 AND TOP-2 FIRMS BY SECTOR AND PERIOD
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 6 — CASE STUDIES: TOP-1 AND TOP-2 FIRMS BY SECTOR AND PERIOD")
print("=" * 110)

case_rows = []

for period_label in PERIOD_ORDER:
    fp = firm_period[firm_period["Period"] == period_label].copy()
    if fp.empty:
        continue

    for sector in SECTOR_ORDER:
        sec = fp[fp["Sector_Clean"] == sector].sort_values("RVI", ascending=False)

        for rank, (_, row) in enumerate(sec.head(2).iterrows(), start=1):
            case_rows.append({
                "Period": period_label,
                "Sector": sector,
                "Rank": rank,
                "Firm": row[COMPANY_COL],
                "RVI": row["RVI"],
                "mbar": row["mbar"],
                "rbar": row["rbar"],
                "A": row["A"],
                "sigma_m": row["sigma_m"],
                "sigma_r": row["sigma_r"],
                "rd_median_billion": row["rd_median_billion"],
            })

case_table = pd.DataFrame(case_rows)

print(case_table.to_string(index=False))
save_csv_and_tex(case_table, "Table6_case_studies_top1_top2")

# Compact label table similar to article case-study tables
case_label_rows = []

for period_label in PERIOD_ORDER:
    row = {"Period": period_label}

    for sector in SECTOR_ORDER:
        sub = case_table[
            (case_table["Period"] == period_label)
            & (case_table["Sector"] == sector)
        ].sort_values("Rank")

        labels = []
        for _, r in sub.iterrows():
            if pd.notna(r["Firm"]):
                labels.append(f"{r['Firm']} ({r['RVI']:.4f})")

        row[sector] = " / ".join(labels)

    case_label_rows.append(row)

case_labels_wide = pd.DataFrame(case_label_rows)

print("\nCompact Top1 / Top2 labels:")
print(case_labels_wide.to_string(index=False))
save_csv_and_tex(case_labels_wide, "Table6_case_studies_top1_top2_labels")

# ============================================================================
# 19. QUALCOMM / ORACLE DIAGNOSTIC
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 7 — QUALCOMM / ORACLE RVI DECOMPOSITION")
print("=" * 110)

diag_rows = []

for firm_pattern in DIAGNOSTIC_FIRMS:
    firm_data = firm_period[
        firm_period["_company_upper"].str.contains(
            firm_pattern,
            na=False,
            regex=False
        )
    ].copy()

    if firm_data.empty:
        print(f"Warning: no firm matched diagnostic pattern: {firm_pattern}")

    for period_label in PERIOD_ORDER:
        rows = firm_data[firm_data["Period"] == period_label]

        if rows.empty:
            diag_rows.append({
                "Period": period_label,
                "Firm": firm_pattern,
                "mbar": np.nan,
                "A": np.nan,
                "mbar_A": np.nan,
                "RVI": np.nan,
            })
        else:
            r = rows.sort_values("RVI", ascending=False).iloc[0]
            diag_rows.append({
                "Period": period_label,
                "Firm": firm_pattern,
                "mbar": r["mbar"],
                "A": r["A"],
                "mbar_A": r["mbar"] * r["A"],
                "RVI": r["RVI"],
            })

diag_table = pd.DataFrame(diag_rows)

print("Long diagnostic table:")
print(diag_table.to_string(index=False))
save_csv_and_tex(diag_table, "Table7_qualcomm_oracle_decomposition_long")

# Leader table: Qualcomm vs Oracle
try:
    rvi_pivot = (
        diag_table.pivot(index="Period", columns="Firm", values="RVI")
        .reindex(PERIOD_ORDER)
    )

    leader_rows = []

    for period_label in PERIOD_ORDER:
        q = rvi_pivot.loc[period_label, "QUALCOMM"] if "QUALCOMM" in rvi_pivot.columns else np.nan
        o = rvi_pivot.loc[period_label, "ORACLE"] if "ORACLE" in rvi_pivot.columns else np.nan

        if pd.isna(q) and pd.isna(o):
            leader = "NA"
            diff = np.nan
        elif pd.isna(o):
            leader = "Qualcomm"
            diff = np.nan
        elif pd.isna(q):
            leader = "Oracle"
            diff = np.nan
        else:
            diff = q - o
            leader = "Qualcomm" if q >= o else "Oracle"

        leader_rows.append({
            "Period": period_label,
            "Qualcomm_RVI": q,
            "Oracle_RVI": o,
            "Leader": leader,
            "Difference_Q_minus_O": diff,
        })

    leader_table = pd.DataFrame(leader_rows)

    print("\nQualcomm / Oracle leader table:")
    print(leader_table.to_string(index=False))
    save_csv_and_tex(leader_table, "Table7_qualcomm_oracle_leader")

except Exception as exc:
    print(f"Could not build Qualcomm / Oracle leader table: {exc}")

# Flat wide decomposition table, fixed version
wide_diag = pd.DataFrame({"Period": PERIOD_ORDER})

for metric in ["mbar", "A", "mbar_A", "RVI"]:
    piv = (
        diag_table.pivot(index="Period", columns="Firm", values=metric)
        .reindex(PERIOD_ORDER)
    )

    for firm in ["QUALCOMM", "ORACLE"]:
        if firm in piv.columns:
            wide_diag[f"{metric}_{firm}"] = piv[firm].to_numpy()
        else:
            wide_diag[f"{metric}_{firm}"] = np.nan

print("\nFlat wide diagnostic table:")
print(wide_diag.to_string(index=False))
save_csv_and_tex(wide_diag, "Table7_qualcomm_oracle_decomposition_wide")

# ============================================================================
# 20. TOYOTA / MERCEDES-BENZ DIAGNOSTIC (AUTO SECTOR)
# ============================================================================

print("\n" + "=" * 110)
print("TABLE 8 — TOYOTA / MERCEDES-BENZ REALIZED VS UNREALIZED REMANENCE")
print("=" * 110)

# ---------------------------------------------------------------------------
# Patterns used to identify the firms.
# Mercedes may appear as DAIMLER in older periods, depending on the source.
# Toyota may appear with subsidiaries; the code selects the largest entity
# by median net sales inside each period.
# ---------------------------------------------------------------------------
TOYOTA_PATTERNS = [
    "TOYOTA MOTOR",
    "TOYOTA",
]

MERCEDES_PATTERNS = [
    "MERCEDES-BENZ",
    "MERCEDES BENZ",
    "DAIMLER",
]

CASE_DEFINITIONS = {
    "Toyota Motor": TOYOTA_PATTERNS,
    "Mercedes-Benz": MERCEDES_PATTERNS,
}

# ---------------------------------------------------------------------------
# Work only on AUTO firms that passed the rvi4_revised filters.
# ---------------------------------------------------------------------------
auto_panel = firm_period[firm_period["Sector_Clean"] == "AUTO"].copy()

if auto_panel.empty:
    print("Warning: no AUTO firms survived the filters. Toyota/Mercedes diagnostic skipped.")

else:

    auto_panel["_company_upper"] = auto_panel[COMPANY_COL].map(clean_upper)

    # Rank inside the AUTO sector, period by period.
    auto_panel["AUTO_rank"] = (
        auto_panel.groupby("Period", observed=True)["RVI"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    # Number of AUTO firms in each period.
    auto_totals = (
        auto_panel.groupby("Period", observed=True)[COMPANY_COL]
        .nunique()
        .rename("n_auto_firms")
        .reset_index()
    )

    auto_panel = auto_panel.merge(
        auto_totals,
        on="Period",
        how="left"
    )

    # -----------------------------------------------------------------------
    # Helper: English ordinal for rank display.
    # -----------------------------------------------------------------------
    def ordinal(n):
        if pd.isna(n):
            return "NA"
        n = int(n)
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    # -----------------------------------------------------------------------
    # Helper: pick the best matching firm-row for a given case and period.
    # If several entities match, prefer the largest by net sales, then by RVI.
    # -----------------------------------------------------------------------
    def pick_case_row(
        period,
        patterns,
        data,
        priority_metric="netsales_median_million"
    ):
        cand = data[
            (data["Period"] == period)
            & contains_any(data["_company_upper"], patterns)
        ].copy()

        if cand.empty:
            return None

        sort_cols = [
            c for c in [priority_metric, "RVI"]
            if c in cand.columns
        ]

        if sort_cols:
            cand = cand.sort_values(
                sort_cols,
                ascending=False,
                na_position="last"
            )

        return cand.iloc[0]

    # -----------------------------------------------------------------------
    # Detailed period-by-period table for Toyota and Mercedes.
    # -----------------------------------------------------------------------
    case_rows = []

    for case_label, patterns in CASE_DEFINITIONS.items():
        for period_label in PERIOD_ORDER:
            row = pick_case_row(period_label, patterns, auto_panel)

            if row is None:
                case_rows.append({
                    "Case": case_label,
                    "Period": period_label,
                    "Matched_firm": np.nan,
                    "AUTO_rank": np.nan,
                    "n_auto_firms": np.nan,
                    "RVI": np.nan,
                    "mbar": np.nan,
                    "rbar": np.nan,
                    "A": np.nan,
                    "sigma_m": np.nan,
                    "sigma_r": np.nan,
                    "rd_median_billion": np.nan,
                })
            else:
                case_rows.append({
                    "Case": case_label,
                    "Period": period_label,
                    "Matched_firm": row[COMPANY_COL],
                    "AUTO_rank": row["AUTO_rank"],
                    "n_auto_firms": row["n_auto_firms"],
                    "RVI": row["RVI"],
                    "mbar": row["mbar"],
                    "rbar": row["rbar"],
                    "A": row["A"],
                    "sigma_m": row["sigma_m"],
                    "sigma_r": row["sigma_r"],
                    "rd_median_billion": row["rd_median_billion"],
                })

    toyota_mercedes_detail = pd.DataFrame(case_rows)

    if not toyota_mercedes_detail.empty:
        toyota_mercedes_detail["Period"] = pd.Categorical(
            toyota_mercedes_detail["Period"],
            categories=PERIOD_ORDER,
            ordered=True,
        )

    print("\nToyota / Mercedes detailed period table:")
    print(toyota_mercedes_detail.to_string(index=False))

    if not toyota_mercedes_detail.empty:
        toyota_mercedes_detail_out = toyota_mercedes_detail.copy()
        for col in ["AUTO_rank", "n_auto_firms"]:
            if col in toyota_mercedes_detail_out.columns:
                toyota_mercedes_detail_out[col] = pd.to_numeric(
                    toyota_mercedes_detail_out[col],
                    errors="coerce"
                )
        save_csv_and_tex(
            toyota_mercedes_detail_out,
            "Table8_toyota_mercedes_detail"
        )

    # -----------------------------------------------------------------------
    # Trajectory summary: first rank, last rank, RVI change, A change,
    # margin change.
    # -----------------------------------------------------------------------
    trajectory_rows = []

    for case_label in CASE_DEFINITIONS.keys():

        if toyota_mercedes_detail.empty:
            trajectory_rows.append({
                "Case": case_label,
                "Matched_firms": np.nan,
                "Trajectory": "No eligible AUTO observation",
                "Ranks_2005_2024": "NA",
                "First_rank": np.nan,
                "Last_rank": np.nan,
                "Rank_improvement": np.nan,
                "First_RVI": np.nan,
                "Last_RVI": np.nan,
                "RVI_change": np.nan,
                "First_A": np.nan,
                "Last_A": np.nan,
                "A_change": np.nan,
                "First_mbar": np.nan,
                "Last_mbar": np.nan,
                "mbar_change": np.nan,
                "RVI_reading": "No data",
            })
            continue

        sub = toyota_mercedes_detail[
            (toyota_mercedes_detail["Case"] == case_label)
            & toyota_mercedes_detail["AUTO_rank"].notna()
            & toyota_mercedes_detail["RVI"].notna()
        ].sort_values("Period")

        if sub.empty:
            trajectory_rows.append({
                "Case": case_label,
                "Matched_firms": np.nan,
                "Trajectory": "No eligible AUTO observation",
                "Ranks_2005_2024": "NA",
                "First_rank": np.nan,
                "Last_rank": np.nan,
                "Rank_improvement": np.nan,
                "First_RVI": np.nan,
                "Last_RVI": np.nan,
                "RVI_change": np.nan,
                "First_A": np.nan,
                "Last_A": np.nan,
                "A_change": np.nan,
                "First_mbar": np.nan,
                "Last_mbar": np.nan,
                "mbar_change": np.nan,
                "RVI_reading": "No data",
            })
            continue

        first = sub.iloc[0]
        last = sub.iloc[-1]

        ranks_all = " -> ".join(
            ordinal(x) if pd.notna(x) else "NA"
            for x in sub["AUTO_rank"]
        )

        rank_improvement = float(first["AUTO_rank"]) - float(last["AUTO_rank"])
        rvi_change = float(last["RVI"]) - float(first["RVI"])
        a_change = float(last["A"]) - float(first["A"])
        mbar_change = float(last["mbar"]) - float(first["mbar"])

        # Simple automatic reading consistent with the article's logic.
        if rank_improvement > 0 and rvi_change > 0 and a_change > 0:
            reading = "Partially realized remanence"
        elif rank_improvement < 0 or a_change <= 0:
            reading = "Accumulated but unrealized remanence"
        else:
            reading = "Mixed trajectory"

        trajectory_rows.append({
            "Case": case_label,
            "Matched_firms": " / ".join(
                sub["Matched_firm"].dropna().unique()
            ),
            "Trajectory": (
                f"{first['Period']} {ordinal(first['AUTO_rank'])} -> "
                f"{last['Period']} {ordinal(last['AUTO_rank'])}"
            ),
            "Ranks_2005_2024": ranks_all,
            "First_rank": first["AUTO_rank"],
            "Last_rank": last["AUTO_rank"],
            "Rank_improvement": rank_improvement,
            "First_RVI": first["RVI"],
            "Last_RVI": last["RVI"],
            "RVI_change": rvi_change,
            "First_A": first["A"],
            "Last_A": last["A"],
            "A_change": a_change,
            "First_mbar": first["mbar"],
            "Last_mbar": last["mbar"],
            "mbar_change": mbar_change,
            "RVI_reading": reading,
        })

    toyota_mercedes_trajectory = pd.DataFrame(trajectory_rows)

    print("\nToyota / Mercedes trajectory summary:")
    print(toyota_mercedes_trajectory.to_string(index=False))

    if not toyota_mercedes_trajectory.empty:
        toyota_mercedes_trajectory_out = toyota_mercedes_trajectory.copy()
        for col in ["First_rank", "Last_rank", "Rank_improvement"]:
            if col in toyota_mercedes_trajectory_out.columns:
                toyota_mercedes_trajectory_out[col] = pd.to_numeric(
                    toyota_mercedes_trajectory_out[col],
                    errors="coerce"
                )
        save_csv_and_tex(
            toyota_mercedes_trajectory_out,
            "Table8_toyota_mercedes_trajectory"
        )

    # -----------------------------------------------------------------------
    # Top 10 AUTO firms by period, for context.
    # This helps verify whether Toyota/Mercedes movements are within-sector
    # or driven by changes in the AUTO competitive field.
    # -----------------------------------------------------------------------
    auto_top10 = (
        auto_panel
        .sort_values(["Period", "RVI"], ascending=[True, False])
        .groupby("Period", as_index=False, observed=True)
        .head(10)
    )

    auto_top10_cols = [
        "Period",
        "AUTO_rank",
        COMPANY_COL,
        "RVI",
        "mbar",
        "rbar",
        "A",
        "sigma_m",
        "sigma_r",
        "rd_median_billion",
        "n_auto_firms",
    ]

    auto_top10_output = auto_top10[
        [c for c in auto_top10_cols if c in auto_top10.columns]
    ].copy()

    print("\nTop 10 AUTO firms by period:")
    print(auto_top10_output.to_string(index=False))

    if not auto_top10_output.empty:
        auto_top10_output_out = auto_top10_output.copy()
        if "AUTO_rank" in auto_top10_output_out.columns:
            auto_top10_output_out["AUTO_rank"] = pd.to_numeric(
                auto_top10_output_out["AUTO_rank"],
                errors="coerce"
            )
        if "n_auto_firms" in auto_top10_output_out.columns:
            auto_top10_output_out["n_auto_firms"] = pd.to_numeric(
                auto_top10_output_out["n_auto_firms"],
                errors="coerce"
            )
        save_csv_and_tex(
            auto_top10_output_out,
            "Table8_auto_top10_by_period"
        )

# ============================================================================
# 21. FINAL SUMMARY OF SAVED FILES
# ============================================================================

print("\n" + "=" * 110)
print("SAVED FILES")
print("=" * 110)

for f in sorted(os.listdir(OUTPUT_DIR)):
    print(os.path.join(OUTPUT_DIR, f))

print("\nDONE.")