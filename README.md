

                  Lahoussine Laanait,
            The Remanent Value Indicator (RVI): 
            An Empirical Instrument for Detecting the Realization of Remanent Value in Cognitive Capitalism
            (Second Version), October 2026

# RVI Final — Replication Package

This repository contains the code and data required to reproduce the results of the manuscript:

**“The Remanent Value Indicator (RVI): An Empirical Instrument for Detecting the Realization of Remanent Value in Cognitive Capitalism (Second Version)”**

The main script is:
RVI_Final.py
 Data files
2.1 Panel_Work_Final_RVI_SCOREBOARD.csv
This is the main raw firm-year panel.
It contains the annual observations used to compute the RVI.
Expected main columns:
Column
Description
company
Firm name
year
Fiscal year
Sector_Final
Sector classification: TECH, PHARMA, INDUS, AUTO
netsales_24
Net sales / revenue, constant 2024 euros
rnd_24
R&D expenditure, constant 2024 euros
profit_24
Operating profit / EBIT proxy, constant 2024 euros
emp
Number of employees
icb4_name
ICB4 industry classification, used for diagnostics
nace_rev2_new
NACE Rev. 2 code, used for diagnostics
This file is the annual source panel. It is not modified directly by the script.
2.2 RVI_SCOREBOARD_DEFINITIF_FINAL_EUR.csv
This file is the purified firm whitelist.
It defines the final list of firms retained in the RVI panel.
Expected main column:
Column
Description
company
Firm name
The script keeps only firms whose name appears in this whitelist.
Important: this file is treated as the final purified panel. The script does not re-apply the earlier documented purge unless explicitly enabled.

What RVI_Final.py does
The script performs the following steps:
Loads the raw annual panel:   Panel_Work_Final_RVI_SCOREBOARD.csv
Loads the purified firm whitelist:   RVI_SCOREBOARD_DEFINITIF_FINAL_EUR.csv
Applies the economic consistency filter:    abs(margin) <= 0.60
   abs(rnd_intensity) <= 0.60
builds 5 disjoint 5-year windows:
2005–2009
2010–2014
2015–2019
2020–2024
For each firm and each window, keeps only firms satisfying:
strict 5-out-of-5 annual observations;
strictly positive average operating margin over the window.
Install dependencies
places the files as follows: 
data/raw/Panel_Work_Final_RVI_SCOREBOARD.csv
data/purified/RVI_SCOREBOARD_DEFINITIF_FINAL_EUR.csv
Main outputs
The script generates CSV, TXT and LaTeX files.
The most important output is:panel_rvi4_revised_firm_period.csv
 Analytical blocks
The script uses two production regimes:
Block
Sectors
Regime
2-cycle
PHARMA, TECH
Duplicable knowledge assets
1-cycle
INDUS, AUTO
Physical usable products
7. Important notes
Do not modify the raw CSV files directly.
The purified whitelist is used as the final firm selection.
The script does not apply the 10-consecutive-years filter.
The RVI windows rely on:
strict 5/5 completeness;
positive average margin;
economic consistency filter at 60%.


