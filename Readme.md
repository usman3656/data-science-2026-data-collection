Currency Crisis Early Warning System (EWS) - Data Pipeline
Project Overview
This repository contains the data extraction and preprocessing pipeline for an Early Warning System (EWS) designed to predict currency crises in 23 Emerging Markets. The model relies on macroeconomic indicators, central bank balance sheets, and external trade data sourced primarily via the IMF SDMX 2.1 API, spanning a 50-year period (1974–2026).

📂 1. Datasets Successfully Collected
The following macroeconomic blocks have been successfully extracted, parsed from XML, and formatted into country-specific time-series matrices:

International Reserves: The primary defense mechanism against currency runs.

Consumer Price Index (CPI) / Inflation: The measure of real economic cost and domestic purchasing power degradation.

Monetary Base (M0): Extracted from the Central Bank Survey (MFS_CBS), representing "High-Powered Money" and direct central bank money printing.

Broad Money (M2): Extracted from the Depository Corporations Survey (MFS_DC), representing total banking system liquidity and potential capital flight risk.

International Trade in Goods (ITG): Total Exports (XG) and Imports (MG) to calculate Net Trade Balances and export momentum.

Interest Rates: Key monetary policy signals including Policy Rates, Interbank Rates, 3-Month T-Bills, and 10-Year Government Bonds (MFS_IR).

Nominal Exchange Rates: The baseline currency value against the USD.

⏳ 2. Pending Datasets (To-Do)
To complete the feature engineering for the Kaminsky-Lizondo-Reinhart (KLR) crisis model, the following datasets still need to be extracted:

Real Effective Exchange Rate (REER): Crucial for measuring currency overvaluation. Needs to be extracted from the Effective Exchange Rate (EER) dataset using CPI-based indices.

Domestic Credit to the Private Sector: Crucial for detecting credit booms and asset bubbles. Can be sourced from the IMF Financial Soundness Indicators (FSI) or Depository Corporations (MFS_DC).

Short-Term External Debt: Crucial for the Greenspan-Guidotti rule (Reserves vs. Short-Term Debt). Sourced via the World Bank International Debt Statistics (IDS) or IMF Quarterly External Debt Statistics (QEDS).

🛠️ 3. Action Plan & Division of Labor
To hit our project deadlines efficiently, the remaining work is split into two parallel tracks:

Track A: Data Acquisition (Team Member 1)
Goal: Hunt down and extract the remaining datasets from the IMF/World Bank APIs.

Task 1: Adapt the existing "Hunter Scripts" to discover the exact 2025 dimension keys for REER, Domestic Credit, and Short-Term Debt.

Task 2: Write the master extraction loops for these final variables across all 23 Emerging Markets.

Task 3: Export the raw data into clean .csv matrices (Dates as rows, Country_Indicator as columns) and push to the repo.

Track B: Data Wrangling & Quality Assurance (Team Member 2)
Goal: Clean, merge, and validate the 7 datasets we already have to prepare the master training dataframe.

Task 1 (Merge): Combine all existing CSVs into a single master dataframe, aligning everything perfectly by the Date index.

Task 2 (QA & Validation): Check for API extraction errors (e.g., a column filled entirely with zeros, massive unjustified spikes due to currency denomination changes, or missing countries). Ensure all YoY growth calculations are mathematically sound. Check for incorrect data or wrong formulas. 



Here are the 23 countries (22 Emerging Markets plus the United States as the global benchmark) that we have been targeting across all the extraction scripts.

Asia-Pacific

China (CHN)

India (IND)

Indonesia (IDN)

South Korea (KOR)

Malaysia (MYS)

Philippines (PHL)

Thailand (THA)


Latin America

Brazil (BRA)

Mexico (MEX)

Chile (CHL)

Colombia (COL)

Peru (PER)


Emerging Europe

Poland (POL)

Czech Republic (CZE)

Hungary (HUN)

Greece (GRC)

Turkey (TUR)


Middle East & Africa

Saudi Arabia (SAU)

United Arab Emirates (ARE)

Qatar (QAT)

Kuwait (KWT)

South Africa (ZAF)


Global Benchmark

United States (USA)

