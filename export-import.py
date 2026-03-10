import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import io

# 1. The 23 Emerging Markets + USA
country_map = {
    'CHN': 'China', 'IND': 'India', 'IDN': 'Indonesia', 'KOR': 'South Korea',
    'MYS': 'Malaysia', 'PHL': 'Philippines', 'THA': 'Thailand', 'BRA': 'Brazil',
    'MEX': 'Mexico', 'CHL': 'Chile', 'COL': 'Colombia', 'PER': 'Peru',
    'POL': 'Poland', 'CZE': 'Czech Republic', 'HUN': 'Hungary', 'GRC': 'Greece',
    'TUR': 'Turkey', 'SAU': 'Saudi Arabia', 'ARE': 'UAE', 'QAT': 'Qatar',
    'KWT': 'Kuwait', 'ZAF': 'South Africa', 'USA': 'United States'
}

# The new 2025 ITG Indicators
indicators = {
    'XG': 'Exports',
    'MG': 'Imports'
}

print("Extracting Export & Import Data across all EMs using the new ITG codes (XG/MG)...")

all_data = []

# 2. Extract Data via Loop
for code, name in country_map.items():
    print(f" -> Processing {name} ({code})...")

    for ind_code, ind_name in indicators.items():
        # 4-Dimension Key: Area . Indicator . Counterpart/Unit (Wildcard) . Frequency
        dimension_key = f"{code}.{ind_code}..M"

        url = f"https://api.imf.org/external/sdmx/2.1/data/ITG/{dimension_key}"

        try:
            response = requests.get(url, headers={"Accept": "application/xml"}, params={"startPeriod": "1974"})

            if response.status_code == 200:
                it = ET.iterparse(io.StringIO(response.text))
                for _, el in it:
                    _, _, el.tag = el.tag.rpartition('}')
                root = it.root

                for series in root.findall('.//Series'):
                    for obs in series.findall('.//Obs'):
                        obs_date = obs.attrib.get('TIME_PERIOD')
                        obs_val = obs.attrib.get('OBS_VALUE')

                        if obs_date and obs_val is not None:
                            all_data.append({
                                'Date': obs_date,
                                'Country': name,
                                'Flow': ind_name,
                                'Value': float(obs_val)
                            })

            # Respect IMF API rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"    [!] Error extracting {ind_name} for {name}: {e}")

# 3. Process into a Master Feature Matrix
if all_data:
    df = pd.DataFrame(all_data)

    # Clean standard SDMX monthly dates
    df['Date'] = df['Date'].str.replace('-M', '-')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Pivot so each country/flow gets its own column
    master_trade = df.pivot_table(index='Date', columns=['Country', 'Flow'], values='Value', aggfunc='first')
    master_trade.sort_index(inplace=True)

    # Flatten multi-level columns: 'Brazil_Exports', 'Brazil_Imports', etc.
    master_trade.columns = [f"{col[0]}_{col[1]}" for col in master_trade.columns.values]

    # 4. Feature Engineering: Net Trade Balance
    print("\nCalculating Net Trade Balances for the EWS model...")
    for country in country_map.values():
        exp_col = f"{country}_Exports"
        imp_col = f"{country}_Imports"

        if exp_col in master_trade.columns and imp_col in master_trade.columns:
            # Trade Balance = Exports - Imports
            master_trade[f"{country}_Trade_Balance"] = master_trade[exp_col] - master_trade[imp_col]

            # Export Momentum (YoY % Change) - Crucial for detecting sudden stops in foreign currency inflows
            master_trade[f"{country}_Export_Momentum_YoY_%"] = master_trade[exp_col].pct_change(12) * 100

    # Sort columns alphabetically to keep country variables grouped
    master_trade = master_trade.reindex(sorted(master_trade.columns), axis=1)

    master_trade.to_csv("Master_EM_Trade_Data_50Y.csv")
    print("\n--- EXTRACTION COMPLETE ---")
    print("Saved matrix to 'Master_EM_Trade_Data_50Y.csv'")
    print(f"Total Columns Generated: {len(master_trade.columns)}")

    preview_cols = [c for c in master_trade.columns if 'Brazil' in c or 'India' in c]
    if preview_cols:
        print("\nPreview of Trade Data:")
        print(master_trade[preview_cols].tail())
else:
    print("\nNo data retrieved across the country loop. Ensure network is stable.")