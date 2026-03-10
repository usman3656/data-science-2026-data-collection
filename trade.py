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

print("Extracting Monetary Base (M0) across all EMs using the verified 2025 SDMX codes...")

all_data = []

# 2. Looping through each country to prevent 400/500 server crashes
for code, name in country_map.items():
    print(f" -> Processing {name} ({code})...")

    # 4-Dimension Key: Area . Indicator . Unit . Frequency
    # Using the discovered 2025 indicator: S121_L_MB_XDC_CBS (Monetary Base in Domestic Currency)
    # Using a wildcard (.) for the Unit dimension to catch everything available
    dimension_key = f"{code}.S121_L_MB_XDC_CBS..M"

    url = f"https://api.imf.org/external/sdmx/2.1/data/MFS_CBS/{dimension_key}"

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
                            'Monetary_Base_XDC': float(obs_val)
                        })
        else:
            print(f"    [!] Failed or empty for {name}.")

        # Respect IMF API rate limits
        time.sleep(0.1)

    except Exception as e:
        print(f"    [!] Error extracting {name}: {e}")

# 3. Process into a Master Matrix
if all_data:
    df = pd.DataFrame(all_data)
    # Clean up standard SDMX monthly date formats (e.g., 2024-M01 to 2024-01)
    df['Date'] = df['Date'].str.replace('-M', '-')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Pivot so each country gets its own column
    master_m0 = df.pivot_table(index='Date', columns='Country', values='Monetary_Base_XDC', aggfunc='first')
    master_m0.sort_index(inplace=True)

    # Rename base columns
    master_m0.columns = [f"{col}_Monetary_Base_XDC" for col in master_m0.columns]

    # 4. Feature Engineering: Calculate YoY Growth Rates
    print("\nCalculating YoY Growth trajectories for EWS model...")
    for country in country_map.values():
        col_name = f"{country}_Monetary_Base_XDC"
        if col_name in master_m0.columns:
            master_m0[f"{country}_M0_Growth_YoY_%"] = master_m0[col_name].pct_change(12) * 100

    # Sort columns alphabetically to group country features together
    master_m0 = master_m0.reindex(sorted(master_m0.columns), axis=1)

    master_m0.to_csv("Master_EM_MonetaryBase_50Y.csv")
    print("\n--- EXTRACTION COMPLETE ---")
    print("Saved matrix to 'Master_EM_MonetaryBase_50Y.csv'")
    print(f"Total Columns Generated: {len(master_m0.columns)}")

    preview_cols = [c for c in master_m0.columns if 'Brazil' in c]
    if preview_cols:
        print("\nPreview of Brazil Data:")
        print(master_m0[preview_cols].tail())
else:
    print("\nNo data retrieved across the country loop.")