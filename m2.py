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

print("Extracting Broad Money (M2) across all EMs using the verified 2025 DCORP_L_BM code...")

all_data = []

# 2. Loop through countries to prevent IMF server timeouts
for code, name in country_map.items():
    print(f" -> Processing {name} ({code})...")

    # 4-Dimension Key: Area . Indicator . Unit (Wildcard) . Frequency
    # We use '..' to wildcard the unit (e.g., XDC for Domestic Currency)
    dimension_key = f"{code}.DCORP_L_BM..M"

    # Using the Depository Corporations dataset
    url = f"https://api.imf.org/external/sdmx/2.1/data/MFS_DC/{dimension_key}"

    try:
        response = requests.get(url, headers={"Accept": "application/xml"}, params={"startPeriod": "1974"})

        if response.status_code == 200:
            it = ET.iterparse(io.StringIO(response.text))
            for _, el in it:
                _, _, el.tag = el.tag.rpartition('}')
            root = it.root

            for series in root.findall('.//Series'):
                # Grab the unit from the metadata attributes
                unit = series.attrib.get('UNIT_MULT') or series.attrib.get('UNIT') or 'Local_Currency'

                for obs in series.findall('.//Obs'):
                    obs_date = obs.attrib.get('TIME_PERIOD')
                    obs_val = obs.attrib.get('OBS_VALUE')

                    if obs_date and obs_val is not None:
                        all_data.append({
                            'Date': obs_date,
                            'Country': name,
                            'Broad_Money_Value': float(obs_val)
                        })
        else:
            print(f"    [!] Failed or empty for {name}.")

        # 0.1s pause to respect IMF API rate limits
        time.sleep(0.1)

    except Exception as e:
        print(f"    [!] Error extracting {name}: {e}")

# 3. Process into a Master Feature Matrix
if all_data:
    df = pd.DataFrame(all_data)

    # Clean SDMX monthly dates (e.g., 2024-M01 -> 2024-01)
    df['Date'] = df['Date'].str.replace('-M', '-')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Pivot so each country gets its own column
    master_m2 = df.pivot_table(index='Date', columns='Country', values='Broad_Money_Value', aggfunc='first')
    master_m2.sort_index(inplace=True)

    # Rename columns explicitly
    master_m2.columns = [f"{col}_Broad_Money_M2" for col in master_m2.columns]

    # 4. Feature Engineering: YoY Broad Money Growth
    print("\nCalculating YoY Growth trajectories for the EWS model...")
    for country in country_map.values():
        col_name = f"{country}_Broad_Money_M2"
        if col_name in master_m2.columns:
            # YoY Growth is a primary trigger in Second-Generation crisis models
            master_m2[f"{country}_M2_Growth_YoY_%"] = master_m2[col_name].pct_change(12) * 100

    # Sort columns alphabetically to group country features together
    master_m2 = master_m2.reindex(sorted(master_m2.columns), axis=1)

    master_m2.to_csv("Master_EM_M2_BroadMoney_50Y.csv")
    print("\n--- EXTRACTION COMPLETE ---")
    print("Saved matrix to 'Master_EM_M2_BroadMoney_50Y.csv'")
    print(f"Total Columns Generated: {len(master_m2.columns)}")

    # Preview the target data
    preview_cols = [c for c in master_m2.columns if 'Brazil' in c or 'India' in c]
    if preview_cols:
        print("\nPreview of Brazil & India M2 Data:")
        print(master_m2[preview_cols].tail())
else:
    print("\nNo data retrieved across the country loop. Double check network connection.")