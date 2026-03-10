import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import io

# 1. The 22 Emerging Markets (USA Excluded)
country_map = {
    'CHN': 'China', 'IND': 'India', 'IDN': 'Indonesia', 'KOR': 'South Korea',
    'MYS': 'Malaysia', 'PHL': 'Philippines', 'THA': 'Thailand', 'BRA': 'Brazil',
    'MEX': 'Mexico', 'CHL': 'Chile', 'COL': 'Colombia', 'PER': 'Peru',
    'POL': 'Poland', 'CZE': 'Czech Republic', 'HUN': 'Hungary', 'GRC': 'Greece',
    'TUR': 'Turkey', 'SAU': 'Saudi Arabia', 'ARE': 'UAE', 'QAT': 'Qatar',
    'KWT': 'Kuwait', 'ZAF': 'South Africa'
}

# 2. Map the IMF's complex codes to clean, readable English names
indicator_map = {
    'MMRT_RT_PT_A_PT': 'Interbank_Rate',
    'DISR_RT_PT_A_PT': 'Policy_Rate',
    'GSTBILY_S3M_RT_PT_A_PT': '3M_TBill',
    'S13BOND_RT_PT_A_PT': '10Y_Bond'
}

# Join indicators for the query, but we'll loop countries to avoid timeouts
indicators_str = "+".join(indicator_map.keys())

print("Extracting 4 Interest Rates for 22 Emerging Markets...")

all_data = []

# 3. Safely loop through countries
for code, name in country_map.items():
    print(f" -> Processing {name} ({code})...")

    # 3-Dimension Key: AREA . INDICATORS . FREQUENCY
    dimension_key = f"{code}.{indicators_str}.M"

    # Removed the 'IMF.STA,' prefix to comply with 2025 routing rules
    url = f"https://api.imf.org/external/sdmx/2.1/data/MFS_IR/{dimension_key}"

    headers = {"Accept": "application/xml"}
    params = {"startPeriod": "1974", "endPeriod": "2026"}

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            it = ET.iterparse(io.StringIO(response.text))
            for _, el in it:
                _, _, el.tag = el.tag.rpartition('}')
            root = it.root

            for series in root.findall('.//Series'):
                # Catch the country code whether the IMF uses REF_AREA or COUNTRY
                raw_country = series.attrib.get('REF_AREA') or series.attrib.get('COUNTRY')
                raw_indicator = series.attrib.get('INDICATOR')

                clean_country = country_map.get(raw_country, name)
                clean_indicator = indicator_map.get(raw_indicator, raw_indicator)

                for obs in series.findall('.//Obs'):
                    obs_date = obs.attrib.get('TIME_PERIOD')
                    obs_val = obs.attrib.get('OBS_VALUE')

                    if obs_date and obs_val is not None:
                        try:
                            all_data.append({
                                'Date': obs_date,
                                'Country': clean_country,
                                'Indicator': clean_indicator,
                                'Rate': float(obs_val)
                            })
                        except ValueError:
                            pass
        else:
            print(f"    [!] Failed API call. Status: {response.status_code}")

        # Respect IMF API rate limits
        time.sleep(0.1)

    except Exception as e:
        print(f"    [!] Extraction failed for {name}: {e}")

# 4. Clean Dates and Structure the DataFrame
if all_data:
    df = pd.DataFrame(all_data)
    df['Date'] = df['Date'].str.replace('-M', '-')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Pivot the table: Dates as Rows, Multi-level Columns (Country -> Indicator)
    master_rates = df.pivot_table(
        index='Date',
        columns=['Country', 'Indicator'],
        values='Rate',
        aggfunc='first'
    )

    master_rates.sort_index(inplace=True)

    # Flatten the column names (e.g., 'Brazil_Policy_Rate')
    master_rates.columns = [f"{col[0]}_{col[1]}" for col in master_rates.columns.values]

    # Sort columns alphabetically to keep country variables grouped
    master_rates = master_rates.reindex(sorted(master_rates.columns), axis=1)

    master_rates.to_csv("Clean_EM_InterestRates_50Y.csv")

    print("\n--- DONE ---")
    print(f"Total rows processed: {len(df)}")
    print("Saved clean matrix to 'Clean_EM_InterestRates_50Y.csv'")
    print("\nPreview of your new clean columns:")

    preview_cols = [c for c in master_rates.columns if 'Brazil' in c]
    if preview_cols:
        print(master_rates[preview_cols].tail())

else:
    print("\nExtraction finished, but no numeric observations were found.")