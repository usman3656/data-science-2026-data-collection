import requests
import xml.etree.ElementTree as ET
import pandas as pd
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
countries = "+".join(country_map.keys())

# 2. Using the exact working URL structure from your Reserves script
dimension_key = f"{countries}.ITG..IX.M"
url = f"https://api.imf.org/external/sdmx/2.1/data/ITG/{dimension_key}"

print(f"Extracting CPI using dynamic attribute mapping...\nURL: {url}")

headers = {"Accept": "application/xml"}
params = {"startPeriod": "1974", "endPeriod": "2026"}

try:
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        print("\nData received! Parsing ALL XML attributes safely...")

        it = ET.iterparse(io.StringIO(response.text))
        for _, el in it:
            _, _, el.tag = el.tag.rpartition('}')
        root = it.root

        all_data = []
        for series in root.findall('.//Series'):
            # Grab EVERY attribute the IMF uses so we literally cannot miss the category
            series_attribs = series.attrib.copy()

            # Map the Country safely
            country_code = series_attribs.get('REF_AREA') or series_attribs.get('COUNTRY')
            clean_country = country_map.get(country_code, country_code)

            for obs in series.findall('.//Obs'):
                obs_date = obs.attrib.get('TIME_PERIOD')
                obs_val = obs.attrib.get('OBS_VALUE')

                if obs_date and obs_val is not None:
                    try:
                        # Build the row dictionary dynamically
                        row_data = series_attribs.copy()
                        row_data['Country'] = clean_country
                        row_data['Date'] = obs_date
                        row_data['CPI_Index'] = float(obs_val)
                        all_data.append(row_data)
                    except ValueError:
                        pass

        if all_data:
            df = pd.DataFrame(all_data)
            df['Date'] = df['Date'].str.replace('-M', '-')
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

            # 3. Auto-Detect the Headline Category
            headline_df = pd.DataFrame()
            category_col = None
            headline_code = None

            for col in df.columns:
                if col not in ['Date', 'Country', 'CPI_Index', 'REF_AREA']:
                    unique_vals = df[col].astype(str).unique()
                    # _T and CP00 are the universal SDMX codes for "Total / All Items"
                    for code in ['CP00', '_T', 'ALL']:
                        if code in unique_vals:
                            category_col = col
                            headline_code = code
                            break
                if category_col:
                    break

            # 4. Filter, Pivot, and create Country-Specific Columns
            if category_col:
                print(
                    f"\n-> SUCCESS: Found Headline Inflation indicator '{headline_code}' inside the hidden column '{category_col}'.")

                # Filter down to just the true Headline Inflation
                headline_df = df[df[category_col] == headline_code]

                # Pivot so each country gets its own column
                master_cpi = headline_df.pivot_table(index='Date', columns='Country', values='CPI_Index',
                                                     aggfunc='first')
                master_cpi.sort_index(inplace=True)

                # Rename the base index columns
                master_cpi.columns = [f"{col}_CPI_Index" for col in master_cpi.columns]

                # 5. CALCULATE TRUE INFLATION PERCENTAGES
                print("Calculating Year-over-Year (YoY) and Month-over-Month (MoM) Inflation percentages...")
                for country in list(master_cpi.columns):
                    base_country_name = country.replace('_CPI_Index', '')

                    # YoY Inflation: (Current Month - Same Month Last Year) / Same Month Last Year * 100
                    master_cpi[f"{base_country_name}_Inflation_YoY_%"] = master_cpi[country].pct_change(
                        periods=12) * 100

                    # MoM Inflation: (Current Month - Last Month) / Last Month * 100
                    master_cpi[f"{base_country_name}_Inflation_MoM_%"] = master_cpi[country].pct_change(periods=1) * 100

                # Sort the columns alphabetically to group each country's metrics together
                master_cpi = master_cpi.reindex(sorted(master_cpi.columns), axis=1)

                master_cpi.to_csv("Master_EM_Y.csv")

                print("\n--- CLEANUP COMPLETE ---")
                print("Saved fully calculated matrix to 'Master_EM_Inflation_Rates_50Y.csv'")
                print(f"Total Columns Generated: {len(master_cpi.columns)}")

                print("\nPreview of the calculated Inflation Rates for Brazil & USA:")
                preview_cols = [c for c in master_cpi.columns if 'United States' in c or 'Brazil' in c]
                print(master_cpi[preview_cols].tail())

            else:
                df.to_csv("Master_EM_CPI_Raw_Full.csv", index=False)
                print("\n[!] Could not auto-detect the 'Total' code. Saved to 'Master_EM_CPI_Raw_Full.csv'.")

        else:
            print("\nXML returned, but no matching numeric observations found.")

    else:
        print(f"Failed. {response.status_code}: {response.text[:200]}")

except Exception as e:
    print(f"Extraction failed: {e}")