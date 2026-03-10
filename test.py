import requests
import xml.etree.ElementTree as ET
import pandas as pd
import io

# 1. The 23 Emerging Markets + USA mapped to their required ISO-3 codes
country_map = {
    'CHN': 'China', 'IND': 'India', 'IDN': 'Indonesia', 'KOR': 'South Korea',
    'MYS': 'Malaysia', 'PHL': 'Philippines', 'THA': 'Thailand', 'BRA': 'Brazil',
    'MEX': 'Mexico', 'CHL': 'Chile', 'COL': 'Colombia', 'PER': 'Peru',
    'POL': 'Poland', 'CZE': 'Czech Republic', 'HUN': 'Hungary', 'GRC': 'Greece',
    'TUR': 'Turkey', 'SAU': 'Saudi Arabia', 'ARE': 'UAE', 'QAT': 'Qatar',
    'KWT': 'Kuwait', 'ZAF': 'South Africa', 'USA': 'United States'
}

countries = "+".join(country_map.keys())

# 2. Querying strictly for Total Reserves (TRGNV_REVS)
dimension_key = f"{countries}.TRGNV_REVS.USD.M"

url = f"https://api.imf.org/external/sdmx/2.1/data/IL/{dimension_key}"

print(f"Extracting strictly Total Reserves directly from IMF Azure Gateway...\nURL: {url}")

headers = {"Accept": "application/xml"}
params = {"startPeriod": "1974", "endPeriod": "2026"}

try:
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        print("\nData received! Parsing and building the Master CSV...")

        # Safely parse XML and strip namespaces
        it = ET.iterparse(io.StringIO(response.text))
        for _, el in it:
            _, _, el.tag = el.tag.rpartition('}')
        root = it.root

        all_data = []
        for series in root.findall('.//Series'):
            country_code = series.attrib.get('COUNTRY')

            for obs in series.findall('.//Obs'):
                all_data.append({
                    'Date': obs.attrib.get('TIME_PERIOD'),
                    'Country': country_map.get(country_code, country_code),
                    'Value': float(obs.attrib.get('OBS_VALUE'))
                })

        # 3. Clean and Pivot the Data
        df = pd.DataFrame(all_data)

        # Fix the IMF's date format (e.g., '2025-M11' -> '2025-11-01')
        df['Date'] = df['Date'].str.replace('-M', '-')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

        # Pivoting naturally populates missing month/country combinations with NaN
        master_df = df.pivot(index='Date', columns='Country', values='Value')
        master_df.sort_index(inplace=True)

        # 4. Save to Final CSV
        master_df.to_csv("Master_EM_Reserves_50Y.csv")

        print("\n--- DONE ---")
        print(f"Saved 'Master_EM_Reserves_50Y.csv' successfully.")
        print(f"Total Countries Locked In: {len(master_df.columns)}")
        print("\nPreview of final dataset (missing values will show as NaN):")
        print(master_df.tail())

    else:
        print(f"Failed. {response.status_code}: {response.text[:200]}")

except Exception as e:
    print(f"Extraction failed: {e}")

# import pandas as pd
# from fredapi import Fred
# import time
#
# # Initialize the FRED API client
# # Replace this with your actual API key
# fred = Fred(api_key='77758ecf1dcc01901e263a16df6a6f06')
#
# # The master list of tickers
# tickers = {
#     'China': 'TRESEGCNM052N', 'India': 'TRESEGINM052N', 'Indonesia': 'TRESEGIDM052N',
#     'South Korea': 'TRESEGKRM052N', 'Malaysia': 'TRESEGMYM052N',
#     'Philippines': 'TRESEGPHM052N', 'Thailand': 'TRESEGTHM052N', 'Brazil': 'TRESEGBRM052N',
#     'Mexico': 'TRESEGMXM052N', 'Chile': 'TRESEGCLM052N', 'Colombia': 'TRESEGCOM052N',
#     'Peru': 'TRESEGPEM052N', 'Poland': 'TRESEGPLM052N', 'Czech Republic': 'TRESEGCZM052N',
#     'Hungary': 'TRESEGHUM052N', 'Greece': 'TRESEGGRM052N', 'Turkey': 'TRESEGTRM052N',
#     'Saudi Arabia': 'TRESEGSAM052N', 'UAE': 'TRESEGAEM052N', 'Qatar': 'TRESEGQAM052N',
#     'Kuwait': 'TRESEGKWM052N', 'South Africa': 'TRESEGZAM052N', 'United States': 'TRESEGUSM052N'
# }
#
# print("Fetching data from FRED safely...")
#
# all_data = {}
# failed_countries = []
#
# # Loop through each country one by one
# for country, ticker in tickers.items():
#     try:
#         print(f"Attempting to fetch {country} ({ticker})...")
#         # Fetch the series
#         series = fred.get_series(ticker, observation_start='1974-01-01')
#         all_data[country] = series
#
#         # Adding a tiny pause to avoid hitting FRED's API rate limits
#         time.sleep(0.5)
#
#     except Exception as e:
#         # If a 404 happens (like Malaysia did), the script catches the error,
#         # logs it, and moves to the next country without crashing.
#         print(f" -> Failed to fetch {country}: Ticker might be missing or deprecated.")
#         failed_countries.append(country)
#
# # Combine all the successful pulls into one DataFrame
# if all_data:
#     df = pd.DataFrame(all_data)
#     df.index.name = 'Date'
#
#     # Save the clean data
#     df.to_csv("fred_monthly_reserves.csv")
#     print(f"\nSuccess! Extracted {len(all_data)} countries and saved to 'fred_monthly_reserves.csv'")
#
#     if failed_countries:
#         print(
#             f"\nNote: The following countries were missing from FRED's database and were skipped: {', '.join(failed_countries)}")
#         print("You will need to pull their data directly from their respective central bank websites.")
# else:
#     print("\nCritical Error: No data could be fetched for any country.")

# import pandas as pd
# from datetime import datetime
#
# urls = [
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.AE.AED.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.BR.BRL.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.CL.CLP.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.CN.CNY.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.CO.COP.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.GR.EUR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.HU.HUF.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.ID.IDR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.IN.INR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.KR.KRW.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.KW.KWD.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.MX.MXN.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.MY.MYR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.PE.PEN.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.PH.PHP.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.PK.PKR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.PL.PLN.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.QA.QAR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.SA.SAR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.TH.THB.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.TR.TRY.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.TW.TWD.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.US.USD.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.XW.XDR.A?format=csv",
#     "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.ZA.ZAR.A?format=csv"
# ]
#
#
# def get_clean_data(url_list):
#     all_data = []
#
#     for url in url_list:
#         try:
#             # Read CSV and parse dates
#             temp_df = pd.read_csv(url)
#
#             # BIS columns: 'TIME_PERIOD' is the date, 'OBS_VALUE' is the rate
#             # 'CURRENCY' tells us which one it is
#             temp_df = temp_df[['TIME_PERIOD', 'OBS_VALUE', 'CURRENCY']]
#             all_data.append(temp_df)
#             print(f"Successfully fetched: {url.split('.')[-2]}")
#         except Exception as e:
#             print(f"Failed to fetch {url}: {e}")
#
#     # Combine all into one tall dataframe
#     combined_df = pd.concat(all_data)
#
#     # Convert to datetime and filter for the last 30 years
#     combined_df['TIME_PERIOD'] = pd.to_datetime(combined_df['TIME_PERIOD'])
#     thirty_years_ago = datetime.now().year - 50
#     combined_df = combined_df[combined_df['TIME_PERIOD'].dt.year >= thirty_years_ago]
#
#     # Pivot to make it a clean time-series: Dates as rows, Currencies as columns
#     final_df = combined_df.pivot(index='TIME_PERIOD', columns='CURRENCY', values='OBS_VALUE')
#
#     return final_df
#
#
# # Run the extraction
# exchange_rates_df = get_clean_data(urls)
#
# # Save to CSV
# exchange_rates_df.to_csv("emerging_markets_50yr_rates.csv")
#
# print("\nExtraction Complete. Preview:")
# print(exchange_rates_df.tail())