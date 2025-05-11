# data_loader.py
import pandas as pd
import json
import logging
from utils import clean_company_name  # Import from utils

logger = logging.getLogger(__name__)


# --- Data Loading Functions ---

def load_stock_data(file_path):
    """
    Loads daily stock data (OHLCV) from a CSV file with a specific multi-header format.
    Revised for robust MultiIndex column creation.
    """
    logger.info(f"Attempting to load stock data from: {file_path}")
    try:
        # Read header rows
        header_metrics_df = pd.read_csv(file_path, nrows=1, header=None)
        header_tickers_df = pd.read_csv(file_path, skiprows=1, nrows=1, header=None)

        metric_types_list = header_metrics_df.iloc[0, 1:].tolist()
        stock_tickers_list = header_tickers_df.iloc[0, 1:].tolist()

        if not metric_types_list or not stock_tickers_list:
            logger.error("Could not extract metric types or stock tickers. Check CSV format.")
            return None
        if len(metric_types_list) != len(stock_tickers_list):
            logger.error(f"Header mismatch: {len(metric_types_list)} metrics, {len(stock_tickers_list)} tickers.")
            return None

        # Read data body
        raw_data_df = pd.read_csv(file_path, skiprows=2, header=None, na_values=['null', 'NA', ''])
        if raw_data_df.empty:
            logger.error("No data rows found after skipping headers.")
            return None

        num_expected_value_cols = len(metric_types_list)
        if raw_data_df.shape[1] != num_expected_value_cols + 1: # +1 for date column
            logger.error(
                f"Data column count mismatch: Expected {num_expected_value_cols + 1}, found {raw_data_df.shape[1]}."
            )
            return None

        # Prepare MultiIndex for columns
        column_multi_index = pd.MultiIndex.from_arrays(
            [metric_types_list, stock_tickers_list],
            names=['Metric', 'Ticker']
        )

        # Prepare dictionary for stock data values, converting to numeric
        stock_data_values = {}
        for i, col_tuple in enumerate(column_multi_index):
            # raw_data_df.iloc[:, 0] is Date, data values start from raw_data_df.iloc[:, 1]
            stock_data_values[col_tuple] = pd.to_numeric(raw_data_df.iloc[:, i + 1], errors='coerce')

        # Create DataFrame for stock data, ensuring columns are the specified MultiIndex
        df_final = pd.DataFrame(stock_data_values, columns=column_multi_index)

        # Set Date index
        date_series = pd.to_datetime(raw_data_df.iloc[:, 0], errors='coerce')
        df_final.index = date_series
        df_final.index.name = 'Date'

        # Drop rows where the Date index itself is NaT (Not a Time)
        df_final = df_final[df_final.index.notna()]

        # Drop rows if all values across all metrics/tickers for a date are NaN
        df_final.dropna(axis=0, how='all', inplace=True)

        if df_final.empty:
            logger.warning("DataFrame is empty after initial creation and NaT date filtering.")
            # Consider returning an empty DataFrame with expected structure if needed downstream
            return df_final # Returns an empty DataFrame

        logger.debug(f"Columns after DataFrame creation: {df_final.columns}")
        logger.debug(f"Column names after DataFrame creation: {df_final.columns.names}") # Should be ['Metric', 'Ticker']

        # Stack to bring 'Ticker' (level 1) and 'Metric' (level 0) from columns into the index.
        stock_data_long = df_final.stack(level=['Ticker', 'Metric'], future_stack=True)

        # Unstack the 'Metric' level to turn metrics (Open, High, Low, Close, Volume) into columns.
        stock_data_long = stock_data_long.unstack(level='Metric')

        stock_data_long.index.names = ['Date', 'Ticker'] # Ensure final index names are set

        # Ensure standard OHLCV column names are present
        expected_metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
        actual_metrics_in_columns = stock_data_long.columns.tolist()

        missing_final_cols = [col for col in expected_metrics if col not in actual_metrics_in_columns]
        if missing_final_cols:
            logger.warning(f"After processing, some expected OHLCV columns are missing: {missing_final_cols}. "
                           f"Available columns: {actual_metrics_in_columns}")
            for col in missing_final_cols:
                stock_data_long[col] = pd.NA # Add missing columns with NA

        stock_data_long.sort_index(inplace=True)
        logger.info(f"Successfully loaded and processed stock data. Shape: {stock_data_long.shape}")

        if not stock_data_long.empty:
            sample_ticker = stock_tickers_list[0] if stock_tickers_list else None
            if sample_ticker and sample_ticker in stock_data_long.index.get_level_values('Ticker'):
                logger.info(f"Sample of processed stock data (first 5 rows of {sample_ticker}):\n"
                            f"{stock_data_long.xs(sample_ticker, level='Ticker', drop_level=False).head()}")
            else:
                logger.info(f"Sample of processed stock data (first 5 rows):\n{stock_data_long.head()}")
        return stock_data_long

    except FileNotFoundError:
        logger.error(f"Stock data file not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data or empty file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"General error processing stock data file {file_path}: {e}", exc_info=True)
        return None


def load_gdelt_data(file_path):
    logger.info(f"Attempting to load GDELT data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gdelt_raw = json.load(f)

        all_events = []
        for date_str, events_dict in gdelt_raw.items():
            try:
                event_date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(event_date):
                    logger.warning(f"Skipping invalid date format in GDELT data: {date_str}")
                    continue

                for event_key, event_data in events_dict.items():
                    if isinstance(event_data, dict):
                        # Provide defaults for essential keys
                        event_data.setdefault('GLOBALEVENTID', None)
                        event_data.setdefault('SQLDATE',
                                              event_date.strftime('%Y%m%d'))  # Use event_date if SQLDATE missing
                        event_data.setdefault('Actor1Name', None)
                        event_data.setdefault('Actor2Name', None)
                        event_data.setdefault('EventCode', None)
                        event_data.setdefault('GoldsteinScale', 0.0)
                        event_data.setdefault('NumArticles', 0)
                        event_data.setdefault('AvgTone', 0.0)
                        event_data.setdefault('SOURCEURL', '')
                        event_data.setdefault('Title From URL', '')
                        # Add other GDELT fields as needed with defaults

                        event_data['Date'] = event_date
                        all_events.append(event_data)
                    else:
                        logger.warning(f"Skipping non-dictionary event item: {event_key} on {date_str}")
            except Exception as e:
                logger.warning(f"Error processing date {date_str} in GDELT data: {e}")
                continue

        if not all_events:
            logger.warning("No valid events extracted from GDELT JSON.")
            return pd.DataFrame()

        gdelt_df = pd.DataFrame(all_events)

        if not gdelt_df.empty:
            logger.debug(f"GDELT data raw (before to_numeric/fillna) - describe a few key columns:\n"
                         f"{gdelt_df[['GoldsteinScale', 'AvgTone', 'NumArticles']].astype(str).describe(include='all')}")  # Use astype(str) for describe if mixed types before numeric conversion
            # Convert to numeric once to avoid issues with describe on mixed types from JSON
            gdelt_df['GoldsteinScale'] = pd.to_numeric(gdelt_df['GoldsteinScale'], errors='coerce')
            gdelt_df['AvgTone'] = pd.to_numeric(gdelt_df['AvgTone'], errors='coerce')
            gdelt_df['NumArticles'] = pd.to_numeric(gdelt_df['NumArticles'], errors='coerce')
            logger.debug(f"GDELT data (after to_numeric, before fillna) - describe key columns:\n"
                         f"{gdelt_df[['GoldsteinScale', 'AvgTone', 'NumArticles']].describe()}")

        gdelt_df['SQLDATE'] = pd.to_datetime(gdelt_df['SQLDATE'], format='%Y%m%d', errors='coerce')

        # Select and ensure essential columns exist
        desired_cols = ['Date', 'GLOBALEVENTID', 'SQLDATE', 'Actor1Name', 'Actor2Name',
                        'EventCode', 'GoldsteinScale', 'NumArticles', 'AvgTone',
                        'SOURCEURL', 'Title From URL']  # Add others if used
        for col in desired_cols:
            if col not in gdelt_df.columns:
                gdelt_df[col] = None  # Add missing columns with None/NaN

        gdelt_df = gdelt_df[desired_cols].copy()

        gdelt_df['GoldsteinScale'] = pd.to_numeric(gdelt_df['GoldsteinScale'], errors='coerce').fillna(0.0)
        gdelt_df['AvgTone'] = pd.to_numeric(gdelt_df['AvgTone'], errors='coerce').fillna(0.0)
        gdelt_df['NumArticles'] = pd.to_numeric(gdelt_df['NumArticles'], errors='coerce').fillna(0).astype(int)

        if not gdelt_df.empty:
            logger.debug(f"GDELT data (after fillna(0)) - describe key columns:\n"
                         f"{gdelt_df[['GoldsteinScale', 'AvgTone', 'NumArticles']].describe()}")


        gdelt_df['Actor1Name_Clean'] = gdelt_df['Actor1Name'].apply(clean_company_name)
        gdelt_df['Actor2Name_Clean'] = gdelt_df['Actor2Name'].apply(clean_company_name)

        logger.info(f"Successfully loaded and processed GDELT data. Shape: {gdelt_df.shape}")
        return gdelt_df

    except FileNotFoundError:
        logger.error(f"GDELT data file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}. Check file format.")
        return None
    except Exception as e:
        logger.error(f"Error processing GDELT data file {file_path}: {e}", exc_info=True)
        return None


def load_stock_info(file_path):
    logger.info(f"Attempting to load stock info data from: {file_path}")
    try:
        # Adjust column names based on the actual 'Stock Information.csv' structure
        # Common names: 'Ticker', 'Name', 'Sector', 'Industry'
        # The original code used: 'Ticker', 'Company', 'GICS Sector', 'GICS Sub-Industry'
        # Let's try to be flexible or assume specific names.
        # For "Stock Information.csv", let's assume 'Ticker', 'Security', 'GICS Sector', 'GICS Sub-Industry'
        # based on common S&P list formats. This needs to be verified with the actual file.
        # The outline mentions "Ticker, Company Name, Sector, Industry". Let's use that.

        use_cols = ['Ticker', 'Company', 'GICS Sector', 'GICS Sub-Industry']
        rename_map = {
            'Company': 'CompanyName',
            'GICS Sector': 'Sector',
            'GICS Sub-Industry': 'Industry'
        }

        try:
            info_df = pd.read_csv(file_path, usecols=use_cols)
        except ValueError as ve:  # Happens if columns are not found
            logger.warning(
                f"Could not read stock info with expected columns {use_cols}. Error: {ve}. Trying common alternatives.")
            # Try original code's expected columns as a fallback
            use_cols_alt = ['Ticker', 'Company Name', 'Sector', 'Industry']  # Adjusted based on outline
            rename_map_alt = {
                'Company Name': 'CompanyName',  # Original code used 'Company' -> 'CompanyName'
                'Sector': 'Sector',  # Original code used 'GICS Sector' -> 'Sector'
                'Industry': 'Industry'  # Original code used 'GICS Sub-Industry' -> 'Industry'
            }
            try:
                info_df = pd.read_csv(file_path, usecols=use_cols_alt)
                rename_map = rename_map_alt  # Use the alt map
            except ValueError as ve_alt:
                logger.error(f"Also failed to read stock info with alternative columns {use_cols_alt}. Error: {ve_alt}")
                logger.error(
                    "Please ensure 'Stock Information.csv' contains 'Ticker' and columns for company name, sector, and industry.")
                return None

        info_df.rename(columns=rename_map, inplace=True)
        # Pandas 3.0+ compatible way for inplace fillna on the DataFrame:
        fill_values = {'Sector': 'Unknown', 'Industry': 'Unknown'}
        info_df.fillna(value=fill_values, inplace=True)
        # Alternatively, assign back:
        # info_df['Sector'] = info_df['Sector'].fillna('Unknown')
        # info_df['Industry'] = info_df['Industry'].fillna('Unknown')
        info_df['CompanyName_Clean'] = info_df['CompanyName'].apply(clean_company_name)

        logger.info(f"Successfully loaded and processed stock info data. Shape: {info_df.shape}")
        return info_df
    except FileNotFoundError:
        logger.error(f"Stock info file not found at {file_path}")
        return None
    except KeyError as e:
        logger.error(
            f"Missing expected column in stock info file during processing: {e}. Check `use_cols` and `rename_map`.")
        return None
    except Exception as e:
        logger.error(f"Error processing stock info file {file_path}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # This block is for basic testing of the data loaders.
    # In a real application, use dedicated test scripts or the main pipeline.
    logger.info("--- Testing Data Loaders (Example Usage) ---")

    # Define dummy paths for testing - replace with actual paths if running standalone
    test_stock_path = 'data/stock_data.csv'
    test_gdelt_path = 'data/events.json'
    test_info_path = 'data/Stock Information.csv'

    # Create dummy files if they don't exist for basic test execution
    import os

    os.makedirs('data', exist_ok=True)
    if not os.path.exists(test_stock_path):
        logger.warning(f"Dummy file {test_stock_path} not found, creating a minimal version for testing.")
        # Price,Close,Close,High,High
        # Ticker,AAPL,MSFT,AAPL,MSFT
        # Date,,,
        # 2023-01-01,170,280,172,282
        # 2023-01-02,171,281,173,283
        # This format is tricky, the loader expects Open,High,Low,Close,Volume for each.
        # Minimal example for the loader's assumed structure for wide_to_long
        # Price,Close,High,Low,Open,Volume,Close,High,Low,Open,Volume
        # Ticker,AAPL,AAPL,AAPL,AAPL,AAPL,MSFT,MSFT,MSFT,MSFT,MSFT
        # Date,,,,,,,,,,
        # 2023-01-01,170,172,169,171,1000,280,282,279,281,2000
        # 2023-01-02,171,173,170,170,1100,281,283,280,280,2100
        # The provided stock_data_loader's wide_to_long assumes input like:
        # Date, Close_AAPL, High_AAPL, Low_AAPL, Open_AAPL, Volume_AAPL, Close_MSFT ...
        # The CSV reading part was very specific.
        # Let's use the format the original code example in feature_engineering.py test block created:
        stock_data_content = """Price,Close,Close,High,High,Low,Low,Open,Open,Volume,Volume
Ticker,AAPL,MSFT,AAPL,MSFT,AAPL,MSFT,AAPL,MSFT,AAPL,MSFT
Date,,,,,,,,,,
2023-01-01,170.0,280.0,172.0,282.0,169.0,279.0,171.0,281.0,100000,200000
2023-01-02,171.0,281.0,173.0,283.0,170.0,280.0,170.5,280.5,110000,210000
"""
        with open(test_stock_path, 'w') as f: f.write(stock_data_content)

    if not os.path.exists(test_gdelt_path):
        logger.warning(f"Dummy file {test_gdelt_path} not found, creating for testing.")
        gdelt_dummy_data = {
            "2023-01-01": {"Event1": {"Actor1Name": "APPLE", "AvgTone": 2.5, "Title From URL": "Apple good news"}},
            "2023-01-02": {
                "Event1": {"Actor1Name": "MICROSOFT CORP", "AvgTone": -1.0, "Title From URL": "MSFT bad news"}}
        }
        with open(test_gdelt_path, 'w') as f: json.dump(gdelt_dummy_data, f)

    if not os.path.exists(test_info_path):
        logger.warning(f"Dummy file {test_info_path} not found, creating for testing.")
        stock_info_data = {'Ticker': ['AAPL', 'MSFT'],
                           'Company Name': ['Apple Inc.', 'Microsoft Corporation'],
                           'Sector': ['Technology', 'Technology'],
                           'Industry': ['Hardware', 'Software']}
        pd.DataFrame(stock_info_data).to_csv(test_info_path, index=False)

    stock_data = load_stock_data(test_stock_path)
    if stock_data is not None:
        logger.info(f"Test: Stock Data Loaded. Head:\n{stock_data.head()}")

    gdelt_data = load_gdelt_data(test_gdelt_path)
    if gdelt_data is not None:
        logger.info(f"Test: GDELT Data Loaded. Head:\n{gdelt_data.head()}")

    stock_info = load_stock_info(test_info_path)
    if stock_info is not None:
        logger.info(f"Test: Stock Info Loaded. Head:\n{stock_info.head()}")