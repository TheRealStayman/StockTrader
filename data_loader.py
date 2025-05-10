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
    Revised to address KeyError: ['Date'] and improve robustness.
    """
    logger.info(f"Attempting to load stock data from: {file_path}")
    try:
        # Read the first header row (Metrics: Price, Close, Close, ...)
        header_metrics_df = pd.read_csv(file_path, nrows=1, header=0)
        # Read the second header row (Tickers: Ticker, A, AA, ...)
        header_tickers_df = pd.read_csv(file_path, skiprows=1, nrows=1, header=None)

        # Extract metric types and ticker symbols
        metric_types_list = header_metrics_df.columns[1:].tolist()  # First item is "Price"
        stock_tickers_list = header_tickers_df.iloc[0, 1:].tolist()  # First item is "Ticker"

        if len(metric_types_list) != len(stock_tickers_list):
            logger.error(
                f"Header mismatch: {len(metric_types_list)} metric headers, {len(stock_tickers_list)} ticker headers."
            )
            return None

        # Read the data body, skipping the first 3 rows (two explicit headers + "Date,,," line)
        raw_data_df = pd.read_csv(file_path, skiprows=3, header=None, na_values=['null', 'NA', ''])

        if raw_data_df.empty:
            logger.error("No data rows found after skipping headers in the CSV file.")
            return None

        # The first column (index 0) of raw_data_df is the date
        # The subsequent columns are the stock data values
        num_expected_value_cols = len(metric_types_list)
        if raw_data_df.shape[1] != num_expected_value_cols + 1:  # +1 for the date column
            logger.error(
                f"Data column count mismatch: Expected {num_expected_value_cols + 1} columns (Date + values), "
                f"found {raw_data_df.shape[1]}. Check CSV structure and skiprows."
            )
            return None

        # Create a dictionary for DataFrame construction
        data_dict = {}

        # Add Date column first
        # The warning about date format inference suggests that some date strings might be problematic.
        # If you know the consistent format of your dates (e.g., 'YYYY-MM-DD'), provide it.
        # Example: data_dict['Date'] = pd.to_datetime(raw_data_df.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
        data_dict['Date'] = pd.to_datetime(raw_data_df.iloc[:, 0], errors='coerce')

        # Create MultiIndex for stock data columns
        column_multi_index = pd.MultiIndex.from_arrays([metric_types_list, stock_tickers_list],
                                                       names=['Metric', 'Ticker'])

        # Add stock data columns
        for i, col_tuple in enumerate(column_multi_index):
            data_dict[col_tuple] = raw_data_df.iloc[:, i + 1]  # i+1 because data_body_df's first col is date

        df_final = pd.DataFrame(data_dict)

        # Ensure 'Date' column is present and drop rows with NaT dates
        if 'Date' not in df_final.columns:
            logger.critical("Programming error: 'Date' column was not correctly added to df_final.")
            return None
        df_final.dropna(subset=['Date'], inplace=True)

        if df_final.empty:
            logger.warning("DataFrame is empty after dropping rows with invalid dates.")
            # You might want to return an empty DataFrame structured as expected by later stages
            # For now, returning None or an empty DataFrame.
            return pd.DataFrame()

        df_final.set_index('Date', inplace=True)

        # df_final.columns is now the MultiIndex: [('Close', 'A'), ('Close', 'AA'), ...]
        # The names of the levels should be ['Metric', 'Ticker']
        logger.debug(f"Columns before stacking: {df_final.columns}")
        logger.debug(f"Column names before stacking: {df_final.columns.names}")

        # Stack to bring 'Ticker' (level 1 of columns) and 'Metric' (level 0 of columns) into the index.
        # The order in `level` specifies which levels to pivot from columns to index.
        stock_data_long = df_final.stack(level=['Ticker', 'Metric'], dropna=False)

        # Unstack the 'Metric' level to turn metrics (Open, High, Low, Close, Volume) into columns.
        stock_data_long = stock_data_long.unstack(level='Metric')

        # The index should now be (Date, Ticker).
        stock_data_long.index.names = ['Date', 'Ticker']

        # Ensure standard OHLCV column names and convert to numeric
        expected_metrics = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Create a rename map if the original metric names in the file are different
        # e.g. if CSV uses "Close Price" map it to "Close"
        # For now, assuming the metric names in the CSV header are 'Open', 'High', 'Low', 'Close', 'Volume'

        cols_to_convert_to_numeric = [col for col in expected_metrics if col in stock_data_long.columns]
        for col in cols_to_convert_to_numeric:
            stock_data_long[col] = pd.to_numeric(stock_data_long[col], errors='coerce')

        missing_final_cols = [col for col in expected_metrics if col not in stock_data_long.columns]
        if missing_final_cols:
            logger.warning(f"After processing, some expected OHLCV columns are missing: {missing_final_cols}. "
                           f"Available columns: {stock_data_long.columns.tolist()}")

        stock_data_long.sort_index(inplace=True)
        logger.info(f"Successfully loaded and processed stock data. Shape: {stock_data_long.shape}")
        if not stock_data_long.empty:
            logger.info(f"Sample of processed stock data (first 5 rows of AAPL if available):\n"
                        f"{stock_data_long.xs('AAPL', level='Ticker', drop_level=False).head() if 'AAPL' in stock_data_long.index.get_level_values('Ticker') else stock_data_long.head()}")
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
        info_df['CompanyName_Clean'] = info_df['CompanyName'].apply(clean_company_name)
        info_df['Sector'].fillna('Unknown', inplace=True)
        info_df['Industry'].fillna('Unknown', inplace=True)

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