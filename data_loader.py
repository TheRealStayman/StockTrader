# data_loader.py
import pandas as pd
import json
from datetime import datetime
import logging

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def clean_company_name(name):
    """Removes common suffixes like Inc., Corp., Ltd. for better matching."""
    if pd.isna(name):
        return None
    name = str(name)
    suffixes = [' Inc', ' Corp', ' Ltd', ' PLC', '.', ','] # Add more as needed
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip().upper() # Convert to uppercase for case-insensitive matching

# --- Data Loading Functions ---

def load_stock_data(file_path):
    """
    Loads daily stock data (OHLCV) from a CSV file with a specific multi-header format.

    Args:
        file_path (str): The path to the stock data CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'Date', 'Ticker', 'Open',
                          'High', 'Low', 'Close', 'Volume', indexed by Date and Ticker.
                          Returns None if loading fails.
    """
    logging.info(f"Attempting to load stock data from: {file_path}")
    try:
        # Read the first two rows to get the structure (Price Type, Ticker)
        header_df = pd.read_csv(file_path, nrows=1) # Reads the first row (Price, Close, Close, ...)
        tickers_row = pd.read_csv(file_path, skiprows=1, nrows=1, header=None) # Reads the second row (Date, Ticker1, Ticker2, ...)

        # Extract price types and tickers
        price_types = header_df.columns[1:].tolist() # e.g., ['Close', 'Close', ..., 'High', ...]
        tickers = tickers_row.iloc[0, 1:].tolist() # e.g., ['A', 'AA', ...]

        # Check for consistency (should be 5 price types per ticker)
        if len(price_types) != len(tickers) * 5:
            logging.error("Stock data header format mismatch: Expected 5 columns (O,H,L,C,V) per ticker.")
            # Attempt a simpler read assuming standard OHLCV columns if the complex header fails
            try:
                logging.warning("Attempting fallback read assuming standard OHLCV columns...")
                stock_data_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if all(col in stock_data_df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                     logging.info("Fallback successful: Loaded with standard OHLCV columns.")
                     # Assuming single stock if fallback works, add Ticker column if possible (needs logic)
                     # This part is ambiguous without knowing the exact fallback format.
                     # For now, we'll assume it fails if the complex header isn't present.
                     raise ValueError("Fallback requires specific handling for ticker identification.")
                else:
                    raise ValueError("Fallback failed: Standard OHLCV columns not found.")
            except Exception as fallback_e:
                 logging.error(f"Fallback loading also failed: {fallback_e}")
                 return None


        # Create unique column names (e.g., 'A_Close', 'AA_Close', ...)
        new_columns = ['Date']
        current_ticker_idx = 0
        for i, p_type in enumerate(price_types):
            # Determine the correct ticker index based on the price type sequence
            if i > 0 and i % 5 == 0: # Assuming OHLCV order repeats every 5 columns
                current_ticker_idx += 1
            ticker = tickers[current_ticker_idx]
            new_columns.append(f"{ticker}_{p_type}")

        # Read the actual data, skipping header rows and assigning new names
        stock_data_df = pd.read_csv(file_path, skiprows=2, header=None, names=new_columns, na_values=['null', 'NA', ''])
        stock_data_df['Date'] = pd.to_datetime(stock_data_df['Date'], errors='coerce')
        stock_data_df = stock_data_df.dropna(subset=['Date']) # Remove rows where date parsing failed
        stock_data_df = stock_data_df.set_index('Date')

        # Convert data to numeric, coercing errors to NaN
        for col in stock_data_df.columns:
            stock_data_df[col] = pd.to_numeric(stock_data_df[col], errors='coerce')

        # Reshape from wide to long format
        stock_data_long = pd.wide_to_long(stock_data_df.reset_index(),
                                          stubnames=['Close', 'High', 'Low', 'Open', 'Volume'],
                                          i='Date',
                                          j='Ticker',
                                          sep='_',
                                          suffix=r'\w+').reset_index()

        # Sort for consistency and potential time-series operations
        stock_data_long.sort_values(by=['Ticker', 'Date'], inplace=True)
        stock_data_long.set_index(['Date', 'Ticker'], inplace=True) # Set multi-index

        logging.info(f"Successfully loaded and processed stock data. Shape: {stock_data_long.shape}")
        return stock_data_long

    except FileNotFoundError:
        logging.error(f"Stock data file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing stock data file {file_path}: {e}", exc_info=True)
        return None


def load_gdelt_data(file_path):
    """
    Loads and preprocesses GDELT event data from a JSON file.

    Args:
        file_path (str): The path to the GDELT JSON file.

    Returns:
        pandas.DataFrame: A DataFrame containing processed GDELT events,
                          or None if loading fails.
    """
    logging.info(f"Attempting to load GDELT data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gdelt_raw = json.load(f)

        all_events = []
        for date_str, events_dict in gdelt_raw.items():
            try:
                event_date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(event_date):
                    logging.warning(f"Skipping invalid date format in GDELT data: {date_str}")
                    continue

                for event_key, event_data in events_dict.items():
                    if isinstance(event_data, dict):
                        # Ensure essential keys exist, provide defaults if not
                        event_data.setdefault('GLOBALEVENTID', None)
                        event_data.setdefault('SQLDATE', None)
                        event_data.setdefault('Actor1Name', None)
                        event_data.setdefault('Actor2Name', None)
                        event_data.setdefault('EventCode', None)
                        event_data.setdefault('GoldsteinScale', 0.0) # Default to neutral
                        event_data.setdefault('NumArticles', 0)
                        event_data.setdefault('AvgTone', 0.0) # Default to neutral
                        event_data.setdefault('SOURCEURL', '')
                        event_data.setdefault('Title From URL', '')
                        event_data.setdefault('Actor1CountryCode', None)
                        event_data.setdefault('Actor2CountryCode', None)
                        event_data.setdefault('ActionGeo_CountryCode', None)
                        event_data.setdefault('ActionGeo_ADM1Code', None)
                        event_data.setdefault('ActionGeo_Lat', None)
                        event_data.setdefault('ActionGeo_Long', None)

                        event_data['Date'] = event_date # Add the parsed date
                        all_events.append(event_data)
                    else:
                        logging.warning(f"Skipping non-dictionary event item: {event_key} on {date_str}")

            except Exception as e:
                logging.warning(f"Error processing date {date_str} in GDELT data: {e}")
                continue

        if not all_events:
            logging.warning("No valid events extracted from GDELT JSON.")
            return pd.DataFrame() # Return empty DataFrame

        gdelt_df = pd.DataFrame(all_events)

        # Convert SQLDATE to datetime, coercing errors
        gdelt_df['SQLDATE'] = pd.to_datetime(gdelt_df['SQLDATE'], format='%Y%m%d', errors='coerce')

        # Select and potentially rename columns for clarity
        gdelt_df = gdelt_df[['Date', 'GLOBALEVENTID', 'SQLDATE', 'Actor1Name', 'Actor2Name',
                             'EventCode', 'GoldsteinScale', 'NumArticles', 'AvgTone',
                             'SOURCEURL', 'Title From URL', 'Actor1CountryCode',
                             'Actor2CountryCode', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
                             'ActionGeo_Lat', 'ActionGeo_Long']].copy() # Select desired columns

        # Handle potential type issues after loading
        gdelt_df['GoldsteinScale'] = pd.to_numeric(gdelt_df['GoldsteinScale'], errors='coerce').fillna(0.0)
        gdelt_df['AvgTone'] = pd.to_numeric(gdelt_df['AvgTone'], errors='coerce').fillna(0.0)
        gdelt_df['NumArticles'] = pd.to_numeric(gdelt_df['NumArticles'], errors='coerce').fillna(0).astype(int)

        # Clean Actor names for potential matching later
        gdelt_df['Actor1Name_Clean'] = gdelt_df['Actor1Name'].apply(clean_company_name)
        gdelt_df['Actor2Name_Clean'] = gdelt_df['Actor2Name'].apply(clean_company_name)

        logging.info(f"Successfully loaded and processed GDELT data. Shape: {gdelt_df.shape}")
        return gdelt_df

    except FileNotFoundError:
        logging.error(f"GDELT data file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}. Check file format.")
        return None
    except Exception as e:
        logging.error(f"Error processing GDELT data file {file_path}: {e}", exc_info=True)
        return None


def load_stock_info(file_path):
    """
    Loads stock information (Ticker, Company Name, Sector, etc.) from a CSV file.

    Args:
        file_path (str): The path to the stock info CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing stock information,
                          or None if loading fails.
    """
    logging.info(f"Attempting to load stock info data from: {file_path}")
    try:
        info_df = pd.read_csv(file_path, usecols=['Ticker', 'Company', 'GICS Sector', 'GICS Sub-Industry'])
        info_df.rename(columns={
            'Company': 'CompanyName',
            'GICS Sector': 'Sector',
            'GICS Sub-Industry': 'Industry'
        }, inplace=True)

        # Clean company names for matching
        info_df['CompanyName_Clean'] = info_df['CompanyName'].apply(clean_company_name)

        # Handle potential missing Sector/Industry data
        info_df['Sector'].fillna('Unknown', inplace=True)
        info_df['Industry'].fillna('Unknown', inplace=True)

        logging.info(f"Successfully loaded and processed stock info data. Shape: {info_df.shape}")
        return info_df

    except FileNotFoundError:
        logging.error(f"Stock info file not found at {file_path}")
        return None
    except KeyError as e:
        logging.error(f"Missing expected column in stock info file: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing stock info file {file_path}: {e}", exc_info=True)
        return None

# --- Main Execution Block (for testing) ---
if __name__ == "__main__":
    # Replace with your actual file paths for testing
    test_stock_path = 'path/to/your/stock_data.csv' # e.g., 'data/stock_prices.csv'
    test_gdelt_path = 'path/to/your/gdelt_events.json' # e.g., 'data/gdelt_data.json'
    test_info_path = 'path/to/your/stock_info.csv' # e.g., 'data/stock_info.csv'

    print("\n--- Testing Data Loaders ---")

    # Test Stock Data Loading
    stock_data = load_stock_data(test_stock_path)
    if stock_data is not None:
        print("\nStock Data Sample:")
        print(stock_data.head())
        print(f"\nStock Data Info:")
        stock_data.info()
    else:
        print("\nFailed to load stock data.")

    # Test GDELT Data Loading
    gdelt_data = load_gdelt_data(test_gdelt_path)
    if gdelt_data is not None:
        print("\nGDELT Data Sample:")
        print(gdelt_data.head())
        print(f"\nGDELT Data Info:")
        gdelt_data.info()
        # Check date range and specific columns
        print("\nGDELT Date Range:", gdelt_data['Date'].min(), "to", gdelt_data['Date'].max())
        print("\nGDELT Columns:", gdelt_data.columns)
        print("\nSample Cleaned Actor Names:")
        print(gdelt_data[['Actor1Name', 'Actor1Name_Clean']].head())
    else:
        print("\nFailed to load GDELT data.")

    # Test Stock Info Loading
    stock_info = load_stock_info(test_info_path)
    if stock_info is not None:
        print("\nStock Info Sample:")
        print(stock_info.head())
        print(f"\nStock Info:")
        stock_info.info()
        print("\nSample Cleaned Company Names:")
        print(stock_info[['CompanyName', 'CompanyName_Clean']].head())
    else:
        print("\nFailed to load stock info data.")