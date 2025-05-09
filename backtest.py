# backtest.py

import backtrader as bt
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

# Import functions/classes from other modules
from data_loader import load_stock_data, load_gdelt_data, load_stock_info  # For potential re-run/verification
from feature_engineering import load_processed_data  # Primarily use this
from trading_strategy import MLTradingStrategy  # Import the strategy

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
PROCESSED_DATA_PATH = './processed_data/final_features.parquet'  # Input data for backtesting
MODEL_PATH = './models/gdelt_stock_model.joblib'  # Input trained model
STOCK_INFO_PATH = 'path/to/your/stock_info.csv'  # Needed to get the list of tickers if not inferred from data

# --- Backtesting Parameters ---
INITIAL_CASH = 100000.0
COMMISSION_PER_TRADE = 0.001  # Example: 0.1% commission
SLIPPAGE_PERCENT = 0.0005  # Example: 0.05% slippage (adjust based on broker/market)
POSITION_SIZE_PERCENT = 5.0  # Max % of portfolio value per trade

# Optional: Define a specific backtest period (if None, uses all data in the file)
# Ensure this period corresponds to the test set used during training or a separate hold-out period
BACKTEST_START_DATE = None  # e.g., '2022-01-01'
BACKTEST_END_DATE = None  # e.g., '2023-12-31'

# --- Load Model and Data ---

# Load the trained model pipeline
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logging.info(f"Model pipeline loaded successfully from {MODEL_PATH}")
    # Extract feature names from the preprocessor step if possible
    # This assumes a ColumnTransformer named 'preprocessor' with a transformer named 'num'
    try:
        # If using ColumnTransformer with named transformers:
        preprocessor_step = model_pipeline.named_steps['preprocessor']
        # Find the numeric transformer to get feature names after potential imputation/scaling
        numeric_transformer = next((item[1] for item in preprocessor_step.transformers_ if item[0] == 'num'), None)
        if numeric_transformer and hasattr(numeric_transformer, 'get_feature_names_out'):
            # If StandardScaler was used, feature names might be lost, rely on pre-defined list
            # A better approach is to save the feature list used during training alongside the model
            logging.warning(
                "Attempting to get feature names from pipeline - might be unreliable. Ensure 'feature_cols' below is correct.")
            # feature_columns = numeric_transformer.get_feature_names_out() # This might not work depending on pipeline structure
            # Manually define based on feature_engineering.py output if needed
            # feature_columns = [...] # Define explicitly here or load from a saved file
        elif 'classifier' in model_pipeline.named_steps and hasattr(model_pipeline.named_steps['classifier'],
                                                                    'feature_name_'):
            # For models like LightGBM that store feature names after fitting the pipeline
            feature_columns = model_pipeline.named_steps['classifier'].feature_name_
            logging.info(f"Extracted {len(feature_columns)} feature names from the model.")
        else:
            logging.error("Could not automatically extract feature names. Please define 'feature_columns' manually.")
            # *** Define feature_columns manually here based on feature_engineering.py ***
            # Example: feature_columns = ['SMA_20', 'RSI_14', ..., 'gdelt_tone_mean_lag_5']
            feature_columns = None  # Force error if not defined
            # exit() # Or exit if manual definition is required

    except Exception as e:
        logging.error(f"Could not extract feature names from the pipeline: {e}. Define 'feature_columns' manually.")
        # *** Define feature_columns manually here based on feature_engineering.py ***
        feature_columns = None
        # exit() # Or exit

except FileNotFoundError:
    logging.error(f"Model file not found at {MODEL_PATH}. Run model_training.py first.")
    exit()
except Exception as e:
    logging.error(f"Error loading model: {e}", exc_info=True)
    exit()

# Load the processed data
all_data = load_processed_data(PROCESSED_DATA_PATH)
if all_data is None or all_data.empty:
    logging.error(f"Failed to load processed data from {PROCESSED_DATA_PATH}. Cannot run backtest.")
    exit()

# Ensure feature_columns is set
if feature_columns is None:
    # Try inferring from the loaded data frame if not extracted from model
    potential_feature_cols = [col for col in all_data.columns if col not in
                              ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target',
                               'SQLDATE', 'GLOBALEVENTID', 'Actor1Name', 'Actor2Name',
                               'SOURCEURL', 'Title From URL', 'CompanyName', 'Sector', 'Industry'] and col not in [
                                  'open', 'high', 'low', 'close', 'volume', 'openinterest']]
    if not potential_feature_cols:
        logging.error("Could not determine feature columns. Exiting.")
        exit()
    feature_columns = potential_feature_cols
    logging.warning(
        f"Inferred {len(feature_columns)} feature names from data columns. Ensure this matches the training features.")
    # print("Inferred Features:", feature_columns) # Uncomment for debugging


# --- Prepare Data for Backtrader ---

# Create a custom PandasData feed to include all features
class PandasDataWithFeatures(bt.feeds.PandasData):
    lines = tuple(feature_columns)  # Add feature columns as lines
    # Define parameters for the feed, mapping DataFrame columns to Backtrader lines
    params = tuple(
        [(col, -1) for col in feature_columns]  # Map each feature column name to a line
        + [('datetime', None),  # Use the DataFrame index for datetime
           ('open', 'Open'),  # Map DataFrame 'Open' column to 'open' line
           ('high', 'High'),  # Map DataFrame 'High' column to 'high' line
           ('low', 'Low'),  # Map DataFrame 'Low' column to 'low' line
           ('close', 'Close'),  # Map DataFrame 'Close' column to 'close' line
           ('volume', 'Volume'),  # Map DataFrame 'Volume' column to 'volume' line
           ('openinterest', -1)]  # Use -1 if 'openinterest' column doesn't exist
    )


# --- Setup Backtrader Cerebro Engine ---
cerebro = bt.Cerebro(stdstats=False)  # Disable default observers initially

# Add Strategy
cerebro.addstrategy(MLTradingStrategy,
                    model=model_pipeline,
                    feature_cols=feature_columns,
                    buy_threshold=0.55,  # Example threshold
                    sell_threshold=0.45,  # Example threshold
                    printlog=False)  # Set to True for detailed strategy logs

# Add Data Feeds
logging.info("Preparing and adding data feeds to Cerebro...")
tickers_in_data = all_data.index.get_level_values('Ticker').unique().tolist()

# Optional: Filter tickers if needed (e.g., based on stock_info or a predefined list)
# tickers_to_backtest = ['AAPL', 'MSFT', 'GOOGL'] # Example subset
tickers_to_backtest = tickers_in_data  # Use all available tickers

added_tickers = []
for ticker in tickers_to_backtest:
    try:
        # Select data for the current ticker
        ticker_data = all_data.xs(ticker, level='Ticker').copy()

        # Apply date filtering if specified
        if BACKTEST_START_DATE:
            ticker_data = ticker_data[ticker_data.index >= pd.to_datetime(BACKTEST_START_DATE)]
        if BACKTEST_END_DATE:
            ticker_data = ticker_data[ticker_data.index <= pd.to_datetime(BACKTEST_END_DATE)]

        # Check if data exists for the period
        if ticker_data.empty:
            logging.warning(f"No data found for ticker {ticker} in the specified date range. Skipping.")
            continue

        # Ensure required columns exist and add 'openinterest' if missing
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + feature_columns
        missing_cols = [col for col in required_cols if col not in ticker_data.columns]
        if missing_cols:
            logging.warning(f"Missing columns for ticker {ticker}: {missing_cols}. Skipping.")
            continue

        if 'openinterest' not in ticker_data.columns:
            ticker_data['openinterest'] = 0

        # Add data feed to Cerebro
        data_feed = PandasDataWithFeatures(dataname=ticker_data)
        cerebro.adddata(data_feed, name=ticker)
        added_tickers.append(ticker)
        logging.debug(f"Added data feed for {ticker}")

    except KeyError:
        logging.warning(f"Ticker {ticker} not found in the processed data index. Skipping.")
    except Exception as e:
        logging.error(f"Error preparing data feed for {ticker}: {e}", exc_info=True)

if not added_tickers:
    logging.error("No data feeds were added to Cerebro. Exiting.")
    exit()

logging.info(f"Added data feeds for {len(added_tickers)} tickers.")

# Configure Broker
cerebro.broker.set_cash(INITIAL_CASH)
cerebro.broker.setcommission(commission=COMMISSION_PER_TRADE)
# Add slippage (optional) - Simple percentage slippage
cerebro.broker.set_slippage_perc(perc=SLIPPAGE_PERCENT)


# Configure Sizer
# Allocate a percentage of the *available cash* for each new position
# cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_SIZE_PERCENT)
# Or, allocate a percentage of the *total portfolio value*
class FixedFractionPortfolioSizer(bt.Sizer):
    params = (('perc', 0.05),)  # Default 5% of portfolio value per trade

    def _getsizing(self, comminfo, cash, data, isbuy):
        size = self.broker.getvalue() * self.p.perc / data.close[0]
        return int(size)  # Return integer number of shares


cerebro.addsizer(FixedFractionPortfolioSizer, perc=POSITION_SIZE_PERCENT / 100.0)

# Add Analyzers
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=252,
                    riskfreerate=0.0)  # Annualized Sharpe
cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')
cerebro.addanalyzer(btanalyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')  # System Quality Number

# --- Run Backtest ---
logging.info(f"Starting backtest with initial capital: ${INITIAL_CASH:,.2f}")
results = cerebro.run()
logging.info("Backtest finished.")

# --- Analyze Results ---
if results and results[0]:
    strategy_instance = results[0]
    final_value = cerebro.broker.getvalue()
    logging.info(f"Final Portfolio Value: ${final_value:,.2f}")
    logging.info(f"Total Return: {((final_value / INITIAL_CASH) - 1) * 100:.2f}%")

    # Access analyzers
    sharpe_ratio = strategy_instance.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
    max_drawdown = strategy_instance.analyzers.drawdown.get_analysis().max.drawdown
    trade_analysis = strategy_instance.analyzers.tradeanalyzer.get_analysis()
    sqn = strategy_instance.analyzers.sqn.get_analysis().get('sqn', 'N/A')

    print("\n--- Backtest Performance Summary ---")
    print(f"Starting Portfolio Value: {INITIAL_CASH:,.2f}")
    print(f"Final Portfolio Value:    {final_value:,.2f}")
    print(f"Total Return:             {((final_value / INITIAL_CASH) - 1) * 100:.2f}%")
    print(f"Annualized Sharpe Ratio:  {sharpe_ratio:.3f}" if isinstance(sharpe_ratio,
                                                                        float) else f"Annualized Sharpe Ratio:  {sharpe_ratio}")
    print(f"Maximum Drawdown:         {max_drawdown:.2f}%")
    print(f"SQN:                      {sqn:.2f}" if isinstance(sqn, float) else f"SQN:                      {sqn}")

    if trade_analysis and trade_analysis.total and trade_analysis.total.total > 0:
        print("\n--- Trade Analysis ---")
        print(f"Total Trades:             {trade_analysis.total.total}")
        print(f"Total Closed Trades:      {trade_analysis.total.closed}")
        print(f"Winning Trades:           {trade_analysis.won.total}")
        print(f"Losing Trades:            {trade_analysis.lost.total}")
        if trade_analysis.total.closed > 0:
            print(f"Win Rate:                 {trade_analysis.won.total / trade_analysis.total.closed * 100:.2f}%")
            print(f"Average Win ($):        {trade_analysis.won.pnl.average:.2f}")
            print(f"Average Loss ($):       {trade_analysis.lost.pnl.average:.2f}")
            profit_factor = abs(
                trade_analysis.won.pnl.total / trade_analysis.lost.pnl.total) if trade_analysis.lost.pnl.total != 0 else float(
                'inf')
            print(f"Profit Factor:            {profit_factor:.2f}")
            print(f"Avg Trade PnL ($):      {trade_analysis.pnl.net.average:.2f}")
        else:
            print("No trades were closed during the backtest.")
    else:
        print("\n--- Trade Analysis ---")
        print("No trades were executed during the backtest.")

    # --- Plotting (Optional) ---
    # Ensure matplotlib is installed: pip install matplotlib
    try:
        import matplotlib

        # You might need to set a non-interactive backend if running on a server without a display
        # matplotlib.use('Agg')
        print("\nGenerating plot...")
        cerebro.plot(style='candlestick', barup='green', bardown='red')
        # The plot will be displayed or saved depending on the environment/matplotlib backend.
        # You might need plt.show() or fig.savefig('backtest_results.png')
        print("Plot generation command issued.")
    except ImportError:
        logging.warning("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
    except Exception as e:
        logging.error(f"Error during plotting: {e}")

else:
    logging.error("Backtest did not produce results.")