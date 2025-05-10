# backtest.py
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import joblib
import logging
import os

# Import from local modules
from utils import load_processed_data, load_model_artefacts, PandasDataWithFeatures, FixedFractionPortfolioSizer
from trading_strategy import MLTradingStrategy

logger = logging.getLogger(__name__)

# --- Default Configuration (can be overridden by a calling script) ---
# These paths would typically be passed in or read from a config if this script is run standalone.
DEFAULT_PROCESSED_DATA_PATH = './processed_data/final_features.parquet'
DEFAULT_MODEL_PATH = './models/gdelt_stock_model.joblib'
DEFAULT_FEATURES_PATH = './models/gdelt_stock_model_features.joblib'
# DEFAULT_STOCK_INFO_PATH = 'data/Stock Information.csv' # User specified path

DEFAULT_INITIAL_CASH = 100000.0
DEFAULT_COMMISSION_PER_TRADE = 0.001
DEFAULT_SLIPPAGE_PERCENT = 0.0005
DEFAULT_POSITION_SIZE_PERCENT = 5.0  # Max % of portfolio value per trade
DEFAULT_BUY_THRESHOLD = 0.55
DEFAULT_SELL_THRESHOLD = 0.45


def run_backtest(processed_data_path=DEFAULT_PROCESSED_DATA_PATH,
                 model_path=DEFAULT_MODEL_PATH,
                 features_path=DEFAULT_FEATURES_PATH,
                 initial_cash=DEFAULT_INITIAL_CASH,
                 commission=DEFAULT_COMMISSION_PER_TRADE,
                 slippage_perc=DEFAULT_SLIPPAGE_PERCENT,
                 position_sizer_perc=DEFAULT_POSITION_SIZE_PERCENT,
                 buy_threshold=DEFAULT_BUY_THRESHOLD,
                 sell_threshold=DEFAULT_SELL_THRESHOLD,
                 backtest_start_date=None,  # e.g., '2022-01-01'
                 backtest_end_date=None,  # e.g., '2023-12-31'
                 plot_results=True):
    """
    Runs a backtest with the given parameters.
    """
    logger.info("--- Starting Backtest via run_backtest function ---")

    # 1. Load Model and Feature List
    model_pipeline, feature_columns = load_model_artefacts(model_path, features_path)
    if model_pipeline is None:
        logger.error("Model could not be loaded. Aborting backtest.")
        return None
    if feature_columns is None:
        logger.warning("Feature columns not loaded. Attempting to infer or assuming they are correctly in data.")
        # This scenario should be handled carefully; model needs specific features.

    # 2. Load Processed Data for Backtesting
    all_feature_data = load_processed_data(processed_data_path)
    if all_feature_data is None or all_feature_data.empty:
        logger.error("Failed to load processed data for backtesting. Aborting.")
        return None

    # Ensure feature_columns are present in the loaded data if not None
    if feature_columns:
        missing_model_features = [fc for fc in feature_columns if fc not in all_feature_data.columns]
        if missing_model_features:
            logger.error(
                f"Processed data is missing columns required by the model: {missing_model_features}. Aborting.")
            return None
    else:  # If feature_columns could not be loaded, we must infer them (less safe)
        logger.warning("Feature columns for the model were not loaded; inferring from data. This is risky.")
        # Define default exclusions for inferring features if feature_columns is None
        cols_to_exclude = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Ticker',
            'GLOBALEVENTID', 'SQLDATE', 'Actor1Name', 'Actor2Name', 'SOURCEURL', 'Title From URL',
            'CompanyName', 'Sector', 'Industry', 'CompanyName_Clean', 'openinterest'
        ]
        feature_columns = [col for col in all_feature_data.columns if col not in cols_to_exclude]
        if not feature_columns:
            logger.error("Could not infer any feature columns from the data. Aborting.")
            return None
        logger.info(f"Inferred {len(feature_columns)} feature columns for the backtest.")

    # 3. Setup Cerebro
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)  # cheat_on_open allows using Open for execution

    # 4. Add Strategy
    cerebro.addstrategy(MLTradingStrategy,
                        model=model_pipeline,
                        feature_cols=feature_columns,
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold,
                        printlog=True)

    # 5. Prepare and Add Data Feeds
    logger.info("Preparing and adding data feeds to Cerebro...")

    # Filter data by date if specified
    backtest_data_filtered = all_feature_data.copy()
    if backtest_start_date:
        backtest_data_filtered = backtest_data_filtered[
            backtest_data_filtered.index.get_level_values('Date') >= pd.to_datetime(backtest_start_date)]
    if backtest_end_date:
        backtest_data_filtered = backtest_data_filtered[
            backtest_data_filtered.index.get_level_values('Date') <= pd.to_datetime(backtest_end_date)]

    if backtest_data_filtered.empty:
        logger.error("No data available for the specified backtest period. Aborting.")
        return None

    tickers_in_data = backtest_data_filtered.index.get_level_values('Ticker').unique().tolist()
    added_tickers_count = 0
    for ticker in tickers_in_data:
        try:
            ticker_data = backtest_data_filtered.xs(ticker, level='Ticker').copy()
            if ticker_data.empty:
                logger.warning(f"No data for ticker {ticker} in period. Skipping.")
                continue

            # Ensure standard OHLCV columns are present (Backtrader needs Open, High, Low, Close, Volume)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in ticker_data.columns:
                    logger.error(f"Ticker {ticker} data missing standard column: {col}. Skipping.")
                    raise KeyError  # Skip this ticker
            if 'openinterest' not in ticker_data.columns:
                ticker_data['openinterest'] = 0.0  # Add dummy if not present

            # The PandasDataWithFeatures class now takes feature_cols in its __init__
            # However, Backtrader's core feed system usually expects lines to be class attributes.
            # A common way is to create a dynamic class or pass params to map columns.
            # For simplicity, let's ensure PandasDataWithFeatures uses the feature_columns
            # passed to addstrategy for its internal line mapping if that's feasible,
            # or ensure data feed has all potential features.
            # The current PandasDataWithFeatures in utils.py dynamically sets lines/params in __init__.

            data_feed = PandasDataWithFeatures(dataname=ticker_data.reset_index(),
                                               # PandasData needs datetime in a column or index
                                               feature_cols=feature_columns,
                                               dtformat=('%Y-%m-%d %H:%M:%S'),  # if Date is index
                                               datetime='Date',  # Column name for datetime
                                               open='Open', high='High', low='Low', close='Close', volume='Volume',
                                               openinterest='openinterest'
                                               )

            cerebro.adddata(data_feed, name=ticker)
            added_tickers_count += 1
            logger.debug(f"Added data feed for {ticker}")
        except KeyError:  # Catch if essential OHLCV columns were missing after check
            logger.warning(f"Skipped ticker {ticker} due to missing essential data columns.")
        except Exception as e:
            logger.error(f"Error preparing data feed for {ticker}: {e}", exc_info=True)

    if added_tickers_count == 0:
        logger.error("No data feeds were successfully added to Cerebro. Aborting backtest.")
        return None
    logger.info(f"Added data feeds for {added_tickers_count} tickers.")

    # 6. Configure Broker and Sizer
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(perc=slippage_perc)
    cerebro.addsizer(FixedFractionPortfolioSizer, perc=position_sizer_perc / 100.0)

    # 7. Add Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=252,
                        riskfreerate=0.0)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')

    # 8. Run Backtest
    logger.info(f"Starting Cerebro run with initial capital: ${initial_cash:,.2f}")
    results = cerebro.run()
    logger.info("Backtest run finished.")

    # 9. Analyze and Print Results
    if results and results[0]:  # results[0] is the strategy instance
        strategy_instance = results[0]
        final_value = cerebro.broker.getvalue()
        logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {((final_value / initial_cash) - 1) * 100:.2f}%")

        # Access analyzers (example)
        sharpe_ratio = strategy_instance.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
        max_drawdown = strategy_instance.analyzers.drawdown.get_analysis().max.drawdown
        sqn = strategy_instance.analyzers.sqn.get_analysis().get('sqn', 'N/A')

        print("\n--- Backtest Performance Summary ---")
        print(f"Starting Portfolio Value: {initial_cash:,.2f}")
        print(f"Final Portfolio Value:    {final_value:,.2f}")
        # ... (add more printouts as in original backtest.py or main.py) ...
        print(f"Annualized Sharpe Ratio:  {sharpe_ratio if isinstance(sharpe_ratio, str) else sharpe_ratio:.3f}")
        print(f"Maximum Drawdown:         {max_drawdown:.2f}%")
        print(f"SQN:                      {sqn if isinstance(sqn, str) else sqn:.2f}")

        trade_analysis = strategy_instance.analyzers.tradeanalyzer.get_analysis()
        if trade_analysis and trade_analysis.total and trade_analysis.total.total > 0:
            print("\n--- Trade Analysis ---")
            # ... (detailed trade stats) ...
        else:
            print("\nNo trades were executed.")

        # 10. Plotting (Optional)
        if plot_results:
            try:
                # Ensure matplotlib is configured for non-interactive backend if needed
                figure = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=True, iplot=False)[0][0]
                figure.set_size_inches(18, 10)
                plot_filename = 'backtest_results_plot.png'
                figure.savefig(plot_filename, dpi=300)
                logger.info(f"Backtest plot saved to {plot_filename}")
            except ImportError:
                logger.warning("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
            except Exception as e:
                logger.error(f"Error during plotting: {e}", exc_info=True)
        return results  # Return strategy results for potential further analysis
    else:
        logger.error("Backtest did not produce results.")
        return None


if __name__ == "__main__":
    # This block allows running backtest.py standalone with default or specified parameters.
    # However, main.py is intended as the primary orchestrator.
    # If running this standalone, ensure the processed data and model files exist at default paths.

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("--- Running backtest.py as standalone script ---")
    logger.warning("This is for isolated backtesting. For the full pipeline, run main.py.")

    # Example: Run with default parameters
    # To run, ensure 'processed_data/final_features.parquet' and model files in 'models/' exist.
    # You might need to run main.py with FORCE_DATA_PROCESSING and FORCE_MODEL_TRAINING first.

    # run_backtest(plot_results=True)

    logger.info("Standalone backtest.py execution finished. "
                "Uncomment `run_backtest()` call above with appropriate paths to execute.")