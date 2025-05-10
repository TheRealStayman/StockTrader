# trading_strategy.py
import backtrader as bt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MLTradingStrategy(bt.Strategy):
    params = (
        ('model', None),  # Trained scikit-learn pipeline
        ('feature_cols', None),  # List of feature column names model expects
        ('buy_threshold', 0.55),  # Probability threshold for BUY
        ('sell_threshold', 0.45),  # Probability threshold to exit (prob_buy < sell_threshold)
        ('printlog', True),
        # Removed ATR params for simplicity, can be added back if stop-loss logic is complex
        # ('atr_period', 14),
        # ('atr_multiplier', 2.0),
    )

    def __init__(self):
        if self.params.model is None:
            raise ValueError("A trained model pipeline must be provided.")
        if self.params.feature_cols is None or not self.params.feature_cols:
            raise ValueError("Feature columns list must be provided and non-empty.")

        self.model = self.params.model
        self.feature_cols = self.params.feature_cols

        # Store quick references to data lines (standard OHLCV)
        self.d_close = {d._name: d.close for d in self.datas}
        # self.d_open = {d._name: d.open for d in self.datas} # If needed
        # self.d_high = {d._name: d.high for d in self.datas} # If needed
        # self.d_low = {d._name: d.low for d in self.datas} # If needed

        # Store references to custom feature lines dynamically
        # This assumes feature_cols are correctly added to the PandasDataWithFeatures feed
        self.d_custom_features = {}
        for d in self.datas:
            ticker_features = {}
            for col_name in self.feature_cols:
                if hasattr(d.lines, col_name.lower()):  # Backtrader often lowercases line names
                    ticker_features[col_name] = d.lines.getlinealias(col_name.lower())
                elif hasattr(d.lines, col_name):  # Check original name
                    ticker_features[col_name] = d.lines.getlinealias(col_name)
                else:
                    logger.warning(
                        f"Feature '{col_name}' not found in data feed lines for {d._name}. Strategy may fail.")
            self.d_custom_features[d._name] = ticker_features

        self.orders = {d._name: None for d in self.datas}
        # self.stop_orders = {d._name: None for d in self.datas} # For stop-loss orders

        logger.info('MLTradingStrategy Initialized.')

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)  # Use first data for date
            log_msg = f'{dt.isoformat()} - {txt}'
            logger.info(log_msg)  # Use strategy's logger

    def notify_order(self, order):
        stock_name = order.data._name if order.data else 'UnknownOrderData'

        if order.status in [order.Submitted, order.Accepted]:
            self.log(f'{stock_name}: Order {order.getstatusname()}')
            self.orders[stock_name] = order  # Store active order
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f'{stock_name}: BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(
                    f'{stock_name}: SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.orders[stock_name] = None  # Clear completed order

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'{stock_name}: Order {order.getstatusname()}')
            self.orders[stock_name] = None  # Clear failed/cancelled order

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'{trade.data._name}: TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        current_date_obj = self.datas[0].datetime.datetime(0)  # Get datetime object for logging

        for i, d in enumerate(self.datas):
            stock_name = d._name
            position_size = self.getposition(d).size

            # Skip if an order is pending for this stock
            if self.orders.get(stock_name):
                continue

            # Ensure enough data for model (depends on lags used for features)
            # This check should ideally align with how NaNs were handled before training.
            # If data feed starts after NaNs from lagging are dropped, this might be less critical.
            if len(d) < max(
                    self.params.model.named_steps.get('preprocessor', {}).get('transformers_', [('num', None, [])])[0][
                        2] if 'preprocessor' in self.params.model.named_steps else [], default=1):  # Rough check
                # A better check would be based on the largest lag in self.feature_cols
                # For now, let's assume data feed has sufficient history or model handles initial missing features.
                pass

            # --- Feature Preparation for Prediction ---
            current_features_dict = {}
            all_features_present = True
            for col_name in self.feature_cols:
                try:
                    # Access feature from the custom feature lines
                    if col_name in self.d_custom_features[stock_name]:
                        current_features_dict[col_name] = self.d_custom_features[stock_name][col_name][0]
                    else:
                        # This case means the feature wasn't found during __init__ mapping
                        logger.error(
                            f"Critical: Feature '{col_name}' configured but not mapped in data feed for {stock_name} on {current_date_obj.date()}.")
                        all_features_present = False
                        break
                except IndexError:  # Not enough data points for the feature yet
                    logger.warning(
                        f"Not enough data for feature '{col_name}' for {stock_name} on {current_date_obj.date()}.")
                    all_features_present = False
                    break
                except Exception as e:
                    logger.error(
                        f"Error accessing feature '{col_name}' for {stock_name} on {current_date_obj.date()}: {e}")
                    all_features_present = False
                    break

            if not all_features_present:
                continue  # Skip prediction if features are not ready

            # Create DataFrame for prediction (model expects this)
            features_df = pd.DataFrame([current_features_dict], columns=self.feature_cols)

            # --- Prediction ---
            try:
                prediction_proba = self.model.predict_proba(features_df)
                prob_buy_signal = prediction_proba[0][1]  # Prob of class 1 (Up/Buy)
            except Exception as e:
                self.log(f'{stock_name}: Error during prediction on {current_date_obj.date()}: {e}')
                continue

            # --- Trading Logic ---
            if not position_size:  # No current position
                if prob_buy_signal > self.params.buy_threshold:
                    # Size calculation should be handled by Cerebro's sizer
                    self.log(
                        f'{stock_name}: BUY SIGNAL at {self.d_close[stock_name][0]:.2f} (Prob: {prob_buy_signal:.3f}) on {current_date_obj.date()}')
                    self.orders[stock_name] = self.buy(data=d)
            else:  # We have a position
                if prob_buy_signal < self.params.sell_threshold:
                    self.log(
                        f'{stock_name}: SELL SIGNAL (Close Position) at {self.d_close[stock_name][0]:.2f} (Prob: {prob_buy_signal:.3f}) on {current_date_obj.date()}')
                    self.orders[stock_name] = self.close(data=d)  # Close existing position

    def stop(self):
        self.log(f'Ending Portfolio Value: {self.broker.getvalue():.2f}', doprint=True)

# No __main__ block here, strategy is meant to be used by a backtesting script like main.py or backtest.py