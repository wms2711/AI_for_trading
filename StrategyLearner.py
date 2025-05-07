""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Wang Ming Shen (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: mwang709 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 903859411 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import random  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import RTLearner as rtl
import BagLearner as bl
from indicators import mmt, sma, macd, rsi
import numpy as np
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt

class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # constructor  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission
        
        # Learner Parameters
        self.leaf_size = 5  # more than 5 to avoid overfitting
        self.bag_size = 20  # 20 or greater

        self.sma_window = 20
        self.rsi_window = 14
        self.mmt_window = 20
        self.macd_short = 12
        self.macd_long = 26
        self.macd_s = 9

        # Initialize learner
        self.learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": self.leaf_size}, bags=self.bag_size, boost=False, verbose=False)  	

    def author(self):
        return "mwang709"
    
    def study_group(self):
        return "mwang709"	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	 	 			  		 			     			  	 
    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        """
        # Get range of dates and retrieve trading data for that symbol, fill missing data and normalize
        dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
        prices = get_data([symbol], dates)[symbol]
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0]

        # Calculate indicators (SMA, MMT, MACD)
        sma_value = sma(prices, window=self.sma_window)
        mmt_value = mmt(prices, window=self.mmt_window)
        rsi_value = rsi(prices, window=self.rsi_window)
        # macd_line, macd_sig, macd_sign_signal = macd(prices, short=self.macd_short, long=self.macd_long, signal=self.macd_s)
        
        # Combine them into single feature and remove rows with any nan
        indicators = pd.DataFrame({
            'SMA': sma_value,
            'MMT': mmt_value,
            'RSI': rsi_value
        })
        indicators = indicators.dropna()

        # Create target / labels, buy if more than 2%, sell if less than 2%
        target_future_returns = 54
        future_returns = prices.shift(-target_future_returns) / prices - 1
        y = np.where(future_returns > 0.07, 1, np.where(future_returns < -0.07, -1, 0))

        # Combine X and y
        dates = indicators.index.intersection(future_returns.dropna().index)
        X_train = indicators.loc[dates].values
        y_train = y[indicators.index.get_indexer(dates)]

        # Start training
        self.learner.add_evidence(X_train, y_train)
        
    # this method should use the existing policy and test it against new data  		  	   		 	 	 			  		 			     			  	 
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # Get range of dates and retrieve trading data for that symbol, fill missing data and normalize
        dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
        prices = get_data([symbol], dates)[symbol]
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0]

        # Calculate indicators (SMA, MMT, MACD)
        sma_value = sma(prices, window=self.sma_window)
        mmt_value = mmt(prices, window=self.mmt_window)
        rsi_value = rsi(prices, window=self.rsi_window)

        # Combine them into single feature and remove rows with any nan
        indicators = pd.DataFrame({
            'SMA': sma_value,
            'MMT': mmt_value,
            'RSI': rsi_value
        })
        indicators = indicators.dropna()

        # Start prediction
        y_pred = self.learner.query(indicators.values)

        # Init trades, starts with initial starting date
        trades = {prices.index[0]: 0}
        current_position = 0

        # Calculate trades
        for i, date in enumerate(indicators.index):
            signal = y_pred[i]
            
            if signal == 1 and current_position != 1000:  # BUY signal
                trades[date] = 1000 - current_position
                current_position = 1000
            elif signal == -1 and current_position != -1000:  # SELL signal
                trades[date] = -1000 - current_position
                current_position = -1000	 	

        if prices.index[-1] not in trades:
            trades[prices.index[-1]] = 0

        return pd.DataFrame.from_dict(trades, orient='index', columns=['Shares'])    	

def plots_strats(symbol, in_start_date, in_end_date, out_start_date, out_end_date, commission, impact, sv):
    learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
    
    # In sample: Strategy Learner compute portvals, and port stats
    learner.add_evidence(symbol=symbol, sd=in_start_date, ed=in_end_date)
    strategy_trades = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    strategy_portval = compute_portvals(symbol=symbol, orders_file=strategy_trades, start_val=sv, commission=commission, impact=impact)
    norm_strategy_portval = strategy_portval / strategy_portval[0]
    strategy_cr, strategy_adr, strategy_sddr, strategy_sr = port_stats(strategy_portval)
    print("Strategy Learner performance", strategy_cr, strategy_adr, strategy_sddr, strategy_sr)
  		  	   		 	 	 			  		 			     			  	 
    # In sample: Plot Strategy strategy
    plt.title("Figure 3: Normalized strategy leaner (In-sample)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio values")
    plt.grid(b=True, linestyle='--')
    plt.plot(norm_strategy_portval, label="Strategy Learner Normalized", color="red")
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(strategy_trades.index[strategy_trades['Shares'] > 0].tolist(), ymin=ymin, ymax=ymax, label="LONG entry points", color="blue", linestyles='dashed', lw=0.5)
    plt.vlines(strategy_trades.index[strategy_trades['Shares'] < 0].tolist(), ymin=ymin, ymax=ymax, label="SHORT entry points", color="black", linestyles='dashed', lw=0.5)
    plt.legend()
    # plt.show()
    plt.savefig("images/Figure3.png", dpi=500)
    plt.clf()

    # Out sample: Strategy Learner compute portvals, and port stats
    learner.add_evidence(symbol=symbol, sd=out_start_date, ed=out_end_date)
    strategy_trades = learner.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
    strategy_portval = compute_portvals(symbol=symbol, orders_file=strategy_trades, start_val=sv, commission=commission, impact=impact)
    norm_strategy_portval = strategy_portval / strategy_portval[0]
    strategy_cr, strategy_adr, strategy_sddr, strategy_sr = port_stats(strategy_portval)
    print("Strategy Learner performance", strategy_cr, strategy_adr, strategy_sddr, strategy_sr)
  		  	   		 	 	 			  		 			     			  	 
    # Out sample: Plot Strategy strategy and benchmark
    plt.title("Figure 4: Normalized strategy leaner (Out-sample)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio values")
    plt.grid(b=True, linestyle='--')
    plt.plot(norm_strategy_portval, label="Strategy Learner Normalized", color="red")
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(strategy_trades.index[strategy_trades['Shares'] > 0].tolist(), ymin=ymin, ymax=ymax, label="LONG entry points", color="blue", linestyles='dashed', lw=0.5)
    plt.vlines(strategy_trades.index[strategy_trades['Shares'] < 0].tolist(), ymin=ymin, ymax=ymax, label="SHORT entry points", color="black", linestyles='dashed', lw=0.5)
    plt.legend()
    # plt.show()
    plt.savefig("images/Figure4.png", dpi=500)
    plt.clf()
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")
    symbol = "JPM"
    in_start_date = dt.datetime(2008,1,1)
    in_end_date = dt.datetime(2009,12,31)
    out_start_date = dt.datetime(2010, 1, 1)
    out_end_date = dt.datetime(2011, 12, 31)
    commission = 9.95
    impact = 0.005
    sv = 100000
    plots_strats(symbol, in_start_date, in_end_date, out_start_date, out_end_date, commission, impact, sv)