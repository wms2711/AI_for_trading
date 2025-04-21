import datetime as dt
import pandas as pd
from util import get_data
from indicators import mmt, sma, macd, rsi
import numpy as np
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt

class ManualStrategy(object):
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.sma_window = 20
        self.rsi_window = 14
        self.mmt_window = 20
        self.macd_short = 12
        self.macd_long = 26
        self.macd_s = 9
    
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=100000,
    ):
        pass

    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=100000,
    ):
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
        macd_line, macd_sig, macd_sign_signal = macd(prices, short=self.macd_short, long=self.macd_long, signal=self.macd_s)

        # Turn indicators into signals
        sma_signals = self.sma_signal(prices, sma_value)
        mmt_signals = self.mmt_signal(mmt_value)
        rsi_signals = self.rsi_signal(rsi_value)
        macd_signals = self.macd_signal(macd_sign_signal)

        # Combine signals and decide long and short signals
        combine_signals = pd.Series(0, index=prices.index)
        # Long condition
        long_cond = (
                    (sma_signals < 0).astype(int) + 
                    (rsi_signals < 0.3).astype(int) +
                    (mmt_signals < 0).astype(int) 
                    # (macd_signals > 0).astype(int)
                    ) >= 3
        # Short condition
        short_cond = (
                    (sma_signals > 0).astype(int) +
                    (rsi_signals > 0.7).astype(int) +
                    (mmt_signals > 0).astype(int) 
                    # (macd_signals < 0).astype(int)
                    ) >= 3
        combine_signals[long_cond] = 1
        combine_signals[short_cond] = -1

        # Init trades, starts with initial starting date
        trades = {prices.index[0]: 0}
        current_position = 0

        # Loop through all dates and decides whether to buy or sell
        for date, signal in combine_signals.items():
            if signal == 1 and current_position != 1000:
                trades[date] = 1000 - current_position
                current_position = 1000
            elif signal == -1 and current_position != -1000:
                trades[date] = -1000 - current_position
                current_position = -1000

        if prices.index[-1] not in trades:
            trades[prices.index[-1]] = 0
            
        return pd.DataFrame.from_dict(trades, orient='index', columns=['Shares'])

    def sma_signal(self, prices, sma_value):
        # Using numpy sign function to indicate bull (+1), bear (-1) ot equal (0)
        return np.sign(prices - sma_value)

    def mmt_signal(self, mmt_value):
        # 1 when momentum positive, -1 when negative
        return np.sign(mmt_value)

    def rsi_signal(self, rsi):
        return rsi / 100 
    
    def macd_signal(self, macd_sign_signal):
        # Using numpy sign function to indicate bull (+1), bear (-1) ot equal (0)
        return np.sign(macd_sign_signal)
    
    def benchMarkPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=100000,
    ):
        # Get range of dates and retrieve trading data for that symbol, fill missing data and normalize
        dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
        prices = get_data([symbol], dates)[symbol]
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        # Buy from first day and add last trading day
        trades = {prices.index[0]: 1000, prices.index[-1]: 0}

        return pd.DataFrame.from_dict(trades, orient='index', columns=['Shares'])

def plots_manual(symbol, in_start_date, in_end_date, out_start_date, out_end_date, commission, impact, sv):
    learner = ManualStrategy(verbose=False, impact=impact, commission=commission)

    # In sample: Manual strategy compute portvals, and port stats
    manual_trades = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    manual_portval = compute_portvals(symbol=symbol, orders_file=manual_trades, start_val=sv, commission=commission, impact=impact)
    norm_manual_portval = manual_portval / manual_portval[0]
    manual_cr, manual_adr, manual_sddr, manual_sr = port_stats(manual_portval)
    print("Manual strategy performance", manual_cr, manual_adr, manual_sddr, manual_sr)

    # In sample: Benchmark compute portvals and port stats
    bench_trades = learner.benchMarkPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    bench_portvals = compute_portvals(symbol=symbol, orders_file=bench_trades, start_val=sv, commission=commission, impact=impact)
    norm_bench_portvals = bench_portvals / bench_portvals[0]
    bench_cr, bench_adr, bench_sddr, bench_sr = port_stats(bench_portvals)
    print("Benchmark performance", bench_cr, bench_adr, bench_sddr, bench_sr)

    # In sample: Plot manual strategy and benchmark
    plt.title("Figure 1: Normalized manual strategy vs Benchmark (In-sample)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio values")
    plt.grid(b=True, linestyle='--')
    plt.plot(norm_manual_portval, label="Manual Strategy", color="red")
    plt.plot(norm_bench_portvals, label="Benchmark", color="purple")
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(manual_trades.index[manual_trades['Shares'] > 0].tolist(), ymin=ymin, ymax=ymax, label="LONG entry points", color="blue", linestyles='dashed', lw=0.5)
    plt.vlines(manual_trades.index[manual_trades['Shares'] < 0].tolist(), ymin=ymin, ymax=ymax, label="SHORT entry points", color="black", linestyles='dashed', lw=0.5)
    plt.legend()
    # plt.show()
    plt.savefig("images/Figure1.png", dpi=500)
    plt.clf()


    # Out sample: Manual strategy compute portvals, and port stats
    manual_trades = learner.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
    manual_portval = compute_portvals(symbol=symbol, orders_file=manual_trades, start_val=sv, commission=commission, impact=impact)
    norm_manual_portval = manual_portval / manual_portval[0]
    manual_cr, manual_adr, manual_sddr, manual_sr = port_stats(manual_portval)
    print("Manual strategy performance", manual_cr, manual_adr, manual_sddr, manual_sr)

    # Out sample: Benchmark compute portvals and port stats
    bench_trades = learner.benchMarkPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
    bench_portvals = compute_portvals(symbol=symbol, orders_file=bench_trades, start_val=sv, commission=commission, impact=impact)
    norm_bench_portvals = bench_portvals / bench_portvals[0]
    bench_cr, bench_adr, bench_sddr, bench_sr = port_stats(bench_portvals)
    print("Benchmark performance", bench_cr, bench_adr, bench_sddr, bench_sr)

    # Out sample: Plot manual strategy and benchmark
    plt.title("Figure 2: Normalized manual strategy vs Benchmark (Out-sample)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio values")
    plt.grid(b=True, linestyle='--')
    plt.plot(norm_manual_portval, label="Manual Strategy", color="red")
    plt.plot(norm_bench_portvals, label="Benchmark", color="purple")
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(manual_trades.index[manual_trades['Shares'] > 0].tolist(), ymin=ymin, ymax=ymax, label="LONG entry points", color="blue", linestyles='dashed', lw=0.5)
    plt.vlines(manual_trades.index[manual_trades['Shares'] < 0].tolist(), ymin=ymin, ymax=ymax, label="SHORT entry points", color="black", linestyles='dashed', lw=0.5)
    plt.legend()
    # plt.show()
    plt.savefig("images/Figure2.png", dpi=500)
    plt.clf()

if __name__ == "__main__":
    symbol = "JPM"
    in_start_date = dt.datetime(2008,1,1)
    in_end_date = dt.datetime(2009,12,31)
    out_start_date = dt.datetime(2010, 1, 1)
    out_end_date = dt.datetime(2011, 12, 31)
    commission = 9.95
    impact = 0.005
    sv = 100000
    plots_manual(symbol, in_start_date, in_end_date, out_start_date, out_end_date, commission, impact, sv)
