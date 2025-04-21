import datetime as dt
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner

def experiment_2(symbol, in_start_date, in_end_date, commission, impacts, sv):
    for index, impact in enumerate(impacts):
        for j in range(1, 4):
            learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
            learner.add_evidence(symbol=symbol, sd=in_start_date, ed=in_end_date)
            strategy_trades = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
            strategy_portval = compute_portvals(symbol=symbol, orders_file=strategy_trades, start_val=sv, commission=commission, impact=impact)
            norm_strategy_portval = strategy_portval / strategy_portval[0]
            strategy_cr, strategy_adr, strategy_sddr, strategy_sr = port_stats(strategy_portval)
            print(f"At impact {impact}, Strategy Learner performance, {strategy_cr}, {strategy_adr}, {strategy_sddr}, {strategy_sr}")
            plt.plot(norm_strategy_portval, label=f'Impact={impact}, replication={j}', color=f'C{index}')

    # Show chart of portfolio values
    plt.title('Figure 7: Portfolio Value for Different Impact Values')
    plt.grid(b=True, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend(fontsize=5)
    # plt.show()
    plt.savefig("images/Figure7.png", dpi=500)
    plt.clf()

if __name__ == "__main__":
    symbol = "JPM"
    in_start_date = dt.datetime(2008,1,1)
    in_end_date = dt.datetime(2009,12,31)
    commission = 9.95
    impacts = [0.005, 0.01, 0.05]
    sv = 100000
    experiment_2(symbol, in_start_date, in_end_date, commission, impacts, sv)
