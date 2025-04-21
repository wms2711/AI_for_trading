import datetime as dt
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner

def experiment_1_in_sample(learner_manual,learner_strategy, symbol, in_start_date, in_end_date, commission, impact, sv):
    # In sample: Manual strategy compute portvals, and port stats
    manual_trades = learner_manual.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    manual_portval = compute_portvals(symbol=symbol, orders_file=manual_trades, start_val=sv, commission=commission, impact=impact)
    norm_manual_portval = manual_portval / manual_portval[0]
    manual_cr, manual_adr, manual_sddr, manual_sr = port_stats(manual_portval)
    print("Manual strategy performance", manual_cr, manual_adr, manual_sddr, manual_sr)

    # In sample: Benchmark compute portvals and port stats
    bench_trades = learner_manual.benchMarkPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    bench_portvals = compute_portvals(symbol=symbol, orders_file=bench_trades, start_val=sv, commission=commission, impact=impact)
    norm_bench_portvals = bench_portvals / bench_portvals[0]
    bench_cr, bench_adr, bench_sddr, bench_sr = port_stats(bench_portvals)
    print("Benchmark performance", bench_cr, bench_adr, bench_sddr, bench_sr)

    # In sample: Strategy Learner compute portvals and port stats
    learner_strategy.add_evidence(symbol=symbol, sd=in_start_date, ed=in_end_date)
    strategy_trades = learner_strategy.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    strategy_portval = compute_portvals(symbol=symbol, orders_file=strategy_trades, start_val=sv, commission=commission, impact=impact)
    norm_strategy_portval = strategy_portval / strategy_portval[0]
    strategy_cr, strategy_adr, strategy_sddr, strategy_sr = port_stats(strategy_portval)
    print("Strategy Learner performance", strategy_cr, strategy_adr, strategy_sddr, strategy_sr)

    # In sample: Plot
    plt.title("Figure 5: Normalized portfolio (In-sample)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio values")
    plt.grid(b=True, linestyle='--')
    plt.plot(norm_manual_portval, label="Manual Strategy Normalized", color="red")
    plt.plot(norm_bench_portvals, label="Benchmark Normalized", color="purple")
    plt.plot(norm_strategy_portval, label="Strategy Learner Normalized")
    plt.legend()
    # plt.show()
    plt.savefig("images/Figure5.png", dpi=500)
    plt.clf()

def experiment_1_out_sample(learner_manual,learner_strategy, symbol, out_start_date, out_end_date, commission, impact, sv):
    # Out sample: Manual strategy compute portvals, and port stats
    manual_trades = learner_manual.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
    manual_portval = compute_portvals(symbol=symbol, orders_file=manual_trades, start_val=sv, commission=commission, impact=impact)
    norm_manual_portval = manual_portval / manual_portval[0]
    manual_cr, manual_adr, manual_sddr, manual_sr = port_stats(manual_portval)
    print("Manual strategy performance", manual_cr, manual_adr, manual_sddr, manual_sr)

    # Out sample: Benchmark compute portvals and port stats
    bench_trades = learner_manual.benchMarkPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
    bench_portvals = compute_portvals(symbol=symbol, orders_file=bench_trades, start_val=sv, commission=commission, impact=impact)
    norm_bench_portvals = bench_portvals / bench_portvals[0]
    bench_cr, bench_adr, bench_sddr, bench_sr = port_stats(bench_portvals)
    print("Benchmark performance", bench_cr, bench_adr, bench_sddr, bench_sr)

    # Out sample: Strategy Learner compute portvals and port stats
    learner_strategy.add_evidence(symbol=symbol, sd=out_start_date, ed=out_end_date)
    strategy_trades = learner_strategy.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
    strategy_portval = compute_portvals(symbol=symbol, orders_file=strategy_trades, start_val=sv, commission=commission, impact=impact)
    norm_strategy_portval = strategy_portval / strategy_portval[0]
    strategy_cr, strategy_adr, strategy_sddr, strategy_sr = port_stats(strategy_portval)
    print("Strategy Learner performance", strategy_cr, strategy_adr, strategy_sddr, strategy_sr)

    # Out sample: Plot
    plt.title("Figure 6: Normalized portfolio (Out-sample)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio values")
    plt.grid(b=True, linestyle='--')
    plt.plot(norm_manual_portval, label="Manual Strategy Normalized", color="red")
    plt.plot(norm_bench_portvals, label="Benchmark Normalized", color="purple")
    plt.plot(norm_strategy_portval, label="Strategy Learner Normalized")
    plt.legend()
    # plt.show()
    plt.savefig("images/Figure6.png", dpi=500)
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
    learner_manual = ManualStrategy(verbose=False, impact=impact, commission=commission)
    learner_strategy = StrategyLearner(verbose=False, impact=impact, commission=commission)
    # experiment_1_in_sample(learner_manual,learner_strategy, symbol, in_start_date, in_end_date, commission, impact, sv)
    experiment_1_out_sample(learner_manual,learner_strategy, symbol, out_start_date, out_end_date, commission, impact, sv)
