import datetime as dt
from StrategyLearner import plots_strats, StrategyLearner
from ManualStrategy import plots_manual, ManualStrategy
from experiment1 import experiment_1_in_sample, experiment_1_out_sample
from experiment2 import experiment_2

def author(self):
    return "mwang709"

def study_group(self):
    return "mwang709"

if __name__ == "__main__":
    symbol = "JPM"
    in_start_date = dt.datetime(2008,1,1)
    in_end_date = dt.datetime(2009,12,31)
    out_start_date = dt.datetime(2010, 1, 1)
    out_end_date = dt.datetime(2011, 12, 31)
    commission = 9.95
    impact = 0.005
    impacts = [0.005, 0.01, 0.05]
    sv = 100000

    # Manual Strategy
    plots_manual(symbol, in_start_date, in_end_date, out_start_date, out_end_date, commission, impact, sv)

    # Strategy Learner
    plots_strats(symbol, in_start_date, in_end_date, out_start_date, out_end_date, commission, impact, sv)

    # Experiment 1
    learner_manual = ManualStrategy(verbose=False, impact=impact, commission=commission)
    learner_strategy = StrategyLearner(verbose=False, impact=impact, commission=commission)
    experiment_1_in_sample(learner_manual,learner_strategy, symbol, in_start_date, in_end_date, commission, impact, sv)
    experiment_1_out_sample(learner_manual,learner_strategy, symbol, out_start_date, out_end_date, commission, impact, sv)

    # Experiment 2
    experiment_2(symbol, in_start_date, in_end_date, commission, impacts, sv)