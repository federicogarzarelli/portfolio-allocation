from math import *
import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from utils import backtest
from strategies import sixtyforty, onlystocks, vanillariskparity, uniform, riskparity
#from observers import WeightsObserver

if __name__ == '__main__':

    start = datetime.datetime(2017, 6, 26)
    end = datetime.datetime(2020, 6, 26)

    # Lists for results
    CAGR = []
    Stddev = []
    SR = []
    DD = []
    MaxDD = []

    assetLabels = ['UGLD', 'UTSL', 'TQQQ', 'TMF', 'TYD']

    data = []
    for assetLabel in assetLabels:
        data.append(bt.feeds.YahooFinanceData(dataname=assetLabel, fromdate=start, todate=end))

    strategies = [sixtyforty, onlystocks, vanillariskparity, uniform, riskparity]
    for tr_strategy in strategies:
        metrics, weights = backtest(data, tr_strategy,
                                   plot=True,
                                   reb_days=20,  # every month rebalance the portfolio
                                   lookback_period_short=60,  # period to calculate the variance
                                   lookback_period_long=180,  # period to calculate the correlation
                                   initial_cash=100000,  # initial amount of cash to be invested
                                   monthly_cash=10000,  # amount of cash to buy invested every month
                                   n_assets=5,  # number of assets
                                   printlog=False,
                                   corrmethod='pearson' # spearman # method for the calculation of the correlation matrix
                                   )

        print('#' * 80)
        print('### Strategy: %s' % tr_strategy.strategy_name)
        print('#' * 80)
        print('CAGR:%0.3f' % metrics[0])
        print('Stddev (annualized from monthly returns):%0.3f' % (metrics[1] * sqrt(12) * 100))
        print('Sharpe Ratio (annualized from monthly excess return):%0.3f' % (metrics[2] * sqrt(12)))
        print('Drawdown:%0.3f' % metrics[3])
        print('Max. Drawdown:%0.3f' % metrics[4])
        print('#' * 80)

        # Write results
        CAGR.append(metrics[0])
        Stddev.append((metrics[1] * sqrt(12) * 100))
        SR.append((metrics[2] * sqrt(12)))
        DD.append(metrics[3])
        MaxDD.append(metrics[4])

        weights.to_csv(r'Asset weights '+ tr_strategy.strategy_name +'.csv')

    # Dataframe for results
    metrics_df = pd.DataFrame(columns=['Strategy', 'StartDate', 'EndDate', 'CAGR', 'Stddev', 'SharpeRatio', 'Drawdown', 'MaxDrawdown'])
    metrics_df['Strategy'] = [strat.strategy_name for strat in strategies]
    metrics_df['StartDate'] = [start.isoformat() for strat in strategies]
    metrics_df['EndDate'] = [end.isoformat() for strat in strategies]
    metrics_df['CAGR'] = CAGR
    metrics_df['Stddev'] = Stddev
    metrics_df['SharpeRatio'] = SR
    metrics_df['Drawdown'] = DD
    metrics_df['MaxDrawdown'] = MaxDD
    metrics_df.to_csv(r'Backtest Metrics.csv', index=False, header=True)






