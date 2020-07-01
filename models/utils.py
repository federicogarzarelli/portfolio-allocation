import pandas as pd
import os
import backtrader as bt
import numpy as np
from observers import WeightsObserver, GetDate
from datetime import datetime as dt

def backtest(datas, strategy, plot=False, **kwargs):
    # initialize cerebro
    cerebro = bt.Cerebro()
    
    # add the data
    for data in datas:
        cerebro.adddata(data)

    # keep track of certain metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03, timeframe=bt.TimeFrame.Months, fund=True)
    cerebro.addanalyzer(bt.analyzers.Returns, fund=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, fund=True)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, timeframe=bt.TimeFrame.Months, fund=True)
    cerebro.addanalyzer(bt.analyzers.PeriodStats, timeframe=bt.TimeFrame.Months, fund=True)

    # Add custom observer to get the weights
    n_assets = kwargs.get('n_assets')
    cerebro.addobserver(WeightsObserver, n_assets=n_assets)
    cerebro.addobserver(GetDate)

    # add the strategy
    cerebro.addstrategy(strategy, **kwargs)

    res = cerebro.run()

    if plot: # plot results if asked
        figure = cerebro.plot(volume=False, iplot=False)[0][0]
        figure.savefig('Strategy %s.png' % strategy.strategy_name)

    metrics = (res[0].analyzers.returns.get_analysis()['rnorm100'],
               res[0].analyzers.periodstats.get_analysis()['stddev'],
               res[0].analyzers.sharperatio.get_analysis()['sharperatio'],
               res[0].analyzers.timedrawdown.get_analysis()['maxdrawdown'],
               res[0].analyzers.drawdown.get_analysis()['max']['drawdown']
               )

    # Asset weights
    size_weights = 60 # get weights for the last 60 days
    weight_df = pd.DataFrame()

    weight_df['Year'] = pd.Series(res[0].observers[3].year.get(size=size_weights))
    weight_df['Month'] = pd.Series(res[0].observers[3].month.get(size=size_weights))
    weight_df['Day'] = pd.Series(res[0].observers[3].day.get(size=size_weights))
    for i in range(0, n_assets):
        weight_df['asset_'+str(i)] = res[0].observers[2].lines[i].get(size=size_weights)

    """    
    weights = [res[0].observers[3].year.get(size=size_weights),
               res[0].observers[3].month.get(size=size_weights),
               res[0].observers[3].day.get(size=size_weights),
               [res[0].observers[2].lines[i].get(size=size_weights) for i in range(0, n_assets)]]
    """
    return metrics, weight_df


def import_process_hist(dataLabel, args):
    wd = os.path.dirname(os.getcwd())

    mapping_path_linux = {
        'GLD':wd+'/modified_data/clean_gld.csv',
        'SP500':wd+'/modified_data/clean_gspc.csv',
        'COM':wd+'/modified_data/clean_spgscitr.csv',
        'LTB':wd+'/modified_data/clean_tyx.csv',
        'ITB':wd+'/modified_data/clean_fvx.csv',
        'TIP':wd+'/modified_data/clean_dfii10.csv'
    }

    mapping_path_windows = {
        'GLD':wd+'\modified_data\clean_gld.csv',
        'SP500':wd+'\modified_data\clean_gspc.csv',
        'COM':wd+'\modified_data\clean_spgscitr.csv',
        'LTB':wd+'\modified_data\clean_tyx.csv',
        'ITB':wd+'\modified_data\clean_fvx.csv',
        'TIP':wd+'\modified_data\clean_dfii10.csv'
    }

    if args.system == 'linux':
        datapath = (mapping_path_linux[dataLabel])
    else:
        datapath = (mapping_path_windows[dataLabel])
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    
    return df

        
def add_leverage(price, leverage=1, expense_ratio=0.0):
    """
    Simulates a leverage ETF given its proxy, leverage, and expense ratio.

    Daily percent change is calculated by taking the daily log-return of
    the price, subtracting the daily expense ratio, then multiplying by the leverage.
    """
    initial_value = price.iloc[0]
    ret = (price - price.shift(1))/price.shift(1)
    ret = (ret - expense_ratio / 252) * leverage
    new_price = initial_value * (1+ret).cumprod()
    new_price[0] = initial_value
    return new_price
