import pandas as pd
import os
import backtrader as bt
import numpy as np

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

    # create broker, still need to tweak it
    broker_kwargs = dict(cash=10000,coc=True)
    cerebro.broker = bt.brokers.BackBroker(**broker_kwargs)

    
    # add the strategy
    cerebro.addstrategy(strategy, **kwargs)
    res = cerebro.run()
    
    if plot: # plot results if asked
        cerebro.plot(volume=False)

    
    return (res[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            res[0].analyzers.returns.get_analysis()['rnorm100'],
            res[0].analyzers.sharperatio.get_analysis()['sharperatio'],
            res[0].analyzers.timedrawdown.get_analysis()['maxdrawdown'],
            res[0].analyzers.periodstats.get_analysis()['stddev'])



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
