import backtrader as bt
import numpy as np


def backtest(datas, strategy, plot=False, **kwargs):
    # initialize cerebro
    cerebro = bt.Cerebro()

    # add the data
    for data in datas:
        cerebro.adddata(data)

    # keep track of certain metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    # create broker, still need to tweak it
    broker_kwargs = dict(coc=True)
    cerebro.broker = bt.brokers.BackBroker(**broker_kwargs)
    
    # add the strategy
    cerebro.addstrategy(strategy, **kwargs)
    res = cerebro.run()
    
    
    if plot: # plot results if asked
        cerebro.plot()


    return (res[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            res[0].analyzers.returns.get_analysis()['rnorm100'],
            res[0].analyzers.sharperatio.get_analysis()['sharperatio'])
