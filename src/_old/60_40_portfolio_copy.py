"""
Code taken and adapted from https://teddykoker.com/2019/04/backtesting-portfolios-of-leveraged-etfs-in-python-with-backtrader/
"""

import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)


def add_leverage2(proxy, leverage=1, expense_ratio=0.0):
    """
    Simulates a leverage ETF given its proxy, leverage, and expense ratio.

    Daily percent change is calculated by taking the daily log-return of
    the price, subtracting the daily expense ratio, then multiplying by the leverage.
    """
    initial_value = proxy.iloc[0]
    pct_change = proxy.pct_change(1)
    pct_change = (pct_change - expense_ratio / 252) * leverage
    new_price = initial_value * (1+pct_change).cumprod()
    new_price[0] = initial_value
    return new_price


class AssetAllocation(bt.Strategy):
    params = (
        ('sp500',0.35),
        ('tmf',0.50),
        ('gld', 0.15)
    )
    def __init__(self):
        self.sp500 = self.datas[0]
        self.tmf = self.datas[1]
        self.gld = self.datas[2]
        self.counter = 0
        
    def next(self):
        if  self.counter % 20 == 0:
            self.order_target_percent(self.sp500, target=self.params.sp500)
            self.order_target_percent(self.tmf, target=self.params.tmf)
            self.order_target_percent(self.gld, target=self.params.gld)
        self.counter += 1


def localbacktest(datas, strategy, plot=False, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(100000)
    for data in datas:
        cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addstrategy(strategy, **kwargs)
    results = cerebro.run()
    if plot:
        cerebro.plot()
    return (results[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            results[0].analyzers.returns.get_analysis()['rnorm100'],
            results[0].analyzers.sharperatio.get_analysis()['sharperatio']
    )


if __name__ == '__main__':

    
    
    start = datetime.datetime(1986,5,19)
    end   = datetime.datetime(2020,6,1)

    # load data
    sp500 = web.DataReader("VFINX", "yahoo", start, end)["Adj Close"]
    tlt = web.DataReader("VUSTX", "yahoo", start, end)["Adj Close"]
    gld = web.DataReader("VGPMX", "yahoo", start, end)["Adj Close"]
    
    
    sp500_sim = add_leverage2(sp500, leverage=3.0, expense_ratio=0.0092).to_frame("close")
    tmf_sim = add_leverage2(tlt, leverage=3.0, expense_ratio=0.0109).to_frame("close")
    gld_sim = add_leverage2(gld, leverage=3.0, expense_ratio=0.0135).to_frame("close")
    
    for column in ["open", "high", "low"]:
        sp500_sim[column] = sp500_sim["close"]
        tmf_sim[column] = tmf_sim["close"]
        gld_sim[column] = gld_sim["close"]

    sp500_sim["volume"] = 0
    tmf_sim["volume"] = 0
    gld_sim["volume"] = 0
    
    sp500_sim = bt.feeds.PandasData(dataname=sp500_sim)
    tmf_sim = bt.feeds.PandasData(dataname=tmf_sim)
    gld_sim = bt.feeds.PandasData(dataname=gld_sim)

    dd, cagr, sharpe = localbacktest([sp500_sim, tmf_sim, gld_sim], AssetAllocation, plot=True)
    print(f"Max Drawdown: {dd:.2f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}")
