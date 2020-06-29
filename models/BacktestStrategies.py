from math import *
import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from utils import backtest
from strategies import sixtyforty, onlystocks, vanillariskparity, uniform, riskparity

if __name__ == '__main__':

    start = datetime.datetime(2017, 6, 26)
    end = datetime.datetime(2020, 6, 26)

    UGLD = bt.feeds.YahooFinanceData(dataname="UGLD", fromdate=start, todate=end)
    UTSL = bt.feeds.YahooFinanceData(dataname="UTSL", fromdate=start, todate=end)
    UPRO = bt.feeds.YahooFinanceData(dataname="TQQQ", fromdate=start, todate=end)
    TMF = bt.feeds.YahooFinanceData(dataname="TMF", fromdate=start, todate=end)
    TYD = bt.feeds.YahooFinanceData(dataname="TYD", fromdate=start, todate=end)
    
    dd, cagr, sharpe, maxdd, stddev = backtest([UGLD, UTSL, UPRO, TMF, TYD], onlystocks,
                                              plot = True,
                                              reb_days = 20,
                                              lookback_period = 60,
                                              initial_cash = 100000,
                                              monthly_cash = 10000,
                                              n_assets = 5,
                                              printlog = False
                                              )
    print('#'*50)
    print('Drawdown:%0.3f' %dd)
    print('CAGR:%0.3f' %cagr)
    print('Stddev (annualized from monthly returns):%0.3f' %(stddev*sqrt(12)*100))
    print('Sharpe Ratio (annualized from monthly excess return):%0.3f' %(sharpe*sqrt(12)))
    print('Max. Drawdown:%0.3f' %maxdd)
    print('#'*50)
