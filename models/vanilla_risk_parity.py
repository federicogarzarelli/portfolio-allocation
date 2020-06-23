import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from utils import backtest




class RiskParity(bt.Strategy):
    """
    Vanilla Risk Parity Strategy. 
    
    We do consider an allocation that is constant through time
    and rebalanced every month (20 days, this is a parameter).
    
    Every month, we add extra cash to the account.
    """
    params = (
        ('reb_days', 20), # every month, we rebalance the portfolio
        ('initial_cash', 100000), # initial amount of cash to be invested
        ('monthly_cash', 10000), # amount of cash to buy invested every month
        ('alloc_ugld', 0.12),
        ('alloc_utsl', 0.13),
        ('alloc_upro', 0.20),
        ('alloc_tmf', 0.15),
        ('alloc_tyd', 0.40)
    )

    def __init__(self):
        self.counter = 0
        self.ugld = self.datas[0]
        self.utsl = self.datas[1]
        self.upro = self.datas[2]
        self.tmf = self.datas[3]
        self.tyd = self.datas[4]
        
    def start(self):
        # Activate the fund mode and set the default value at 100000
        self.broker.set_fundmode(fundmode=True, fundstartval=self.params.initial_cash)

        self.cash_start = self.broker.get_cash()
        self.val_start =  self.params.initial_cash

        # Add a timer which will be called on the 1st trading day of the month
        self.add_timer(
            bt.timer.SESSION_END,
            monthdays = [1], # called on the 1st day of the month
            monthcarry = True # called on another day if 1st day is vacation/weekend)
        )

    def notify_timer(self,timer, when, *args, **kwargs):
        # Add the monthly cash to the broker
        self.broker.add_cash(self.params.monthly_cash)
        
        
    def next(self):
        if self.counter % self.params.reb_days:
            self.order_target_percent(self.ugld, target=self.params.alloc_ugld)
            self.order_target_percent(self.utsl, target=self.params.alloc_utsl)
            self.order_target_percent(self.upro, target=self.params.alloc_upro)
            self.order_target_percent(self.tmf, target=self.params.alloc_tmf)
            self.order_target_percent(self.tyd, target=self.params.alloc_tyd)
        self.counter +=1


if __name__ == '__main__':

    

    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2020, 6, 1)

    UGLD = bt.feeds.YahooFinanceData(dataname="UGLD", fromdate=start, todate=end)
    UTSL = bt.feeds.YahooFinanceData(dataname="UTSL", fromdate=start, todate=end)
    UPRO = bt.feeds.YahooFinanceData(dataname="UPRO", fromdate=start, todate=end)
    TMF = bt.feeds.YahooFinanceData(dataname="TMF", fromdate=start, todate=end)
    TYD = bt.feeds.YahooFinanceData(dataname="TYD", fromdate=start, todate=end)
    
    dd, cagr, sharpe = backtest([UGLD, UTSL, UPRO, TMF, TYD], RiskParity,
                                plot = True, 
                                reb_days = 20, 
                                initial_cash = 100000, 
                                monthly_cash = 0, 
                                alloc_ugld = 0.12,
                                alloc_utsl = 0.13,
                                alloc_upro = 0.20,
                                alloc_tmf = 0.15,
                                alloc_tyd = 0.40)

    print('Drowndown:%0.3f' %dd)
    print('CAGR:%0.3f' %cagr)
    print('Sharpe:%0.3f' %sharpe)
