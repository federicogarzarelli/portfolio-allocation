# -*- coding: utf-8 -*-
import matplotlib
from math import *
import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from risk_budgeting import target_risk_contribution

# Create a subclass of bt.Strategy to define the indicators and logic.
class StandaloneStrat(bt.Strategy):
    # parameters which are configurable for the strategy
    params = (
        ('reb_days', 20),  # every month, we rebalance the portfolio
        ('lookback_period', 60),  # period to calculate the indicators used by the strategy (e.g. covariance)
        ('initial_cash', 100000),  # initial amount of cash to be invested
        ('monthly_cash', 10000),  # amount of cash to buy invested every month
        ('n_assets', 5),
        ('printlog', True),
    )

    def __init__(self):
        self.assets = []
        self.dataclose = []
        for asset in range(0, self.params.n_assets):
            self.assets.append(self.datas[asset])
            self.dataclose.append(self.datas[asset].close)  # Keep a reference to the close price

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def start(self):
        # Activate the fund mode and set the default value at 100000
        self.broker.set_fundmode(fundmode=True, fundstartval=self.params.initial_cash)

        self.cash_start = self.broker.get_cash()  # unused
        self.val_start = self.params.initial_cash # unused

        # Add a timer which will be called on the 1st trading day of the month
        self.add_timer(
            bt.timer.SESSION_END,
            monthdays=[1],  # called on the 1st day of the month
            monthcarry=True  # called on another day if 1st day is vacation/weekend)
        )

    def notify_timer(self, timer, when, *args, **kwargs):
        # Add the monthly cash to the broker
        self.broker.add_cash(self.params.monthly_cash)

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy txt is the statement and dt can be used to specify a specific datetime'''
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

"""
The child classes below are specific to one strategy. 
"""
class sixtyforty(StandaloneStrat):
    def next(self):
        alloc_target = [0, 0, 0.6, 0.20, 0.20]
        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=alloc_target[asset])

class onlystocks(StandaloneStrat):
    def next(self):
        alloc_target = [0, 0, 1, 0, 0]
        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=alloc_target[asset])

class vanillariskparity(StandaloneStrat):
    def next(self):
        alloc_target = [0.12, 0.13, 0.20, 0.15, 0.40]
        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=alloc_target[asset])

class uniform(StandaloneStrat):
    def next(self):
        alloc_target = [1 / self.params.n_assets] * self.params.n_assets
        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=alloc_target[asset])

# Optimal tangent portfolio according to the Modern Portfolio theory by Markowitz. The implementation is based on:
# https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
class meanvarStrat(StandaloneStrat):
    def next(self):
        print("Work in progress")

# Risk parity portfolio. The implementation is based on:
# https: // thequantmba.wordpress.com / 2016 / 12 / 14 / risk - parityrisk - budgeting - portfolio - in -python /
class riskparity(StandaloneStrat):
    def next(self):
        target_risk = [1 / self.params.n_assets] * self.params.n_assets

        if len(self) > self.params.lookback_period:
            # stack the closes together into a numpy array for ease of calculation
            closes = np.stack(([d.close.get(size = self.params.lookback_period).tolist() for d in self.datas]))
            rets = np.diff(np.log(closes))
            # covariance matrix. will be shoved somewhere else for records
            cov = np.cov(rets)
            alloc_target = target_risk_contribution(target_risk, cov)

            if len(self) % self.params.reb_days == 0:
                for asset in range(0, self.params.n_assets):
                    self.order_target_percent(self.assets[asset], target=alloc_target[asset])


"""
Simple test of the Strategy classes. 
"""

if __name__ == '__main__':
    start = datetime.datetime(2017, 6, 26)
    end = datetime.datetime(2020, 6, 26)

    UGLD = bt.feeds.YahooFinanceData(dataname="UGLD", fromdate=start, todate=end)
    UTSL = bt.feeds.YahooFinanceData(dataname="UTSL", fromdate=start, todate=end)
    UPRO = bt.feeds.YahooFinanceData(dataname="TQQQ", fromdate=start, todate=end)
    TMF = bt.feeds.YahooFinanceData(dataname="TMF", fromdate=start, todate=end)
    TYD = bt.feeds.YahooFinanceData(dataname="TYD", fromdate=start, todate=end)

    # strategies = ["cross", "simple1", "simple2", "BB"]
    # for tr_strategy in strategies:
    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

    cerebro.adddata(UGLD)  # Add the data feed
    cerebro.adddata(UTSL)  # Add the data feed
    cerebro.adddata(UPRO)  # Add the data feed
    cerebro.adddata(TMF)  # Add the data feed
    cerebro.adddata(TYD)  # Add the data feed

    cerebro.addstrategy(riskparity)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    result = cerebro.run()  # run it all
    figure = cerebro.plot(iplot=False)[0][0]
    figure.savefig('example.png')

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
#   ind = strategies.index(uniformStrat)
#   strategy_final_values[ind] = cerebro.broker.getvalue()

#   print("Final Values for Strategies")
# for tr_strategy in strategies:
#   ind = strategies.index(uniformStrat)
#  print("{} {}  ".format(uniformStrat, strategy_final_values[ind]))

