# -*- coding: utf-8 -*-
import matplotlib
from math import *
import pandas as pd
import pandas_datareader.data as web
import datetime
from scipy import stats, optimize, interpolate
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from risk_budgeting import target_risk_contribution
import os

"""
Custom observer to save the weights 
"""

"""
Custom indicator to set the minimum period. 
"""


class MinPeriodSetter(bt.Indicator):
    lines = ('dummyline',)

    params = (('period', 180),)

    def __init__(self):
        self.addminperiod(self.params.period)

# Create a subclass of bt.Strategy to define the indicators and logic.
class StandaloneStrat(bt.Strategy):
    # parameters which are configurable for the strategy
    params = (
        ('reb_days', 20),  # every month, we rebalance the portfolio
        ('lookback_period_short', 60),  # period to calculate the variance
        ('lookback_period_long', 180),  # period to calculate the correlation
        ('initial_cash', 100000),  # initial amount of cash to be invested
        ('monthly_cash', 10000),  # amount of cash to buy invested every month
        ('n_assets', 5),  # number of assets
        ('printlog', True),
        ('corrmethod', 'pearson'),  # 'spearman' # method for the calculation of the correlation matrix
    )

    def __init__(self):
        self.assets = []  # Save data to backtest into assets, other data (e.g. used in indicators) will not be saved here
        self.assetclose = []  # Keep a reference to the close price
        for asset in range(0, self.params.n_assets):
            self.assets.append(self.datas[asset])
            self.assetclose.append(self.datas[asset].close)

        MinPeriodSetter(period=self.params.lookback_period_long)  # Set the minimum period

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.weights = [0] * self.params.n_assets

    def start(self):
        self.broker.set_fundmode(fundmode=True)  # Activate the fund mode, default has 100 shares
        self.broker.set_cash(self.params.initial_cash)  # Set initial cash of the account

        # Add a timer which will be called on the 20st trading day of the month, when salaries are paid
        self.add_timer(
            bt.timer.SESSION_END,
            monthdays=[20],  # called on the 20th day of the month
            monthcarry=True  # called on another day if 20th day is vacation/weekend)
        )

    def notify_timer(self, timer, when, *args, **kwargs):
        # Add the monthly cash to the broker
        self.broker.add_cash(self.params.monthly_cash)  # Add monthly cash on the 20th day
        self.log('MONTHLY CASH ADDED')

    def log(self, txt, dt=None):
        ''' Logging function for this strategy txt is the statement and dt can be used to specify a specific datetime'''
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

    def get_weights(self):
        return self.weights


"""
The child classes below are specific to one strategy.
"""


class sixtyforty(StandaloneStrat):
    strategy_name = "60-40 Portfolio"

    def next(self):
        self.weights = [0, 0, 0.6, 0.20, 0.20]

        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=self.weights[asset])


class onlystocks(StandaloneStrat):
    strategy_name = "Only Stocks Portfolio"

    def next(self):
        self.weights = [0, 0, 1, 0, 0]

        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=self.weights[asset])


class vanillariskparity(StandaloneStrat):
    strategy_name = "Vanilla Risk Parity Portfolio"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def next(self):
        self.weights = [0.12, 0.13, 0.20, 0.15, 0.40]

        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=self.weights[asset])


class uniform(StandaloneStrat):
    strategy_name = "Uniform Portfolio"

    def next(self):
        self.weights = [1 / self.params.n_assets] * self.params.n_assets

        if len(self) % self.params.reb_days == 0:
            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=self.weights[asset])


                
# Risk parity portfolio. The implementation is based on:
# https: // thequantmba.wordpress.com / 2016 / 12 / 14 / risk - parityrisk - budgeting - portfolio - in -python /
class riskparity(StandaloneStrat):
    strategy_name = "Risk Parity"

    def next(self):
        target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity

        if len(self) % self.params.reb_days == 0:
            logrets = [np.diff(np.log(x.get(size=self.params.lookback_period_long))) for x in self.assetclose]
            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = target_risk_contribution(target_risk, cov)

            for asset in range(0, self.params.n_assets):
                self.order_target_percent(self.assets[asset], target=self.weights[asset])

            self.log("Shares %.2f, Current cash %.2f, Fund value %.2f" % (self.broker.get_fundshares(),
                                                                          self.broker.get_cash(),
                                                                          self.broker.get_fundvalue()))


# Optimal tangent portfolio according to the Modern Portfolio theory by Markowitz. The implementation is based on:
# https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
class meanvarStrat(StandaloneStrat):
    strategy_name = "Tangent Portfolio"

    def next(self):
        print("Work in progress")


