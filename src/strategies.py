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
import riskparityportfolio as rp
from backtrader.utils.py3 import range
from datetime import timedelta



class StFetcher(object):
    _STRATS = []

    @classmethod
    def register(cls, target):
        cls._STRATS.append(target)

    @classmethod
    def COUNT(cls):
        return range(len(cls._STRATS))

    def __new__(cls, *args, **kwargs):
        idx = kwargs.pop('idx')

        obj = cls._STRATS[idx](*args, **kwargs)
        return obj

"""
Custom observer to save the target weights 
"""


class targetweightsobserver(bt.observer.Observer):
    params = (('n_assets', 100),)  # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_' + str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        for asset in range(0, self.params.n_assets):
            self.lines[asset][0] = self._owner.weights[asset]

"""
Custom observer to save the effective weights 
"""

class effectiveweightsobserver(bt.observer.Observer):
    params = (('n_assets', 100),)  # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_' + str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        effectiveweights = self._owner.get_effectiveweights()
        for asset in range(0, self.params.n_assets):
            self.lines[asset][0] = effectiveweights[asset]

"""
Custom observer to get dates 
"""


class GetDate(bt.observer.Observer):
    lines = ('year', 'month', 'day',)

    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.year[0] = self._owner.datas[0].datetime.date(0).year
        self.lines.month[0] = self._owner.datas[0].datetime.date(0).month
        self.lines.day[0] = self._owner.datas[0].datetime.date(0).day




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
        ('shareclass', []),  # class of assets
        ('assetweights', []),  # weights of portfolio items (if provided)
        ('printlog', True),
        ('corrmethod', 'pearson'),  # 'spearman' # method for the calculation of the correlation matrix
    )


    def __init__(self):
        # To keep track of pending orders and buy price/commission
        self.order = None
        #self.cheating = self.cerebro.p.cheat_on_open
        self.buyprice = None
        self.buycomm = None

        self.weights = [0] * self.params.n_assets
        self.effectiveweights = [0] * self.params.n_assets

        self.startdate = None
        self.timeframe = self.get_timeframe()
        if self.timeframe == "Days":
            self.log("Strategy: you are using data with daily frequency", dt=None)
        elif self.timeframe == "Years":
            self.log("Strategy: you are using data with yearly frequency", dt=None)

        self.assets = []  # Save data to backtest into assets, other data (e.g. used in indicators) will not be saved here
        self.assetclose = []  # Keep a reference to the close price
        for asset in range(0, self.params.n_assets):
            self.assets.append(self.datas[asset])
            self.assetclose.append(self.datas[asset].close)

        self.indassets = []  # Save indicators here
        self.indassetsclose = []  # Keep a reference to the close price
        indicator_idxs = [i for i, e in enumerate(self.params.shareclass) if e == 'non-tradable']
        if indicator_idxs:
            for indicator_idx in indicator_idxs:
                self.indassets.append(self.datas[indicator_idx])
                self.indassetsclose.append(self.datas[indicator_idx].close)

        MinPeriodSetter(period=self.params.lookback_period_long)  # Set the minimum period

    def start(self):
        self.broker.set_fundmode(fundmode=True, fundstartval=100.00)  # Activate the fund mode, default has 100 shares
        self.broker.set_cash(self.params.initial_cash)  # Set initial cash of the account

        if self.timeframe == "Days":
            # Add a timer which will be called on the 20st trading day of the month, when salaries are paid
            self.add_timer(
                bt.timer.SESSION_START,
                monthdays=[20],  # called on the 20th day of the month
                monthcarry=True  # called on another day if 20th day is vacation/weekend)
            )
        elif self.timeframe == "Years":
            # Add a timer which will be called every year
            self.add_timer(bt.timer.SESSION_START)

    def get_timeframe(self):
        dates = [self.datas[0].datetime.datetime(i) for i in range(1, len(self.datas[0].datetime.array) + 1)]
        datediff = stats.mode(np.diff(dates))[0][0]
        if datediff > timedelta(days=250):
            return "Years"
        elif datediff < timedelta(days=2):
            return "Days"

    def notify_timer(self, timer, when, *args, **kwargs):
        # Add the monthly cash to the broker
        if self.startdate is not None and self.datas[0].datetime.datetime(0) >= self.startdate:
            if self.params.monthly_cash > 0:
                self.broker.add_cash(self.params.monthly_cash)  # Add monthly cash on the 20th day
                self.log('MONTHLY CASH ADDED: new cash amount is %.f' % (self.broker.get_cash() + self.params.monthly_cash))

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
                    'BUY EXECUTED, Price: %.2f, Size: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    'SELL EXECUTED, Price: %.2f, Size: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected.')

        # Write down: no pending order
        self.order = None

    def get_weights(self):
        return self.weights

    def get_effectiveweights(self):
        PortfolioValue = self.broker.get_value()
        for asset in range(0, self.params.n_assets):
            position = self.broker.getposition(data=self.assets[asset]).size
            if len(self.assetclose[asset].get(0)) == 0:
                ThisAssetval = 0
            else:
                ThisAssetval = self.assetclose[asset].get(0)[0] * position
            self.effectiveweights[asset] = ThisAssetval / PortfolioValue
        return self.effectiveweights


    def nextstart(self):
        self.startdate = self.datas[0].datetime.datetime(1) # the first date is the next one, since nextstart is called one bar before next

        #self.initial_buy()

        super(StandaloneStrat, self).nextstart()

    """
    def initial_buy(self):
    # Buy at open price
    for asset in range(0, self.params.n_assets):
        target_size = int(self.broker.get_cash() * self.weights[asset] / self.assetclose[asset].get(0)[0])
        if target_size > 0:
            self.buy(data=self.datas[asset], exectype=bt.Order.Limit,
                     price=self.assetclose[asset].get(0)[0], size=target_size, coo=True)
    """


    def rebalance(self):
        sold_value = 0
        # Sell at open price
        for asset in range(0, self.params.n_assets):
            position = self.broker.getposition(data=self.assets[asset]).size
            if position > 0:
                self.sell(data=self.datas[asset], exectype=bt.Order.Limit,
                          price=self.assetclose[asset].get(0)[0], size=position, coo=True)
                sold_value = sold_value + (self.assetclose[asset].get(0)[0] * position)

        cash_after_sell = sold_value + self.broker.get_cash()
        # Buy at open price
        for asset in range(0, self.params.n_assets):
            target_size = int(cash_after_sell * self.weights[asset] / self.assetclose[asset].get(0)[0])
            if target_size > 0:
                self.buy(data=self.datas[asset], exectype=bt.Order.Limit,
                         price=self.assetclose[asset].get(0)[0], size=target_size, coo=True)

"""
The child classes below are specific to one strategy.
"""

class customweights(StandaloneStrat):
    strategy_name = "Custom weights"

    def prenext(self):
        self.weights = self.params.assetweights

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

#@StFetcher.register
class sixtyforty(StandaloneStrat):
    strategy_name = "60-40 Portfolio"

    def prenext(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 0.6,
            "bond_lt": 0.2,
            "bond_it": 0.2
        }

        tradable_shareclass = [x for x in self.params.shareclass if x != 'non-tradable']

        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()


#@StFetcher.register
class onlystocks(StandaloneStrat):
    strategy_name = "Only Stocks Portfolio"

    def prenext(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 1,
            "bond_lt": 0,
            "bond_it": 0
        }

        tradable_shareclass = [x for x in self.params.shareclass if x != 'non-tradable']

        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                    self.broker.get_cash(),
                    self.broker.get_value(),
                    self.broker.get_fundshares(),
                    self.broker.get_fundvalue()))

            self.rebalance()


#@StFetcher.register
class vanillariskparity(StandaloneStrat):
    strategy_name = "Vanilla Risk Parity Portfolio"

    def prenext(self):
        assetclass_allocation = {
            "gold": 0.12,
            "commodity": 0.13,
            "equity": 0.2,
            "bond_lt": 0.15,
            "bond_it": 0.4
        }

        tradable_shareclass = [x for x in self.params.shareclass if x != 'non-tradable']

        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

#@StFetcher.register
class uniform(StandaloneStrat):
    strategy_name = "Uniform Portfolio"

    def prenext(self):
        assetclass_allocation = {
            "gold": 0.2,
            "commodity": 0.2,
            "equity": 0.2,
            "bond_lt": 0.2,
            "bond_it": 0.2
        }

        tradable_shareclass = [x for x in self.params.shareclass if x != 'non-tradable']

        assetclass_cnt = {}
        assetclass_flag = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count
            if count > 0:
                assetclass_flag[key] = 1
            else:
                assetclass_flag[key] = 0

        num_assetclasses = sum(assetclass_flag.values())
        assetclass_weight = {k: v / num_assetclasses for k, v in assetclass_flag.items()}

        a = list(map(assetclass_weight.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

#@StFetcher.register
class rotationstrat(StandaloneStrat):
    strategy_name = "Asset rotation strategy"

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            assetclass_allocation = {
                "gold": 0,
                "commodity": 0,
                "equity": 0,
                "bond_lt": 0,
                "bond_it": 0
            }

            strat = {
                1: "gold",
                2: "bond_lt",
                3: "equity"
            }

            which_max = self.indassetsclose.index(max(self.indassetsclose))

            winningAsset = strat.get(which_max)

            tradable_shareclass = [x for x in self.params.shareclass if x != 'non-tradable']

            for key in assetclass_allocation:
                if key == winningAsset:
                    assetclass_allocation[key] = 1

            assetclass_cnt = {}
            for key in assetclass_allocation:
                count = sum(map(lambda x: x == key, tradable_shareclass))
                assetclass_cnt[key] = count

            a = list(map(assetclass_allocation.get, tradable_shareclass))
            b = list(map(assetclass_cnt.get, tradable_shareclass))

            self.weights = [float(x) / y for x, y in zip(a, b)]

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()


# Risk parity portfolio. The implementation is based on:
# https: // thequantmba.wordpress.com / 2016 / 12 / 14 / risk - parityrisk - budgeting - portfolio - in -python /
# Here the risk parity is run only at portfolio level
#@StFetcher.register
class riskparity(StandaloneStrat):
    strategy_name = "Risk Parity"

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in self.assetclose]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = target_risk_contribution(target_risk, cov)

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

# Risk parity portfolio. The implementation is based on:
# https: // thequantmba.wordpress.com / 2016 / 12 / 14 / risk - parityrisk - budgeting - portfolio - in -python /
# Here the risk parity is run first at asset class level and then at portfolio level. To be used when more than an asset
# is present in each category
#@StFetcher.register
class riskparity_nested(StandaloneStrat):
    strategy_name = "Risk Parity (Nested)"

    def next_open(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 0,
            "bond_lt": 0,
            "bond_it": 0
        }

        if len(self) % self.params.reb_days == 0:

            shareclass_prices = []
            shareclass_cnt = []
            assetWeights = []
            # First run risk parity at asset class level
            for key in assetclass_allocation:
                # Get the assets whose asset class is "key"
                count = sum(map(lambda x: x == key, self.params.shareclass))
                shareclass_cnt.append(count)
                if count > 1:
                    thisAssetClass_target_risk = [1 / count] * count # Same risk for each assetclass = risk parity

                    # calculate the logreturns
                    thisAssetClass_idx = [i for i, e in enumerate(self.params.shareclass) if e == key]
                    thisAssetClassClose = [self.assetclose[i].get(size=self.params.lookback_period_long) for i in thisAssetClass_idx]

                    # Check if asset prices is equal to the lookback period for all assets exist
                    thisAssetClassClose_len = [len(i) for i in thisAssetClassClose]
                    if not all(elem == self.params.lookback_period_long for elem in thisAssetClassClose_len):
                        return

                    logrets = [np.diff(np.log(x)) for x in thisAssetClassClose]

                    if self.params.corrmethod == 'pearson':
                        corr = np.corrcoef(logrets)
                    elif self.params.corrmethod == 'spearman':
                        corr, p, = stats.spearmanr(logrets, axis=1)

                    stddev = np.array([np.std(x[len(x)-self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
                    stddev_matrix = np.diag(stddev)
                    cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

                    # Calculate the asset class weights
                    thisAssetClassWeights = target_risk_contribution(thisAssetClass_target_risk, cov)
                    assetWeights.append(thisAssetClassWeights)

                    # Calculate the synthetic asset class price
                    prod = 0
                    for i in range(0, count):
                        prod = np.add(prod, np.multiply(thisAssetClassWeights[i], thisAssetClassClose[i]))

                    shareclass_prices.append(prod)

                if count == 1:
                    thisAssetClass_idx = [i for i, e in enumerate(self.params.shareclass) if e == key]
                    thisAssetClassClose = [self.assetclose[i].get(size=self.params.lookback_period_long) for i in thisAssetClass_idx]
                    shareclass_prices.append(thisAssetClassClose[0])
                    assetWeights.append(np.asarray([1]))

            # Check if asset prices is equal to the lookback period for all assets exist
            shareclass_prices_len = [len(i) for i in shareclass_prices]
            if not all(elem == self.params.lookback_period_long for elem in shareclass_prices_len):
                return

            # Now re-run risk parity at portfolio level, using the synthetic assets
            target_risk = [1 / np.count_nonzero(shareclass_cnt)] * np.count_nonzero(shareclass_cnt)  # Same risk for each assetclass = risk parit
            logrets = [np.diff(np.log(x)) for x in shareclass_prices]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x)-self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            assetClass_weights = target_risk_contribution(target_risk, cov)

            # Now cascade down the weights at asset level
            keys = pd.DataFrame()
            keys["keys"]=list(assetclass_allocation.keys())
            keys["cnt"]=shareclass_cnt
            keys = keys[keys.cnt > 0]
            keys_lst = keys['keys'].tolist()

            weights = pd.DataFrame(columns=["sort", "weights"])
            for i in range(0, len(assetClass_weights)):
                thisAssetClass_idx = [k for k, e in enumerate(self.params.shareclass) if e == keys_lst[i]]
                for j in range(0, len(assetWeights[i])):
                    to_append=[thisAssetClass_idx[j], assetClass_weights[i] * assetWeights[i][j]]
                    a_series = pd.Series(to_append, index=weights.columns)
                    weights = weights.append(a_series, ignore_index=True)

            # and rearrange the weights
            weights_lst = weights.sort_values("sort")["weights"].to_list()

            self.weights = weights_lst

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                self.broker.get_cash(),
                self.broker.get_value(),
                self.broker.get_fundshares(),
                self.broker.get_fundvalue()))

            self.rebalance()

# Risk parity portfolio. The implementation is based on the Python library riskparity portfolio
#@StFetcher.register
class riskparity_pylib(StandaloneStrat):
    strategy_name = "Risk Parity (PythonLib)"

    def next_open(self):
        if len(self) % self.params.reb_days == 0:
            target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in self.assetclose]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = rp.RiskParityPortfolio(covariance=cov, budget=target_risk).weights

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

# Optimal tangent portfolio according to the Modern Portfolio theory by Markowitz. The implementation is based on:
# https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
class meanvarStrat(StandaloneStrat):
    strategy_name = "Tangent Portfolio"


    def next(self):
        print("Work in progress")

