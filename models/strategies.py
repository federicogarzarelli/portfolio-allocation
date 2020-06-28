# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime
import backtrader as bt

# Create a subclass of Strategy to define the indicators and logic
class Strats(bt.Strategy):
    # parameters which are configurable for the strategy
    params = (
        ('reb_days', 20),  # every month, we rebalance the portfolio
        ('initial_cash', 100000),  # initial amount of cash to be invested
        ('monthly_cash', 10000),  # amount of cash to buy invested every month
        ('n_assets', 1),
    )
    params['tr_strategy'] = None

    def log(self, txt, dt = None, doprint = False):
        ''' Logging function fot this strategy txt is the statement and dt can be used to specify a specific datetime'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.tr_strategy = self.params.tr_strategy
        self.counter = 0
        self.assets = []
        self.dataclose = []
        for asset in range(1, self.params.n_assets):
            self.assets[asset] = self.datas[asset]
            self.dataclose.append(self.datas[asset].close)  # Keep a reference to the close price

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def next(self, strategy_type=""):
        tr_str = self.tr_strategy
        print(self.tr_strategy)

        # Log the closing prices of the series
        self.log("Close, {0:8.2f} ".format(self.dataclose[0]))

        if tr_str == "only_stocks":
            alloc_vanilla_risk_parity = [0, 0, 1, 0, 0]
            if self.counter % self.params.reb_days:
                for asset in range(1, self.params.n_assets):
                    self.order_target_percent(self.assets[asset], target=alloc_vanilla_risk_parity[asset])
            self.counter += 1

        if tr_str == "sixty_forty":
            alloc_vanilla_risk_parity = [0, 0, 0.6, 0.20, 0.20]
            if self.counter % self.params.reb_days:
                for asset in range(1, self.params.n_assets):
                    self.order_target_percent(self.assets[asset], target=alloc_vanilla_risk_parity[asset])
            self.counter += 1

        if tr_str == "uniform":
            alloc_vanilla_risk_parity = [1/self.params.n_assets] * self.params.n_assets
            if self.counter % self.params.reb_days:
                for asset in range(1, self.params.n_assets):
                    self.order_target_percent(self.assets[asset], target=alloc_vanilla_risk_parity[asset])
            self.counter += 1

        if tr_str == "mean_var":
        # https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/

        if tr_str == "vanilla_risk_parity":
            alloc_vanilla_risk_parity = [0.12, 0.13, 0.20, 0.15, 0.40]
            if self.counter % self.params.reb_days:
                for asset in range(1, self.params.n_assets):
                    self.order_target_percent(self.assets[asset], target=alloc_vanilla_risk_parity[asset])
            self.counter += 1

        if tr_str == "risk_parity":

        print('Current Portfolio Value: %.2f' % cerebro.broker.getvalue())
"""
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS {0:8.2f}, NET {1:8.2f}'.format(
            trade.pnl, trade.pnlcomm))


strategy_final_values = [0, 0, 0, 0]
strategies = ["cross", "simple1", "simple2", "BB"]

for tr_strategy in strategies:
    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

    data = bt.feeds.GenericCSVData(
        dataname='GE.csv',

        fromdate=datetime(2019, 1, 1),
        todate=datetime(2019, 9, 13),

        nullvalue=0.0,

        dtformat=('%Y-%m-%d'),

        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        adjclose=5,
        volume=6,
        openinterest=-1

    )

    print("data")
    print(data)
    cerebro.adddata(data)  # Add the data feed

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.addstrategy(Strats, tr_strategy=tr_strategy)  # Add the trading strategy
    result = cerebro.run()  # run it all
    figure = cerebro.plot(iplot=False)[0][0]
    figure.savefig('example.png')

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    ind = strategies.index(tr_strategy)
    strategy_final_values[ind] = cerebro.broker.getvalue()

print("Final Vaues for Strategies")
for tr_strategy in strategies:
    ind = strategies.index(tr_strategy)
    print("{} {}  ".format(tr_strategy, strategy_final_values[ind]))

"""