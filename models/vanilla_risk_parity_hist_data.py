# -*- coding: utf-8 -*-

from math import *
import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import numpy as np
import matplotlib.pyplot as plt
from utils import backtest
import os
import argparse

def import_process_hist(dataLabel, args):
    wd = os.path.dirname(os.getcwd())

    if dataLabel == 'GLD':
        if args.system == 'linux':
            datapath = (wd + '/data/Gold.csv')
        else:
            datapath = (wd + '\data\Gold.csv')
        df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True,
                         date_parser=lambda x: pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
        df = df.rename(columns={"Gold USD": "close"}, index={'Date': 'date'})
        df['close'] = df['close'].str.replace(',', '').astype(float)

    elif dataLabel == 'SP500':
        if args.system == 'linux':
            datapath = (wd + '/data/^GSPC.csv')
        else:
            datapath = (wd + '\data\^GSPC.csv')
        df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
        df = df.rename(columns={"Adj Close": "close"}, index={'Date': 'date'})

    elif dataLabel == 'COM':
        if args.system == 'linux':
            datapath = (wd + '/data/SPGSCITR_IND.csv')
        else:
            datapath = (wd + '\data\SPGSCITR_IND.csv')
        df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
        df = df.rename(columns={"Close": "close"}, index={'Date': 'date'})
        df['close'] = df['close'].str.replace(',', '').astype(float)
        # Eliminate the flash crashes between Apr 08 - Jul 08 from the data
        df['log_ret'] = (np.log(df.close) - np.log(df.close.shift(1)))
        df = df.dropna()
        df = df[(abs(df['log_ret']) < 1)]  # filter out the flash crash!
        df['close'] = df['close'].iloc[0] * np.exp(np.cumsum(df['log_ret']))

    elif dataLabel == 'LTB':
        if args.system == 'linux':
            datapath = (wd + '/data/^TYX.csv')
        else:
            datapath = (wd + '\data\^TYX.csv')
        df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True,
                         date_parser=lambda x: pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
        df = df.rename(columns={"Adj Close": "yield"}, index={'Date': 'date'})
        df = df[df['yield'] != 'null']
        df['close'] = 100 / np.power(1 + df['yield'] / 100, 30)
        df = df.dropna()

    elif dataLabel == 'ITB':
        if args.system == 'linux':
            datapath = (wd + '/data/^FVX.csv')
        else:
            datapath = (wd + '\data\^FVX.csv')
        df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
        df = df.rename(columns={"Adj Close": "yield"}, index={'Date': 'date'})
        df = df[df['yield'] != 'null']
        df['close'] = 100 / np.power(1 + df['yield'] / 100, 5)
        df = df.dropna()

    elif dataLabel == 'TIP':
        if args.system == 'linux':
            datapath = (wd + '/data/DFII10.csv')
        else:
            datapath = (wd + '\data\DFII10.csv')
        df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
        df = df.rename(columns={"DFII10": "yield"}, index={'DATE': 'date'})
        df = df[df['yield'] != '.']
        df['yield'] = df['yield'].astype(float)
        df['close'] = 100 / np.power(1 + df['yield'] / 100, 10)
        df = df.dropna()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0

    df = df[["open", "high", "low", "close", "volume"]]
    return (df)


class RiskParity(bt.Strategy):
    """
    Vanilla Risk Parity Strategy.

    We do consider an allocation that is constant through time
    and rebalanced every month (20 days, this is a parameter).

    Every month, we add extra cash to the account.
    """
    params = (
        ('reb_days', 20),  # every month, we rebalance the portfolio
        ('initial_cash', 100000),  # initial amount of cash to be invested
        ('monthly_cash', 10000),  # amount of cash to buy invested every month
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
        self.val_start = self.params.initial_cash

        # Add a timer which will be called on the 1st trading day of the month
        self.add_timer(
            bt.timer.SESSION_END,
            monthdays=[1],  # called on the 1st day of the month
            monthcarry=True  # called on another day if 1st day is vacation/weekend)
        )

    def notify_timer(self, timer, when, *args, **kwargs):
        # Add the monthly cash to the broker
        self.broker.add_cash(self.params.monthly_cash)

    def next(self):
        if self.counter % self.params.reb_days:
            self.order_target_percent(self.ugld, target=self.params.alloc_ugld)
            self.order_target_percent(self.utsl, target=self.params.alloc_utsl)
            self.order_target_percent(self.upro, target=self.params.alloc_upro)
            self.order_target_percent(self.tmf, target=self.params.alloc_tmf)
            self.order_target_percent(self.tyd, target=self.params.alloc_tyd)
        self.counter += 1

def add_leverage(price, leverage=1, expense_ratio=0.0):
    """
    Simulates a leverage ETF given its proxy, leverage, and expense ratio.

    Daily percent change is calculated by taking the daily log-return of
    the price, subtracting the daily expense ratio, then multiplying by the leverage.
    """
    initial_value = price.iloc[0]
    log_ret = np.log(price) - np.log(price.shift(1))
    log_ret = (log_ret - expense_ratio / 252) * leverage
    new_price = initial_value * np.exp(np.cumsum(log_ret))
    new_price[0] = initial_value
    return new_price


if __name__ == '__main__':

    wd = os.path.dirname(os.getcwd())
    start = datetime.datetime(1979, 1, 1)
    end = datetime.datetime(2020, 6, 1)

    # parse several options to be run
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--system', type=str, help='operating system, to deal with different paths',
                        default='microsoft')
    args = parser.parse_args()

    data = []

    assetLabels = ['GLD', 'COM', 'SP500', 'LTB', 'ITB']

    for assetLabel in assetLabels:
        df = import_process_hist(assetLabel, args)
        for column in ["open", "high", "low", "close"]:
            df[column] = add_leverage(df[column], leverage = 1, expense_ratio=0.0)
        data.append(bt.feeds.PandasData(dataname=df, fromdate=start, todate=end, timeframe=bt.TimeFrame.Days))

    dd, cagr, sharpe, maxdd, stddev = backtest(data, RiskParity,
                                               plot=True,
                                               reb_days=20,
                                               initial_cash=100000,
                                               monthly_cash=1000,
                                               alloc_ugld=0.12,
                                               alloc_utsl=0.13,
                                               alloc_upro=0.20,
                                               alloc_tmf=0.15,
                                               alloc_tyd=0.40)

    print('#' * 50)
    print('Drawdown:%0.3f' % dd)
    print('CAGR:%0.3f' % cagr)
    print('Stddev (annualized from monthly returns):%0.3f' % (stddev * sqrt(12) * 100))
    print('Sharpe Ratio (annualized from monthly excess return):%0.3f' % (sharpe * sqrt(12)))
    print('Max. Drawdown:%0.3f' % maxdd)
    print('#' * 50)