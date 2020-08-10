import pandas as pd
import os
import backtrader as bt
import numpy as np
import datetime
from strategies import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web
import math
import numpy.random as nrand
from GLOBAL_VARS import *


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

    # Add custom observer to get the weights
    n_assets = kwargs.get('n_assets')
    cerebro.addobserver(WeightsObserver, n_assets=n_assets)
    cerebro.addobserver(GetDate)

    # add the strategy
    cerebro.addstrategy(strategy, **kwargs)

    res = cerebro.run()

    if plot:  # plot results if asked
        figure = cerebro.plot(volume=False, iplot=False)[0][0]
        figure.savefig('Strategy %s.png' % strategy.strategy_name)

    metrics = (res[0].analyzers.returns.get_analysis()['rnorm100'],
               res[0].analyzers.periodstats.get_analysis()['stddev'],
               res[0].analyzers.sharperatio.get_analysis()['sharperatio'],
               res[0].analyzers.timedrawdown.get_analysis()['maxdrawdown'],
               res[0].analyzers.drawdown.get_analysis()['max']['drawdown']
               )

    # Asset weights
    size_weights = 60  # get weights for the last 60 days
    weight_df = pd.DataFrame()

    weight_df['Year'] = pd.Series(res[0].observers[3].year.get(size=size_weights))
    weight_df['Month'] = pd.Series(res[0].observers[3].month.get(size=size_weights))
    weight_df['Day'] = pd.Series(res[0].observers[3].day.get(size=size_weights))
    for i in range(0, n_assets):
        weight_df['asset_' + str(i)] = res[0].observers[2].lines[i].get(size=size_weights)

    """    
    weights = [res[0].observers[3].year.get(size=size_weights),
               res[0].observers[3].month.get(size=size_weights),
               res[0].observers[3].day.get(size=size_weights),
               [res[0].observers[2].lines[i].get(size=size_weights) for i in range(0, n_assets)]]
    """
    return metrics, weight_df


def delete_in_dir(mydir, *args, **kwargs):
    """
    Deletes all files in a directory
    """

    file_extension = kwargs.get('file_extension', None)
    if file_extension is None:
        filelist = [f for f in os.listdir(mydir)]
    else:
        filelist = [f for f in os.listdir(mydir) if f.endswith(file_extension)]

    for f in filelist:
        os.remove(os.path.join(mydir, f))


def print_section_divider(strategy_name):
    """
    Prints a section divider with the strategy name
    """
    print("##############################################################################")
    print("###")
    print("### Backtest strategy: " + strategy_name)
    print("###")
    print("##############################################################################")


def print_header(args):
    print('##############################################################################')
    print('##############################################################################')
    print('### Backtest starting')
    print('###  Parameters:')
    print('###    --historic' + ' ' + str(vars(args)['historic']))
    print('###    --shares' + ' ' + str(vars(args)['shares']))
    print('###    --shareclass' + ' ' + str(vars(args)['shareclass']))
    print('###    --weights' + ' ' + str(vars(args)['weights']))
    print('###    --indicators' + ' ' + str(vars(args)['indicators']))
    print('###    --initial_cash' + ' ' + str(vars(args)['initial_cash']))
    print('###    --monthly_cash' + ' ' + str(vars(args)['monthly_cash']))
    print('###    --create_report' + ' ' + str(vars(args)['create_report']))
    print('###    --report_name' + ' ' + str(vars(args)['report_name']))
    print('###    --strategy' + ' ' + str(vars(args)['strategy']))
    print('###    --startdate' + ' ' + str(vars(args)['startdate']))
    print('###    --enddate' + ' ' + str(vars(args)['enddate']))
    print('###    --system' + ' ' + str(vars(args)['system']))
    print('###    --leverage' + ' ' + str(vars(args)['leverage']))
    print('##############################################################################')


def reportbacktest(datas, strategy, initial_cash, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(initial_cash)

    for data in datas:
        cerebro.adddata(data)

    cerebro.addstrategy(strategy, **kwargs)

    cerebro.run()
    cerebro.plot(volume=False)
    cerebro.report('reports')


def import_process_hist(dataLabel, args):
    wd = os.path.dirname(os.getcwd())

    mapping_path_linux = {
        'GLD': wd + '/modified_data/clean_gld.csv',
        'SP500': wd + '/modified_data/clean_gspc.csv',
        'COM': wd + '/modified_data/clean_spgscitr.csv',
        'LTB': wd + '/modified_data/clean_tyx.csv',
        'ITB': wd + '/modified_data/clean_fvx.csv',
        'TIP': wd + '/modified_data/clean_dfii10.csv',
        'GLD_LNG': wd + '/modified_data/clean_gld_yearly.csv',
        'OIL_LNG': wd + '/modified_data/clean_oil_yearly.csv',
        'EQ_LNG' : wd + '/modified_data/clean_equity_yearly.csv',
        'RE_LNG': wd + '/modified_data/clean_housing_yearly.csv',
        'LTB_LNG': wd + '/modified_data/clean_bond_yearly.csv',
        'ITB_LNG': wd + '/modified_data/clean_bill_yearly.csv'
    }

    mapping_path_windows = {
        'GLD': wd + '\modified_data\clean_gld.csv',
        'SP500': wd + '\modified_data\clean_gspc.csv',
        'COM': wd + '\modified_data\clean_spgscitr.csv',
        'LTB': wd + '\modified_data\clean_tyx.csv',
        'ITB': wd + '\modified_data\clean_fvx.csv',
        'TIP': wd + '\modified_data\clean_dfii10.csv',
        'GLD_LNG': wd + '\modified_data\clean_gld_yearly.csv',
        'OIL_LNG': wd + '\modified_data\clean_oil_yearly.csv',
        'EQ_LNG': wd + '\modified_data\clean_equity_yearly.csv',
        'RE_LNG': wd + '\modified_data\clean_housing_yearly.csv',
        'LTB_LNG': wd + '\modified_data\clean_bond_yearly.csv',
        'ITB_LNG': wd + '\modified_data\clean_bill_yearly.csv'
    }

    if args.system == 'linux':
        datapath = (mapping_path_linux[dataLabel])
    else:
        datapath = (mapping_path_windows[dataLabel])
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)

    return df


def add_leverage(proxy, leverage=1, expense_ratio=0.0, timeframe=bt.TimeFrame.Days):
    """
    Simulates a leverage ETF given its proxy, leverage, and expense ratio.

    Daily percent change is calculated by taking the daily log-return of
    the price, subtracting the daily expense ratio, then multiplying by the leverage.
    """
    initial_value = proxy.iloc[0]
    pct_change = proxy.pct_change(1)
    if timeframe == bt.TimeFrame.Days:
        pct_change = (pct_change - expense_ratio / DAYS_IN_YEAR) * leverage
    elif timeframe == bt.TimeFrame.Years:
        pct_change = ((1 + pct_change) ** (1 / DAYS_IN_YEAR)) - 1 # Transform into daily returns
        pct_change = (pct_change - expense_ratio / DAYS_IN_YEAR) * leverage # Apply leverage
        pct_change = ((1 + pct_change) ** DAYS_IN_YEAR) - 1 # Re-transform into yearly returns
    new_price = initial_value * (1 + pct_change).cumprod()
    new_price.iloc[0] = initial_value
    return new_price


class WeightsObserver(bt.observer.Observer):
    params = (('n_assets', 100),)  # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_' + str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        for asset in range(0, self.params.n_assets):
            self.lines[asset][0] = self._owner.weights[asset]


class GetDate(bt.observer.Observer):
    lines = ('year', 'month', 'day',)

    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.year[0] = self._owner.datas[0].datetime.date(0).year
        self.lines.month[0] = self._owner.datas[0].datetime.date(0).month
        self.lines.day[0] = self._owner.datas[0].datetime.date(0).day


def calculate_portfolio_var(w, cov):
    # function that calculates portfolio risk
    return (w.T @ cov @ w)


def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    # Marginal Risk Contribution
    MRC = cov @ w.T
    # Risk Contribution
    RC = np.multiply(MRC, w.T) / calculate_portfolio_var(w, cov)
    return RC


def target_risk_contribution(target_risk, cov):
    """
    Returns the weights of the portfolio such that the contributions to portfolio risk are as close as possiblem
    to the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    # construct the constants
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda w: np.sum(w) - 1
                        }

    def msd_risk(w, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions between weights and target_risk
        """
        w_contribs = risk_contribution(w, cov)
        return ((w_contribs - target_risk) ** 2).sum()

    w = minimize(msd_risk, init_guess,
                 args=(target_risk, cov), method='SLSQP',
                 options={'disp': False},
                 constraints=weights_sum_to_1,
                 bounds=bounds)
    return w.x


def covariances(shares=['GLD', 'TLT', 'SPY'], start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020, 6, 1)):
    '''
    function that provides the covariance matrix of a certain number of shares
    
    :param shares: (list) shares that we would like to compute
    
    :return: covariance matrix
    '''
    prices = pd.DataFrame([web.DataReader(t,
                                          'yahoo',
                                          start,
                                          end).loc[:, 'Adj Close']
                           for t in shares],
                          index=shares).T.asfreq('B').ffill()

    covariances = 52.0 * \
                  prices.asfreq('W-FRI').pct_change().iloc[1:, :].cov().values

    return covariances


"""
Brownian bridge
"""
def brownian_bridge(N, a, b):
    dt = 1.0 / (N-1)
    B = np.empty((1, N), dtype=np.float32)
    B[:, 0] = a-b
    for n in range(N - 2):
         t = n * dt
         xi = np.random.randn() * np.sqrt(dt)
         B[:, n + 1] = B[:, n] * (1-dt / (1-t)) + xi
    B[:, -1] = 0
    return B+b
"""
M = 1
N = 100 #Steps
B = sample_path_batch(M, N, 0, 0)
"""


def timestamp2str(ts):
    """ Converts Timestamp object to str containing date and time
    """
    date = ts.date().strftime("%Y-%m-%d")
    return date


def get_now():
    """ Return current datetime as str
    """
    return timestamp2str(datetime.datetime.now())


def dir_exists(foldername):
    """ Return True if folder exists, else False
    """
    return os.path.isdir(foldername)


"""
load indicators for the rotational strategy
"""


def load_economic_curves(start, end):
    list_fundamental = ['T10YIE', 'DFII20', 'T10Y2Y']
    df_fundamental = web.DataReader(list_fundamental, "fred", start=start, end=end)
    df_fundamental = df_fundamental.dropna()
    df_fundamental['T10YIE_T10Y2Y'] = df_fundamental['T10YIE'] - df_fundamental['T10Y2Y']
    df_fundamental = df_fundamental.drop(['T10YIE'], axis=1)
    df_fundamental['Max'] = df_fundamental.idxmax(axis=1)
    df_fundamental.index.name = 'Date'
    return df_fundamental


def strat_dictionary(stratname):
    assert stratname in ['sixtyforty', 'onlystocks',
                         'vanillariskparity', 'uniform', 'riskparity', 'meanvar'], "unknown strategy"

    if stratname == 'sixtyforty':
        return sixtyforty

    elif stratname == 'onlystocks':
        return onlystocks

    elif stratname == 'vanillariskparity':
        return vanillariskparity

    elif stratname == 'riskparity':
        return riskparity

    elif stratname == 'uniform':
        return uniform

    elif stratname == 'meanvarStrat':
        return meanvarStrat


"""
Functions to calculate performance metrics

From: http://www.turingfinance.com/computational-investing-with-python-week-one/
http://quantopian.github.io/empyrical/_modules/empyrical/stats.html (for omega ratio and validation)

"""

"""
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
"""


def vol(returns):
    # Return the standard deviation of returns
    return np.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = np.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    if np.std(market) != 0:
        return np.cov(m)[0][1] / np.std(market)
    else:
        return math.nan


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the difference to the power of order
    return np.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def var(returns, alpha):
    # This method calculates the historical simulation var of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    if index > 0:
        return abs(sum_var / index)
    else:
        return math.nan


def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def average_dd_squared(returns, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def treynor_ratio(er, returns, market, rf):
    if beta(returns, market) != 0:
        return (er - rf) / beta(returns, market)
    else:
        return math.nan

def sharpe_ratio(er, returns, rf):
    if vol(returns) != 0:
        return (er - rf) / vol(returns)
    else:
        return math.nan

def information_ratio(returns, benchmark):
    diff = returns - benchmark
    if vol(diff) != 0:
        return np.mean(diff) / vol(diff)
    else:
        return math.nan


def modigliani_ratio(er, returns, benchmark, rf):
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(er, returns, rf, alpha):
    if var(returns, alpha) != 0:
        return (er - rf) / var(returns, alpha)
    else:
        return math.nan


def conditional_sharpe_ratio(er, returns, rf, alpha):
    if cvar(returns, alpha) != 0:
        return (er - rf) / cvar(returns, alpha)
    else:
        return math.nan


def omega_ratio(er, returns, rf, target=0):
    """
    Omega ratio definition replaced by the definition found in http://quantopian.github.io/empyrical/_modules/empyrical/stats.html
    that matches the Wikipedia definition https://en.wikipedia.org/wiki/Omega_ratio

    old definition:

    def omega_ratio(er, returns, rf, target=0):
        return (er - rf) / lpm(returns, target, 1)
    """
    if lpm(returns, target+rf, 1) != 0:
        return -hpm(returns, target+rf, 1)/lpm(returns, target+rf, 1)
    else:
        return math.nan


def sortino_ratio(er, returns, rf, target=0):
    if math.sqrt(lpm(returns, target, 2)) != 0:
        return (er - rf) / math.sqrt(lpm(returns, target, 2))
    else:
        return math.nan


def kappa_three_ratio(er, returns, rf, target=0):
    if math.pow(lpm(returns, target, 3), float(1 / 3)) != 0:
        return (er - rf) / math.pow(lpm(returns, target, 3), float(1 / 3))
    else:
        return math.nan


def gain_loss_ratio(returns, target=0):
    if lpm(returns, target, 1) != 0:
        return hpm(returns, target, 1) / lpm(returns, target, 1)
    else:
        return math.nan


def upside_potential_ratio(returns, target=0):
    if math.sqrt(lpm(returns, target, 2)) != 0:
        return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))
    else:
        return math.nan


def calmar_ratio(er, returns, rf):
    if max_dd(returns)!=0:
        return (er - rf) / max_dd(returns)
    else:
        return math.nan



def sterling_ration(er, returns, rf, periods):
    if average_dd(returns, periods)!=0:
        return (er - rf) / average_dd(returns, periods)
    else:
        return math.nan


def burke_ratio(er, returns, rf, periods):
    if math.sqrt(average_dd_squared(returns, periods))!=0:
        return (er - rf) / math.sqrt(average_dd_squared(returns, periods))
    else:
        return math.nan

"""
def test_risk_metrics():
    # This is just a testing method
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    print("vol =", vol(r))
    print("beta =", beta(r, m))
    print("hpm(0.0)_1 =", hpm(r, 0.0, 1))
    print("lpm(0.0)_1 =", lpm(r, 0.0, 1))
    print("VaR(0.05) =", var(r, 0.05))
    print("CVaR(0.05) =", cvar(r, 0.05))
    print("Drawdown(5) =", dd(r, 5))
    print("Max Drawdown =", max_dd(r))


def test_risk_adjusted_metrics():
    # Returns from the portfolio (r) and market (m)
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    # Expected return
    e = np.mean(r)
    # Risk free rate
    f = 0.06
    # Risk-adjusted return based on Volatility
    print("Treynor Ratio =", treynor_ratio(e, r, m, f))
    print("Sharpe Ratio =", sharpe_ratio(e, r, f))
    print("Information Ratio =", information_ratio(r, m))
    # Risk-adjusted return based on Value at Risk
    print("Excess VaR =", excess_var(e, r, f, 0.05))
    print("Conditional Sharpe Ratio =", conditional_sharpe_ratio(e, r, f, 0.05))
    # Risk-adjusted return based on Lower Partial Moments
    print("Omega Ratio =", omega_ratio(e, r, f))
    print("Sortino Ratio =", sortino_ratio(e, r, f))
    print("Kappa 3 Ratio =", kappa_three_ratio(e, r, f))
    print("Gain Loss Ratio =", gain_loss_ratio(r))
    print("Upside Potential Ratio =", upside_potential_ratio(r))
    # Risk-adjusted return based on Drawdown risk
    print("Calmar Ratio =", calmar_ratio(e, r, f))
    print("Sterling Ratio =", sterling_ration(e, r, f, 5))
    print("Burke Ratio =", burke_ratio(e, r, f, 5))


if __name__ == "__main__":
    test_risk_metrics()
    test_risk_adjusted_metrics()
"""