import pandas as pd
import os
import backtrader as bt
import numpy as np
import datetime
from strategies import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pandas_datareader.data as web
import math
import numpy.random as nrand
from GLOBAL_VARS import *
from PortfolioDB import PortfolioDB
import platform
import streamlit as st

# finds file in a folder
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files or name in dirs:
            return os.path.join(root, name)

# Converts a date in "yyyy-mm-dd" format to a dateTime object
def convertDate(dateString):
   return datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S')

# Takes in a date in the format "yyyy-mm-dd hh:mm:ss" and increments it by one day. Or if the
# day is a Friday, increment by 3 days, so the next day of data we get is the next
# Monday.
def incrementDate(dateString):
    dateTime = datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S')
    # If the day of the week is a friday increment by 3 days.
    if dateTime.isoweekday() == 5:
        datePlus = dateTime + timedelta(3)
    else:
        datePlus = dateTime + timedelta(1)
    return str(datePlus)

# Clear the output folder only the first time this is run
@st.cache(allow_output_mutation=True)
def delete_output_first():
    outputdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputdir = find("output", outputdir)
    delete_in_dir(outputdir)

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

def print_header(params, strategy_list):
    print('##############################################################################')
    print('##############################################################################')
    print('### Backtest starting')
    print('###  Parameters:')
    print('###    historic' + ' ' + str(params['historic']))
    print('###    shares' + ' ' + str(params['shares']))
    print('###    shareclass' + ' ' + str(params['shareclass']))
    print('###    weights' + ' ' + str(params['weights']))
    print('###    indicators' + ' ' + str(params['indicator']))
    print('###    initial_cash' + ' ' + str(params['initial_cash']))
    print('###    contribution' + ' ' + str(params['contribution']))
    print('###    create_report' + ' ' + str(params['create_report']))
    print('###    report_name' + ' ' + str(params['report_name']))
    print('###    strategy' + ' ' + str(strategy_list))
    print('###    startdate' + ' ' + str(params['startdate']))
    print('###    enddate' + ' ' + str(params['enddate']))
    print('###    leverage' + ' ' + str(params['leverage']))
    print('##############################################################################')

def import_histprices_db(dataLabel):
    db = PortfolioDB(databaseName = DB_NAME)
    df = db.getPrices(dataLabel)

    stock_info = db.getStockInfo(dataLabel)
    if stock_info['treatment_type'].values[0] == 'yield':
        maturity = stock_info['maturity'].values[0]
        if not type(maturity) == np.float64:
            print("Error: maturity is needed for ticker " + dataLabel + '. Please update DIM_STOCKS.')
            #return

        frequency = stock_info['frequency'].values[0]
        if frequency == 'D':
            dt = 1 / params['DAYS_IN_YEAR_BOND_PRICE']
        elif frequency == 'Y':
            dt = 1

        total_return = bond_total_return(ytm=df[['close']], dt=dt, maturity=maturity)
        df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
        df['close'].iloc[0] = 100
        df = df.dropna()

    df = df.set_index('date')
    df.index.name = 'Date'
    df = df[['close','open','high','low','volume']]

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
        pct_change = (pct_change - expense_ratio / params['DAYS_IN_YEAR']) * leverage
    elif timeframe == bt.TimeFrame.Years:
        pct_change = ((1 + pct_change) ** (1 / params['DAYS_IN_YEAR'])) - 1 # Transform into daily returns
        pct_change = (pct_change - expense_ratio / params['DAYS_IN_YEAR']) * leverage # Apply leverage
        pct_change = ((1 + pct_change) ** params['DAYS_IN_YEAR']) - 1 # Re-transform into yearly returns
    new_price = initial_value * (1 + pct_change).cumprod()
    new_price.iloc[0] = initial_value
    return new_price

"""
bond total return based on formula 5 of paper 
https://mpra.ub.uni-muenchen.de/92607/1/MPRA_paper_92607.pdf
See also https://quant.stackexchange.com/questions/22837/how-to-calculate-us-treasury-total-return-from-yield

"""
def bond_total_return(ytm, dt, maturity):
    ytm_pc = ytm.to_numpy()/100

    P0 = 1/np.power(1+ytm_pc, maturity) # price

    # first and second price derivatives
    P0_backward = np.roll(P0, 1)
    P0_backward = np.delete(P0_backward, 0, axis=0)
    P0_backward = np.delete(P0_backward, len(P0_backward)-1, axis=0)

    P0_forward = np.roll(P0, -1)
    P0_forward = np.delete(P0_forward, 0, axis=0)
    P0_forward = np.delete(P0_forward, len(P0_forward) - 1, axis=0)

    d_ytm = np.roll(np.diff(ytm_pc, axis=0),-1)
    d_ytm = np.delete(d_ytm, len(d_ytm)-1, axis=0)

    dP0_dytm = (P0_forward-P0_backward)/(2*d_ytm)
    dP0_dytm[dP0_dytm == np.inf] = 0
    dP0_dytm[dP0_dytm == -np.inf] = 0
    dP0_dytm[np.isnan(dP0_dytm)] = 0

    d2P0_dytm2 = (P0_forward - 2 * P0[1:len(P0)-1] + P0_backward) / (np.power(d_ytm, 2))
    d2P0_dytm2[d2P0_dytm2 == np.inf] = 0
    d2P0_dytm2[d2P0_dytm2 == -np.inf] = 0
    d2P0_dytm2[np.isnan(d2P0_dytm2)] = 0

    # Duration and convexity
    duration = -dP0_dytm/P0[1:len(P0)-1]
    convexity = d2P0_dytm2/P0[1:len(P0)-1]

    yield_income = (np.log(1+ytm_pc[1:len(ytm_pc)-1])*dt)   # First term
    duration_term = -duration * d_ytm
    convexity_term = 0.5 * (convexity - np.power(duration,2)) * np.power(d_ytm,2)
    time_term = (1/(1+ytm_pc))[1:len(ytm_pc)-1]*dt*d_ytm
    total_return = yield_income + duration_term + convexity_term + time_term

    total_return_df = pd.DataFrame(data=total_return, index=ytm.index[1:len(ytm.index)-1])
    total_return_df.columns = ["total_return"]
    return total_return_df

def common_dates(data, fromdate, todate, timeframe):
    # Get latest startdate and earlier end date
    start = fromdate
    end = todate
    for i in range(0, len(data)):
        start = max(data[i].index[0], start)
        end = min(data[i].index[-1], end)

    if timeframe == bt.TimeFrame.Days: # 5
        dates = pd.bdate_range(start, end)
    elif timeframe == bt.TimeFrame.Years: # 8
        dates = pd.date_range(start,end,freq='ys')

    left = pd.DataFrame(index=dates)
    data_dates = []
    for i in range(0, len(data)):
        right=data[i]
        this_data_dates = pd.merge(left, right, left_index=True,right_index=True, how="left")
        this_data_dates = this_data_dates.fillna(method='ffill')
        data_dates.append(this_data_dates)
    return data_dates


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


def covariances(shares, start, end):
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
    #return timestamp2str(datetime.datetime.now())
    return timestamp2str(datetime.now())


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
        return hpm(returns, target+rf, 1)/lpm(returns, target+rf, 1)
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