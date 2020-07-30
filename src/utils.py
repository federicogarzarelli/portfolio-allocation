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


"""
Deletes all files in a directory
"""


def delete_in_dir(mydir, *args, **kwargs):
    file_extension = kwargs.get('file_extension', None)
    if file_extension is None:
        filelist = [f for f in os.listdir(mydir)]
    else:
        filelist = [f for f in os.listdir(mydir) if f.endswith(file_extension)]

    for f in filelist:
        os.remove(os.path.join(mydir, f))


"""
Prints a section divider with the strategy name
"""


def print_section_divider(strategy_name):
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


def add_leverage(proxy, leverage=1, expense_ratio=0.0):
    """
    Simulates a leverage ETF given its proxy, leverage, and expense ratio.

    Daily percent change is calculated by taking the daily log-return of
    the price, subtracting the daily expense ratio, then multiplying by the leverage.
    """
    initial_value = proxy.iloc[0]
    pct_change = proxy.pct_change(1)
    pct_change = (pct_change - expense_ratio / 252) * leverage
    new_price = initial_value * (1 + pct_change).cumprod()
    new_price[0] = initial_value
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


def covariances(shares=['GLD', 'TLT', 'SPY'],
                start=datetime.datetime(2020, 1, 1),
                end=datetime.datetime(2020, 6, 1)):
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
def brownian_bridge(M, N, a, b, sigma):
    dt = 1.0 / (N-1)
    dt_sqrt = np.sqrt(dt)
    B = np.empty((M, N), dtype=np.float32)
    B[:, 0] = a-b
    for n in range(N - 2):
         t = n * dt
         xi = np.random.randn(M) * dt_sqrt * sigma
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
