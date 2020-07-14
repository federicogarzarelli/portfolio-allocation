import backtrader as bt
import sys
import matplotlib.pyplot as plt
import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from utils import timestamp2str, get_now, dir_exists
import numpy as np
from strategies import *
import pyfolio as pf


class PerformanceReport:
    """ Report with performance stats for given backtest run
    """

    def __init__(self, stratbt, outfile, user, memo, system):
        self.stratbt = stratbt  # works for only 1 strategy
        self.outfile = outfile
        self.user = user
        self.memo = memo
        self.check_and_assign_defaults()
        self.system = system

    def check_and_assign_defaults(self):
        """ Check initialization parameters or assign defaults
        """
        if not self.user:
            self.user = 'Fabio & Federico'
        if not self.memo:
            self.memo = 'Testing'


    def get_performance_stats(self):
        """ Return dict with performance stats for given strategy withing backtest
        """
        st = self.stratbt
        dt = self.get_date_index()

        bt_period = dt[-1] - dt[0]
        bt_period_days = bt_period.days
        drawdown = st.analyzers.myDrawDown.get_analysis()
        sharpe_ratio = st.analyzers.mySharpe.get_analysis()['sharperatio']
        returns = st.analyzers.myReturns.get_analysis() # For the annual return in fund mode
        annualreturns = st.analyzers.myAnnualReturn.get_analysis() # For total and annual returns in asset mode
        endValue = st.observers.broker.lines[1].array[len(dt)-1:len(dt)][0]
        vwr = st.analyzers.myVWR.get_analysis()['vwr']

        tot_return = 1
        for key, value in annualreturns.items():
            tot_return = tot_return * (1 + value)
        tot_return = tot_return - 1

        kpi = {# PnL
               'start_cash': self.get_startcash(),
               'end_value': endValue,
               'total_return': 100*tot_return,
               'annual_return': returns['rnorm100'],
               'annual_return_asset': 100*((1 + tot_return)**(365.25 / bt_period_days) - 1),
               'max_money_drawdown': drawdown['max']['moneydown'],
               'max_pct_drawdown': drawdown['max']['drawdown'],
               #  performance
               'vwr': vwr,
               'sharpe_ratio': sharpe_ratio,
               }
        return kpi

    def get_equity_curve(self):
        """ Return series containing equity curve
        """
        st = self.stratbt
        value = st.observers.broker.lines[1].array
        vv = np.asarray(value)
        vv = vv[~np.isnan(vv)]

        dt = self.get_date_index()

        curve = pd.Series(data=vv, index=dt)
        return 100 * curve / curve.iloc[0]

    def __str__(self):
        msg = ("*** PnL: ***\n"
               "Start capital         : {start_cash:4.2f}\n"
               "End capital           : {end_value:4.2f}\n"
               "Total return          : {total_return:4.2f}%\n"
               "Annual return (asset) : {annual_return_asset:4.2f}%\n"
               "Annual return (fund)  : {annual_return:4.2f}%\n"
               "Max. money drawdown   : {max_money_drawdown:4.2f}\n"
               "Max. percent drawdown : {max_pct_drawdown:4.2f}%\n\n"
               "*** Performance ***\n"
               "Variability Weighted Return: {vwr:4.2f}\n"
               "Sharpe ratio          : {sharpe_ratio:4.2f}\n"
               )
        kpis = self.get_performance_stats()
        # see: https://stackoverflow.com/questions/24170519/
        # python-# typeerror-non-empty-format-string-passed-to-object-format
        kpis = {k: -999 if v is None else v for k, v in kpis.items()}
        return msg.format(**kpis)

    def plot_equity_curve(self, fname='equity_curve.png'):
        """ Plots equity curve to png file
        """
        curve = self.get_equity_curve()
        xrnge = [curve.index[0], curve.index[-1]]
        dotted = pd.Series(data=[100, 100], index=xrnge)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Net Asset Value (start=100)')
        ax.set_title('Equity curve')
        _ = curve.plot(kind='line', ax=ax)
        _ = dotted.plot(kind='line', ax=ax, color='grey', linestyle=':')
        return fig

    def _get_periodicity(self):
        """ Maps length backtesting interval to appropriate periodiciy for return plot
        """
        curve = self.get_equity_curve()
        startdate = curve.index[0]
        enddate = curve.index[-1]
        time_interval = enddate - startdate
        time_interval_days = time_interval.days
        if time_interval_days > 5 * 365.25:
            periodicity = ('Yearly', 'Y')
        elif time_interval_days > 365.25:
            periodicity = ('Monthly', 'M')
        elif time_interval_days > 50:
            periodicity = ('Weekly', '168H')
        elif time_interval_days > 5:
            periodicity = ('Daily', '24H')
        elif time_interval_days > 0.5:
            periodicity = ('Hourly', 'H')
        elif time_interval_days > 0.05:
            periodicity = ('Per 15 Min', '15M')
        else: periodicity = ('Per minute', '1M')
        return periodicity

    def plot_return_curve(self, fname='return_curve.png'):
        """ Plots return curve to png file
        """
        curve = self.get_equity_curve()
        period = self._get_periodicity()
        values = curve.resample(period[1]).ohlc()['close']
        returns = values.diff() / values
        returns.index = returns.index.date
        is_positive = returns > 0
        fig, ax = plt.subplots(1, 1)
        ax.set_title("{} returns".format(period[0]))
        ax.set_xlabel("date")
        ax.set_ylabel("return (%)")
        _ = returns.plot.bar(color=is_positive.map({True: 'green', False: 'red'}), ax=ax)
        return fig

    def generate_html(self):
        """ Returns parsed HTML text string for report
        """
        basedir = os.path.abspath(os.path.dirname(__file__))
        images = os.path.join(basedir, 'templates')
        eq_curve = os.path.join(images, 'equity_curve.png')
        rt_curve = os.path.join(images, 'return_curve.png')
        fig_equity = self.plot_equity_curve()
        fig_equity.savefig(eq_curve)
        fig_return = self.plot_return_curve()
        fig_return.savefig(rt_curve)
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("templates/template.html")
        header = self.get_header_data()
        kpis = self.get_performance_stats()
        if self.system == 'windows':
            graphics = {'url_equity_curve': 'file:\\' + eq_curve,
                        'url_return_curve': 'file:\\' + rt_curve
                        }
        else:
            graphics = {'url_equity_curve': 'file://' + eq_curve,
                        'url_return_curve': 'file://' + rt_curve
                        }

        # weights
        weights = self.get_weights().tail(30)
        formatter = {}
        for i in weights.columns:
            formatter[i] = '{:,.1%}'.format
        weights = weights.to_html(formatters=formatter, classes=["table table-hover"],index=True,escape=False,col_space='200px')
        weights = {'weights_table':weights}
        all_numbers = {**header, **kpis, **graphics, **weights}
        html_out = template.render(all_numbers)
        return html_out

    def get_weights(self):
        st = self.stratbt
        n_assets = self.get_strategy_params().get('n_assets')

        # Asset weights
        size_weights = 100  # get weights for the last 60 days
        idx = self.get_date_index()[len(self.get_date_index()) - size_weights:len(self.get_date_index())]
        weight_df = pd.DataFrame(index=idx)

        for i in range(0, n_assets):
            weight_df['asset_' + str(i)] = st.observers.weightsobserver.lines[i].get(size=size_weights)

        return weight_df

    def generate_pdf_report(self):
        """ Returns PDF report with backtest results
        """
        html = self.generate_html()
        outfile = self.outfile + "_" + self.get_strategy_name() + "_" + get_now() + ".pdf"
        HTML(string=html).write_pdf(outfile)
        msg = "See {} for report with backtest results."
        print(msg.format(outfile))

    def get_strategy_name(self):
        return self.stratbt.__class__.__name__

    def get_strategy_params(self):
        return self.stratbt.cerebro.strats[0][0][-1]

    def get_date_index(self):
        st = self.stratbt
        # Get dates from the observer
        year = st.observers.getdate.lines[0].array
        month = st.observers.getdate.lines[1].array
        day = st.observers.getdate.lines[2].array

        # Put all together and drop na
        df = pd.DataFrame(data = list(zip(day.tolist(), month.tolist(), year.tolist())),
                          columns=["day", "month", "year"])
        df = df.dropna()

        # Transform into DatetimeIndex
        df = pd.to_datetime(df[["day", "month", "year"]])
        df.index = df
        return df.index

    def get_start_date(self):
        """ Return first datafeed datetime
        """

        dt = self.get_date_index()
        return timestamp2str(dt[0])

    def get_end_date(self):
        """ Return first datafeed datetime
        """
        dt = self.get_date_index()
        return timestamp2str(dt[-1])

    def get_header_data(self):
        """ Return dict with data for report header
        """
        header = {'strategy_name': self.get_strategy_name(),
                  'params': self.get_strategy_params(),
                  'start_date': self.get_start_date(),
                  'end_date': self.get_end_date(),
                  'name_user': self.user,
                  'processing_date': get_now(),
                  'memo_field': self.memo
                  }
        return header

    def get_startcash(self):
        return self.stratbt.broker.startingcash

    def get_pyfolio(self):
        st = self.stratbt
        pyfoliozer  = st.analyzers.getbyname('myPyFolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        return returns, positions, transactions, gross_lev

    def generate_pyfolio_report(self):
        returns, positions, transactions, gross_lev = self.get_pyfolio()
        """
         pf.create_simple_tear_sheet(
            returns,
            positions=positions,
            transactions=transactions)       
        """
        pf.create_returns_tear_sheet(
            returns,
            positions=positions,
            transactions=transactions)

    def get_aggregated_data(self):
        kpis = self.get_performance_stats()
        kpis_df = pd.DataFrame.from_dict(kpis, orient='index')
        kpis_df.loc['total_return'] = kpis_df.loc['total_return']/100
        kpis_df.loc['annual_return'] = kpis_df.loc['annual_return'] / 100
        kpis_df.loc['annual_return_asset'] = kpis_df.loc['annual_return_asset'] / 100
        kpis_df.loc['max_pct_drawdown'] = kpis_df.loc['max_pct_drawdown'] / 100
        kpis_df.loc['vwr'] = kpis_df.loc['vwr'] / 100

        returns, positions, transactions, gross_lev = self.get_pyfolio()
        perf_stats_all = pf.timeseries.perf_stats(returns=returns, factor_returns=None, positions=None, transactions=None,
                                   turnover_denom="AGB")
        all_stats = pd.concat([kpis_df, perf_stats_all], keys=['backtrader', 'pyfolio'])
        all_stats.columns = [self.get_strategy_name()]
        return all_stats

    def output_all_data(self):
        prices = self.get_equity_curve()
        prices.index = prices.index.date
        returns = prices.diff() / prices

        prices = pd.DataFrame(data=prices, columns=[self.get_strategy_name()])
        returns = pd.DataFrame(data=returns, columns=[self.get_strategy_name()])
        returns = returns.dropna()

        perf_data = self.get_aggregated_data()
        weight = self.get_weights().tail(1).T
        weight.columns = [self.get_strategy_name()]
        return prices, returns, perf_data, weight


class Cerebro(bt.Cerebro):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.add_report_analyzers()
        self.add_report_observers()

    def add_report_observers(self):
        self.addobserver(GetDate)


    def add_report_analyzers(self, riskfree=0.01):
            """ Adds performance stats, required for report
            """
            self.addanalyzer(bt.analyzers.SharpeRatio,
                             _name="mySharpe",
                             riskfreerate=riskfree,
                             timeframe=bt.TimeFrame.Days,
                             convertrate=True,
                             factor=252,
                             annualize=True,
                             fund=True)
            self.addanalyzer(bt.analyzers.DrawDown, fund=True,
                             _name="myDrawDown")
            self.addanalyzer(bt.analyzers.AnnualReturn,
                             _name="myAnnualReturn")
            self.addanalyzer(bt.analyzers.Returns, fund=True,
                             _name="myReturns")
            self.addanalyzer(bt.analyzers.SQN,
                             _name="mySqn")
            self.addanalyzer(bt.analyzers.VWR,
                             timeframe=bt.TimeFrame.Days,
                             tau=2,
                             sdev_max=0.2,
                             fund=True,
                             _name="myVWR")
            self.addanalyzer(bt.analyzers.PyFolio,
                             _name="myPyFolio")



    def get_strategy_backtest(self):
        return self.runstrats[0][0]

    def report(self, outfile, user=None, memo=None, system=None, report_type=None):
        bt = self.get_strategy_backtest()
        rpt = PerformanceReport(bt, outfile=outfile, user=user, memo=memo, system=system)
        rpt.generate_pdf_report()
        rpt.generate_pyfolio_report()

        prices, returns, perf_data, weight = rpt.output_all_data()
        return prices, returns, perf_data, weight



