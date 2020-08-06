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
import pybloqs.block.table_formatters as tf
from pybloqs import Block
from scipy import stats
from datetime import timedelta
from myanalyzers import MyAnnualReturn, MyTimeReturn, MySharpeRatio, MyReturns, MyDrawDown, \
                        MyTimeDrawDown, MyLogReturnsRolling, MyDistributionMoments, MyRiskAdjusted_VolBased, \
                        MyRiskAdjusted_VaRBased, MyRiskAdjusted_LPMBased, MyRiskAdjusted_DDBased


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

    def get_timeframe(self):
        datediff = stats.mode(np.diff(self.get_date_index().to_list()))[0][0]
        if datediff > timedelta(days=250):
            #print("Report: you are using data with yearly frequency")
            return "Years"
        elif datediff < timedelta(days=2):
            #print("Report: you are using data with daily frequency")
            return "Days"


    def get_performance_stats(self):
        """ Return dict with performance stats for given strategy withing backtest
        """
        timeframe = self.get_timeframe()

        st = self.stratbt
        dt = self.get_date_index()

        # Total period for the backtesting in days
        bt_period = dt[-1] - dt[0]
        bt_period_days = bt_period.days
        bt_period_years = round(bt_period.days/365.2422)

        # Import analyzers results
        # Returns
        annualreturns = st.analyzers.myAnnualReturn.get_analysis() # For total and annual returns in asset mode
        timeret = st.analyzers.myTimeReturn.get_analysis()
        logret =st.analyzers.myLogReturnsRolling.get_analysis()
        returns = st.analyzers.myReturns.get_analysis() # For the annual return in fund mode

        # Drawdowns
        drawdown = st.analyzers.myDrawDown.get_analysis()
        timedd = st.analyzers.myTimeDrawDown.get_analysis()

        # Distribution
        ret_distrib = st.analyzers.MyDistributionMoments.get_analysis()

        # Risk-adjusted return based on Volatility
        RiskAdjusted_VolBased = st.analyzers.MyRiskAdjusted_VolBased.get_analysis()
        sharpe_ratio = st.analyzers.mySharpe.get_analysis()['sharperatio']

        # Risk-adjusted return based on Value at Risk
        RiskAdjusted_VaRBased = st.analyzers.MyRiskAdjusted_VaRBased.get_analysis()

        # Risk-adjusted return based on Lower Partial Moments
        RiskAdjusted_LPMBased = st.analyzers.MyRiskAdjusted_LPMBased.get_analysis()

        # Risk-adjusted return based on Drawdown risk
        RiskAdjusted_DDBased = st.analyzers.MyRiskAdjusted_DDBased.get_analysis()

        vwr = st.analyzers.myVWR.get_analysis()['vwr']

        # Calculate end value and total return (portfolio asset mode)
        endValue = st.observers.broker.lines[1].get(size=len(dt))[-1]

        tot_return = 1
        for key, value in timeret.items():
            tot_return = tot_return * (1 + value)
        tot_return = tot_return - 1

        if timeframe == "Days":
            annual_return_asset = 100 * ((1 + tot_return) ** (365.2422 / bt_period_days) - 1)
        elif timeframe == "Years":
            annual_return_asset = 100 * ((1 + tot_return) ** (1 / bt_period_years) - 1)

        kpi = {# PnL
               'start_cash': self.get_startcash(),
               'end_value': endValue,
               'total_return': 100*tot_return,
               'annual_return': 100*returns['rnorm'],
               'annual_return_asset': annual_return_asset,
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
        dt = self.get_date_index()
        value = st.observers.broker.lines[1].get(size=len(dt))
        vv = np.asarray(value)
        vv = vv[~np.isnan(vv)]

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
        kpis = self.get_aggregated_data_html()
        if self.system == 'windows':
            graphics = {'url_equity_curve': 'file:\\' + eq_curve,
                        'url_return_curve': 'file:\\' + rt_curve
                        }
        else:
            graphics = {'url_equity_curve': 'file://' + eq_curve,
                        'url_return_curve': 'file://' + rt_curve
                        }

        # targetweights
        targetweights, effectiveweights = self.get_weights() # only last month
        timeframe = self.get_timeframe()
        if timeframe == "Days": # if daily frequency, take the last month worth of weights
            targetweights = targetweights.tail(30)
            effectiveweights = effectiveweights.tail(30)
        elif timeframe == "Years": # if daily frequency, take the last 5 years
            targetweights = targetweights.tail(5)
            effectiveweights = effectiveweights.tail(5)

        fmt_pct = tf.FmtPercent(1, apply_to_header_and_index=False)
        fmt_align = tf.FmtAlignTable("left")
        fmt_background = tf.FmtStripeBackground(first_color=tf.colors.LIGHT_GREY, second_color=tf.colors.WHITE, header_color=tf.colors.BLACK)

        targetweights = Block(targetweights, formatters=[fmt_pct, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        effectiveweights = Block(effectiveweights, formatters=[fmt_pct, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()

        targetweights = {'targetweights_table':targetweights}
        effectiveweights = {'effectiveweights_table':effectiveweights}

        all_numbers = {**header, **kpis, **graphics, **targetweights, **effectiveweights}
        html_out = template.render(all_numbers)
        return html_out

    def get_weights(self):
        st = self.stratbt
        n_assets = self.get_strategy_params().get('n_assets')

        # Asset targetweights
        size_weights = 100  # get targetweights for the last 100 days
        if len(self.get_date_index()) > size_weights:
            idx = self.get_date_index()[len(self.get_date_index()) - size_weights:len(self.get_date_index())]
        else:
            size_weights = len(self.get_date_index())
            idx = self.get_date_index()[0:len(self.get_date_index())]

        targetweights_df = pd.DataFrame(index=idx)
        effectiveweights_df = pd.DataFrame(index=idx)

        for i in range(0, n_assets):
            targetweights_df[st.assets[i]._name] = st.observers.targetweightsobserver.lines[i].get(size=size_weights)
            effectiveweights_df[st.assets[i]._name] = st.observers.effectiveweightsobserver.lines[i].get(size=size_weights)
        return targetweights_df, effectiveweights_df

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
        df = df[df >= st.startdate]
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
        pyfoliozer = st.analyzers.getbyname('myPyFolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        return returns, positions, transactions, gross_lev

    def generate_pyfolio_report(self):
        timeframe = self.get_timeframe()
        if timeframe == "Days":
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
            return
        else:
            return

    def get_aggregated_data(self):
        kpis = self.get_performance_stats()
        kpis_df = pd.DataFrame.from_dict(kpis, orient='index')
        kpis_df.loc['total_return'] = kpis_df.loc['total_return']/100
        kpis_df.loc['annual_return'] = kpis_df.loc['annual_return'] / 100
        kpis_df.loc['annual_return_asset'] = kpis_df.loc['annual_return_asset'] / 100
        kpis_df.loc['max_pct_drawdown'] = kpis_df.loc['max_pct_drawdown'] / 100
        kpis_df.loc['vwr'] = kpis_df.loc['vwr'] / 100

        timeframe = self.get_timeframe()
        if timeframe == "Days": # Pyfolio works only with daily data
            returns, positions, transactions, gross_lev = self.get_pyfolio()
            perf_stats_all = pf.timeseries.perf_stats(returns=returns, factor_returns=None, positions=None, transactions=None,
                                       turnover_denom="AGB")
            all_stats = pd.concat([kpis_df, perf_stats_all], keys=['backtrader', 'pyfolio'])
        else:
            all_stats = pd.concat([kpis_df], keys=['backtrader'])

        all_stats.columns = [self.get_strategy_name()]
        return all_stats

    def get_aggregated_data_html(self):
        perf_data = self.get_aggregated_data()
        timeframe = self.get_timeframe()
        if timeframe == "Days":
            pct_rows=[('backtrader','total_return'),
                      ('backtrader', 'total_return'),
                      ('backtrader', 'annual_return'),
                      ('backtrader', 'annual_return_asset'),
                      ('backtrader', 'max_pct_drawdown'),
                      ('backtrader', 'vwr'),
                      ('pyfolio', 'Annual return'),
                      ('pyfolio', 'Cumulative returns'),
                      ('pyfolio', 'Annual volatility'),
                      ('pyfolio', 'Max drawdown'),
                      ('pyfolio', 'Daily value at risk')]
            dec_rows=[('backtrader','start_cash'),
                      ('backtrader','end_value'),
                      ('backtrader',  'max_money_drawdown'),
                      ('backtrader', 'sharpe_ratio'),
                      ('pyfolio', 'Sharpe ratio'),
                      ('pyfolio', 'Calmar ratio'),
                      ('pyfolio', 'Stability'),
                      ('pyfolio', 'Omega ratio'),
                      ('pyfolio', 'Sortino ratio'),
                      ('pyfolio', 'Skew'),
                      ('pyfolio', 'Kurtosis'),
                      ('pyfolio', 'Tail ratio')]
        else:
            pct_rows=[('backtrader','total_return'),
                      ('backtrader', 'total_return'),
                      ('backtrader', 'annual_return'),
                      ('backtrader', 'annual_return_asset'),
                      ('backtrader', 'max_pct_drawdown'),
                      ('backtrader', 'vwr')]
            dec_rows=[('backtrader','start_cash'),
                      ('backtrader','end_value'),
                      ('backtrader',  'max_money_drawdown'),
                      ('backtrader', 'sharpe_ratio')]

        fmt_pct = tf.FmtPercent(1, rows=pct_rows, apply_to_header_and_index=False)
        fmt_dec = tf.FmtDecimals(2, rows=dec_rows, apply_to_header_and_index=False)
        fmt_align = tf.FmtAlignTable("left")
        fmt_background = tf.FmtStripeBackground(first_color=tf.colors.LIGHT_GREY, second_color=tf.colors.WHITE, header_color=tf.colors.BLACK)
        fmt_multiindex = tf.FmtExpandMultiIndex(operator=tf.OP_NONE)

        perf_data = Block(perf_data, formatters=[fmt_multiindex, fmt_pct, fmt_dec, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        perf_data = {'performance_table': perf_data}
        return perf_data

    def output_all_data(self):
        prices = self.get_equity_curve()
        prices.index = prices.index.date
        returns = prices.diff() / prices

        prices = pd.DataFrame(data=prices, columns=[self.get_strategy_name()])
        returns = pd.DataFrame(data=returns, columns=[self.get_strategy_name()])
        returns = returns.dropna()

        perf_data = self.get_aggregated_data()
        targetweights, effectiveweights = self.get_weights()
        targetweights = targetweights.tail(1).T
        effectiveweights = effectiveweights.tail(1).T
        targetweights.columns = [self.get_strategy_name()]
        effectiveweights.columns = [self.get_strategy_name()]

        return prices, returns, perf_data, targetweights, effectiveweights


class Cerebro(bt.Cerebro):
    def __init__(self, timeframe=None, **kwds):
        super().__init__(**kwds)
        self.timeframe = timeframe
        self.add_report_analyzers(timeframe = self.timeframe)
        self.add_report_observers()

    def add_report_observers(self):
        self.addobserver(GetDate)


    def add_report_analyzers(self, riskfree=0.01, targetrate=0.01, alpha=0.05, market_mu = 0.07, market_sigma = 0.15,
                                timeframe = None):
            """ Adds performance stats, required for report
            """
            if timeframe == bt.TimeFrame.Years:
                scalar = 1
            elif timeframe == bt.TimeFrame.Days:
                scalar = 365.2422

            # Returns
            self.addanalyzer(MyAnnualReturn, _name="myAnnualReturn")
            self.addanalyzer(MyTimeReturn, _name="myTimeReturn",
                             fund=True)
            self.addanalyzer(MyLogReturnsRolling, _name="myLogReturnsRolling",
                             fund=True)
            self.addanalyzer(MyReturns,_name="myReturns",
                             fund=True,
                             tann=scalar)
            # Drawdowns
            self.addanalyzer(MyDrawDown,_name="myDrawDown",
                             fund=True)
            self.addanalyzer(MyTimeDrawDown,_name="myTimeDrawDown",
                             fund=True)
            # Distribution
            self.addanalyzer(MyDistributionMoments, _name="MyDistributionMoments",
                             timeframe=timeframe,
                             compression=1,
                             annualize=True,
                             factor=scalar,
                             stddev_sample=True,
                             logreturns=True,
                             fund=True)
            # Risk-adjusted return based on Volatility
            self.addanalyzer(MyRiskAdjusted_VolBased, _name="MyRiskAdjusted_VolBased",
                             timeframe=timeframe,
                             compression=1,
                             annualize=True,
                             stddev_sample=True,
                             logreturns=True,
                             fund=True,
                             riskfreerate=riskfree,
                             market_mu = market_mu,
                             market_sigma = market_sigma,
                             factor=scalar,
                             convertrate=False)

            self.addanalyzer(MySharpeRatio, _name="mySharpe",
                             riskfreerate=riskfree,
                             timeframe=timeframe,
                             annualize=True,
                             stddev_sample=True,
                             logreturns=True,
                             fund=True)
            # Risk-adjusted return based on Value at Risk
            self.addanalyzer(MyRiskAdjusted_VaRBased, _name="MyRiskAdjusted_VaRBased",
                             timeframe=timeframe,
                             compression=1,
                             annualize=True,
                             stddev_sample=True,
                             logreturns=True,
                             fund=True,
                             riskfreerate=riskfree,
                             targetrate=targetrate,
                             factor=scalar,
                             convertrate=False,
                             alpha=alpha)
            # Risk-adjusted return based on Lower Partial Moments
            self.addanalyzer(MyRiskAdjusted_LPMBased, _name="MyRiskAdjusted_LPMBased",
                             timeframe=timeframe,
                             compression=1,
                             annualize=True,
                             stddev_sample=True,
                             logreturns=True,
                             fund=True,
                             riskfreerate=riskfree,
                             targetrate=targetrate,
                             factor=scalar,
                             convertrate=False)

            # Risk-adjusted return based on Drawdown risk
            self.addanalyzer(MyRiskAdjusted_DDBased, _name="MyRiskAdjusted_DDBased",
                             timeframe=timeframe,
                             compression=1,
                             annualize=True,
                             stddev_sample=True,
                             logreturns=True,
                             fund=True,
                             riskfreerate=riskfree,
                             factor=scalar,
                             convertrate=False)

            # VWR
            self.addanalyzer(bt.analyzers.VWR,
                                     timeframe=timeframe,
                                     tau=2,
                                     sdev_max=0.2,
                                     fund=True,
                                     _name="myVWR")

            # Pyfolio
            self.addanalyzer(bt.analyzers.PyFolio,
                             timeframe=timeframe,
                             _name="myPyFolio")



    def get_strategy_backtest(self):
        return self.runstrats[0][0]

    def report(self, outfile, user=None, memo=None, system=None, report_type=None):
        bt = self.get_strategy_backtest()
        rpt = PerformanceReport(bt, outfile=outfile, user=user, memo=memo, system=system)
        rpt.generate_pdf_report()

        try:
            rpt.generate_pyfolio_report()
        except:
            print("Error raised in rpt.generate_pyfolio_report().")
            pass

        prices, returns, perf_data, targetweights, effectiveweights = rpt.output_all_data()
        return prices, returns, perf_data, targetweights, effectiveweights



