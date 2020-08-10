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
import pybloqs.block.table_formatters as tf
from pybloqs import Block


class ReportAggregator:
    """ Aggregates one-strategy reports and creates a multi-strategy report
    """

    def __init__(self, outfilename, outputdir, user, memo, system, prices, returns, perf_data, targetweights,
                 effectiveweights):
        self.outfilename = outfilename
        self.outputdir = outputdir
        self.user = user
        self.memo = memo
        self.check_and_assign_defaults()
        self.system = system
        self.prices = prices
        self.returns = returns 
        self.perf_data = perf_data
        self.targetweights = targetweights
        self.effectiveweights = effectiveweights

        self.outfile = os.path.join(self.outputdir, self.outfilename)


    """
    Multistrategy report
    """
    def generate_csv(self):
        # Output into CSV for now, later create a PDF from a HTML
        self.prices.to_csv(self.outputdir+r'/Prices_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.returns.to_csv(self.outputdir+r'/Returns_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.perf_data.to_csv(self.outputdir+r'/Performance_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.targetweights.to_csv(self.outputdir+r'/Target_Weights_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.effectiveweights.to_csv(self.outputdir+r'/Effective_Weights_MultiStrategy_' + get_now().replace(':', '') +'.csv')


    def check_and_assign_defaults(self):
        """ Check initialization parameters or assign defaults
        """
        if not self.outfilename:
            self.outfilename = 'Not given'
        if not dir_exists(self.outputdir):
            msg = "*** ERROR: outputdir {} does not exist."
            print(msg.format(self.outputdir))
            sys.exit(0)
        if not self.user:
            self.user = 'Fabio & Federico'
        if not self.memo:
            self.memo = 'Testing'

    def plot_equity_curve(self):
        """ Plots equity curve to png file
        """
        curve = self.prices
        xrnge = [curve.index[0], curve.index[-1]]
        dotted = pd.Series(data=[100, 100], index=xrnge)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Net Asset Value (start=100)')
        ax.set_title('Equity curve')
        _ = curve.plot(kind='line', ax=ax)
        _ = dotted.plot(kind='line', ax=ax, color='grey', linestyle=':')
        return fig

    def get_strategy_names(self):
        return self.perf_data.columns.to_list()

    def get_start_date(self):
        return self.prices.index[0].strftime("%Y-%m-%d")

    def get_end_date(self):
        return self.prices.index[-1].strftime("%Y-%m-%d")

    def get_performance_stats_html(self):

        pct_rows = [('P&L', 'Total return'),
                    ('P&L', 'Annual return'),
                    ('P&L', 'Annual return (asset mode)'),
                    ('Risk-adjusted return based on Drawdown', 'Max percentage drawdown')]
        dec_rows = [('P&L', 'Starting cash'),
                    ('P&L', 'End value'),
                    ('Risk-adjusted return based on Drawdown', 'Max money drawdown'),
                    # Distribution
                    ('Distribution moments', 'Returns volatility'),
                    ('Distribution moments', 'Returns skewness'),
                    ('Distribution moments', 'Returns kurtosis'),
                    # Risk-adjusted return based on Volatility
                    ('Risk-adjusted return based on Volatility', 'Treynor ratio'),
                    ('Risk-adjusted return based on Volatility', 'Sharpe ratio'),
                    ('Risk-adjusted return based on Volatility', 'Information ratio'),
                    # Risk-adjusted return based on Value at Risk
                    ('Risk-adjusted return based on Value at Risk', 'VaR'),
                    ('Risk-adjusted return based on Value at Risk', 'Expected Shortfall'),
                    ('Risk-adjusted return based on Value at Risk', 'Excess var'),
                    ('Risk-adjusted return based on Value at Risk', 'Conditional sharpe ratio'),
                    # Risk-adjusted return based on Lower Partial Moments
                    ('Risk-adjusted return based on Lower Partial Moments', 'Omega ratio'),
                    ('Risk-adjusted return based on Lower Partial Moments', 'Sortino ratio'),
                    ('Risk-adjusted return based on Lower Partial Moments', 'Kappa three ratio'),
                    ('Risk-adjusted return based on Lower Partial Moments', 'Gain loss ratio'),
                    ('Risk-adjusted return based on Lower Partial Moments', 'Upside potential ratio'),
                    # Risk-adjusted return based on Drawdown
                    ('Risk-adjusted return based on Drawdown', 'Calmar ratio')]

        fmt_pct = tf.FmtPercent(1, rows=pct_rows, apply_to_header_and_index=False)
        fmt_dec = tf.FmtDecimals(2, rows=dec_rows, apply_to_header_and_index=False)
        fmt_align = tf.FmtAlignTable("left")
        fmt_background = tf.FmtStripeBackground(first_color=tf.colors.LIGHT_GREY, second_color=tf.colors.WHITE, header_color=tf.colors.BLACK)
        fmt_multiindex = tf.FmtExpandMultiIndex(operator=tf.OP_NONE)

        perf_data = Block(self.perf_data, formatters=[fmt_multiindex, fmt_pct, fmt_dec, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        perf_data = {'performance_table': perf_data}
        return perf_data

    def get_weights_html(self):
        fmt_pct = tf.FmtPercent(1, apply_to_header_and_index=False)
        fmt_align = tf.FmtAlignTable("left")
        fmt_background = tf.FmtStripeBackground(first_color=tf.colors.LIGHT_GREY, second_color=tf.colors.WHITE, header_color=tf.colors.BLACK)

        targetweights = Block(self.targetweights, formatters=[fmt_pct, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        targetweights = {'targetweights_table': targetweights}
        effectiveweights = Block(self.effectiveweights, formatters=[fmt_pct, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        effectiveweights = {'effectiveweights_table': effectiveweights}
        return targetweights, effectiveweights

    def generate_html(self):
        """ Returns parsed HTML text string for report
        """
        basedir = os.path.abspath(os.path.dirname(__file__))
        images = os.path.join(basedir, 'templates')
        eq_curve = os.path.join(images, 'equity_curve_multistrat.png')
        fig_equity = self.plot_equity_curve()
        fig_equity.savefig(eq_curve)
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("templates/template_multistrat.html")
        header = self.get_header_data()
        if self.system == 'windows':
            graphics = {'url_equity_curve_multistrat': 'file:\\' + eq_curve}
        else:
            graphics = {'url_equity_curve_multistrat': 'file://' + eq_curve}

        kpis = self.get_performance_stats_html()
        targetweights, effectiveweights = self.get_weights_html()
        all_numbers = {**header, **graphics, **kpis, **targetweights, **effectiveweights}
        html_out = template.render(all_numbers)
        return html_out

    def generate_pdf_report(self):
        """ Returns PDF report with backtest results
        """
        html = self.generate_html()
        HTML(string=html).write_pdf(self.outfile)
        msg = "See {} for report with backtest results of different strategies."
        print(msg.format(self.outfile))

    def get_header_data(self):
        """ Return dict with data for report header
        """
        header = {'strategy_names': self.get_strategy_names(),
                  'file_name': self.outfilename,
                  'start_date': self.get_start_date(),
                  'end_date': self.get_end_date(),
                  'name_user': self.user,
                  'processing_date': get_now(),
                  'memo_field': self.memo
                  }
        return header

    def report(self):
        self.generate_csv()
        self.generate_pdf_report()


