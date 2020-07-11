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


class ReportAggregator:
    """ Aggregates one-strategy reports and creates a multi-strategy report
    """

    def __init__(self, outfilename, outputdir, user, memo, system, prices, returns, perf_data, weight):
        self.outfilename = outfilename
        self.outputdir = outputdir
        self.user = user
        self.memo = memo
        self.check_and_assign_defaults()
        self.system = system
        self.prices = prices
        self.returns = returns 
        self.perf_data = perf_data
        self.weight = weight

        self.outfile = os.path.join(self.outputdir, self.outfilename)


    """
    Multistrategy report
    """
    def generate_csv(self):
        # Output into CSV for now, later create a PDF from a HTML
        self.prices.to_csv(self.outputdir+r'Prices_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.returns.to_csv(self.outputdir+r'/Returns_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.perf_data.to_csv(self.outputdir+r'/Performance_MultiStrategy_' + get_now().replace(':', '') +'.csv')
        self.weight.to_csv(self.outputdir+r'/Weights_MultiStrategy_' + get_now().replace(':', '') +'.csv')

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

    def get_performance_stats(self):

        formatter = {}
        for i in self.perf_data.columns:
            formatter[i] = '{:,.2f}'.format
        perf_data = self.perf_data.to_html(formatters=formatter, index=True,escape=False,col_space='200px')
        perf_data = {'performance_table':perf_data}
        return perf_data

    def get_weights(self):
        formatter = {}
        for i in self.weight.columns:
            formatter[i] = '{:,.1%}'.format
        weights = self.weight.to_html(formatters=formatter, index=True,escape=False,col_space='200px')
        weights = {'weights_table':weights}
        return weights

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

        kpis = self.get_performance_stats()
        weights = self.get_weights()
        all_numbers = {**header, **graphics, **kpis, **weights}
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


