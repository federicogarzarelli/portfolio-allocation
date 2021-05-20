import sys
import matplotlib.pyplot as plt
import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from utils import get_now, dir_exists
from strategies import *
import pybloqs.block.table_formatters as tf
from pybloqs import Block
import GLOBAL_VARS as GLOBAL_VARS
from datetime import datetime

class ReportAggregator:
    """ Aggregates one-strategy reports and creates a multi-strategy report
    """

    def __init__(self, outfilename, outputdir, user, memo, leverage, system, InputList):
        self.outfilename = outfilename
        self.outputdir = outputdir
        self.user = user
        self.memo = memo
        self.leverage = leverage
        self.check_and_assign_defaults()
        self.system = system

        self.InputList = []
        for i in range(0,len(InputList)):
            self.InputList.append(InputList[i])

        self.outfile = os.path.join(self.outputdir, self.outfilename)

    """
    Report
    """
    def generate_csv(self):
        outputfilename = ["/Fund_Prices_", "/Returns_", "/PerformanceMetrics_", "/Target_Weights_",
                          "/Effective_Weights_", "/Portfolio_Drawdown_", "/Asset_Prices_", "/Assets_drawdown_"]

        # Output into CSV
        for i in range(0, len(self.InputList)-1): # -1 is to exclude the parameters output
            self.InputList[i].to_csv(self.outputdir + outputfilename[i] + get_now().replace(':', '') +'.csv')

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
        curve = self.InputList[0]
        xrnge = [curve.index[0], curve.index[-1]]
        dotted = pd.Series(data=[curve.iloc[0].values[0], curve.iloc[0].values[0]], index=xrnge)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Value')
        ax.set_xlabel("Date")
        ax.set_title('Portfolio value')
        _ = curve.plot(kind='line', ax=ax)
        _ = dotted.plot(kind='line', ax=ax, color='grey', linestyle=':')
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        return fig

    def plot_equity_dd(self):
        """ Plots equity curve to png file
        """
        curve = self.InputList[5]
        xrnge = [curve.index[0], curve.index[-1]]
        dotted = pd.Series(data=[curve.iloc[0].values[0], curve.iloc[0].values[0]], index=xrnge)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('% Drawdown')
        ax.set_xlabel("Date")
        ax.set_title('Portfolio drawdown')
        _ = curve.plot(kind='line', ax=ax)
        _ = dotted.plot(kind='line', ax=ax, color='grey', linestyle=':')
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        return fig

    def plot_asset_prices(self):
        """ Plots assets' relative prices to png file
        """
        curve = self.InputList[6]
        xrnge = [curve.index[0], curve.index[-1]]
        dotted = pd.Series(data=[curve.iloc[0].values[0], curve.iloc[0].values[0]], index=xrnge)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Assets value')
        ax.set_xlabel("Date")
        ax.set_title('Assets values (normalized to 1)')
        _ = curve.plot(kind='line', ax=ax)
        _ = dotted.plot(kind='line', ax=ax, color='grey', linestyle=':')
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        return fig

    def plot_asset_dd(self):
        """ Plots assets' relative prices to png file
        """
        curve = self.InputList[7]
        xrnge = [curve.index[0], curve.index[-1]]
        dotted = pd.Series(data=[curve.iloc[0].values[0], curve.iloc[0].values[0]], index=xrnge)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Assets % Drawdown')
        ax.set_xlabel("Date")
        ax.set_title('Assets drawdown')
        _ = curve.plot(kind='line', ax=ax)
        _ = dotted.plot(kind='line', ax=ax, color='grey', linestyle=':')
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        return fig

    def get_report_params(self):
        report_params = {
            'fundmode': GLOBAL_VARS.params['fundmode'],  # Calculate metrics in fund model vs asset mode
            'alpha': GLOBAL_VARS.params['alpha'],
            # confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio)
            'annualize': GLOBAL_VARS.params['annualize'],  # calculate annualized metrics by annualizing returns first
            'riskfree': GLOBAL_VARS.params['riskfree'],
            # Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc
            'targetrate': GLOBAL_VARS.params['targetrate'],
            # target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio
            'market_mu': GLOBAL_VARS.params['market_mu'],
            # avg return of the market, to be used in Treynor ratio, Information ratio
            'market_sigma': GLOBAL_VARS.params['market_sigma'],
            # std dev of the market, to be used in Treynor ratio, Information ratio
            'stddev_sample': GLOBAL_VARS.params['stddev_sample'],
            # Bessel correction (N-1) when calculating standard deviation from a sample
            'logreturns': GLOBAL_VARS.params['logreturns']
            # Use logreturns instead of percentage returns when calculating metrics (not recommended)
        }
        return report_params

    def get_strategy_names(self):
        return self.InputList[2].columns.to_list() # InputList[2] = Performance data

    def get_strategy_params(self):
        return self.InputList[8] # InputList[8] = Parameters

    def get_start_date(self):
        return self.InputList[0].index[0].strftime("%Y-%m-%d") # InputList[0] = prices

    def get_end_date(self):
        return self.InputList[0].index[-1].strftime("%Y-%m-%d")

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

        perf_data = Block(self.InputList[2], formatters=[fmt_multiindex, fmt_pct, fmt_dec, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        perf_data = {'performance_table': perf_data}
        return perf_data

    def get_weights_html(self):
        fmt_pct = tf.FmtPercent(1, apply_to_header_and_index=False)
        fmt_align = tf.FmtAlignTable("left")
        fmt_background = tf.FmtStripeBackground(first_color=tf.colors.LIGHT_GREY, second_color=tf.colors.WHITE, header_color=tf.colors.BLACK)

        targetweights = Block(self.InputList[3], formatters=[fmt_pct, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        targetweights = {'targetweights_table': targetweights}
        effectiveweights = Block(self.InputList[4], formatters=[fmt_pct, fmt_align, fmt_background], use_default_formatters=False)._repr_html_()
        effectiveweights = {'effectiveweights_table': effectiveweights}
        return targetweights, effectiveweights

    def generate_html(self):
        """ Returns parsed HTML text string for report
        """
        basedir = os.path.abspath(os.path.dirname(__file__))
        images = os.path.join(basedir, 'templates')
        eq_curve = os.path.join(images, 'equity_curve.png')
        assets_curve = os.path.join(images, 'assetprices_curve.png')
        eq_dd_curve = os.path.join(images, 'equitydrawdown_curve.png')
        assets_dd_curve = os.path.join(images, 'assetdrawdown_curve.png')
        fig_equity = self.plot_equity_curve()
        fig_assets = self.plot_asset_prices()
        fig_equity_dd = self.plot_equity_dd()
        fig_assets_dd = self.plot_asset_dd()

        fig_equity.savefig(eq_curve)
        fig_assets.savefig(assets_curve)
        fig_equity_dd.savefig(eq_dd_curve)
        fig_assets_dd.savefig(assets_dd_curve)

        env = Environment(loader=FileSystemLoader(images))
        template = env.get_template("template.html")
        header = self.get_header_data()
        if self.system == 'Windows':
            graphics = {'url_equity_curve': 'file:\\' + eq_curve,
                        'url_assetprices': 'file:\\' + assets_curve,
                        'url_eq_dd_curve': 'file:\\' + eq_dd_curve,
                        'url_assets_dd_curve': 'file:\\' + assets_dd_curve}
        else:
            graphics = {'url_equity_curve': 'file://' + eq_curve,
                        'url_assetprices': 'file://' + assets_curve,
                        'url_eq_dd_curve': 'file://' + eq_dd_curve,
                        'url_assets_dd_curve': 'file://' + assets_dd_curve}

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
                  'params': self.get_strategy_params(),
                  'start_date': self.get_start_date(),
                  'end_date': self.get_end_date(),
                  'name_user': self.user,
                  'processing_date': get_now(),
                  'memo_field': self.memo,
                  'leverage': self.leverage,
                  'report_params': self.get_report_params()
                  }

        return header

    def report(self):
        self.generate_csv()
        self.generate_pdf_report()


