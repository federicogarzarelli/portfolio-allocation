#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
# This file contains a centralized list of global variables used all over the
# project.
#
# Federico Garzarelli
###############################################################################
from datetime import date, timedelta, datetime


# parameters used in main.py
# Set the strategy parameters

expense_ratio = 0.01
APPLY_LEVERAGE_ON_LIVE_STOCKS = False

strat_params_days = {
    'reb_days': 21,  # rebalance the portfolio every month
    'lookback_period_short': 20,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 120,  # period to calculate the correlation (Minimum 2)
    'moving_average_period': 250, # period to calculate the moving average (Minimum 2)
    'momentum_period': 250, # period to calculate the momentum returns (Minimum 2)
    'momentum_percentile': 0.5, # percentile of assets with the highest return in a period to form the relative momentum portfolio
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

strat_params_years = {
    'reb_days': 1,  # rebalance the portfolio every year
    'lookback_period_short': 5,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 10,  # period to calculate the correlation (Minimum 2)
    'moving_average_period': 2,  # period to calculate the moving average (Minimum 2)
    'momentum_period': 2,  # period to calculate the momentum returns (Minimum 2)
    'momentum_percentile': 0.75, # percentile of assets with the highest return in a period to form the relative momentum portfolio
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

strat_params_weights = {
    'reb_days': 21,  #  rebalance the portfolio every month
    'lookback_period_short': 2,  # set the minimum period to the minimum possible value (i.e. 2), when weights are provided
    'lookback_period_long': 2,
    'moving_average_period': 2,  # period to calculate the moving average (Minimum 2)
    'momentum_period': 2,  # period to calculate the momentum returns (Minimum 2)
    'momentum_percentile': 0.75, # percentile of assets with the highest return in a period to form the relative momentum portfolio.
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

#  parameters used in report.py
DAYS_IN_YEAR = 252  # 365.2422
DAYS_IN_YEAR_BOND_PRICE = 360

report_params = {
    'fundmode': True,  # Calculate metrics in fund model vs asset mode OVERRIDDEN IN MAIN -> TODO delete
    'alpha': 0.01, # confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio)
    'annualize': True,  # calculate annualized metrics by annualizing returns first
    'riskfree': 0.01,  # Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc
    'targetrate': 0.01, # target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio
    'market_mu': 0.07,  # avg return of the market, to be used in Treynor ratio, Information ratio
    'market_sigma': 0.15,  # std dev of the market, to be used in Treynor ratio, Information ratio
    'stddev_sample': True,  # Bessel correction (N-1) when calculating standard deviation from a sample
    'logreturns': False  # Use logreturns instead of percentage returns when calculating metrics (not recommended)
}

# DB parameters
DEFAULT_DATE = str(date.today())+ " 00:00:00"
DEFAULT_STARTDATE = "1920-01-01 00:00:00" #"1975-01-01 00:00:00"

DB_NAME = 'myPortfolio.db'