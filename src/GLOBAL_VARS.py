#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
# This file contains a centralized list of global variables used all over the
# project.
#
# Federico Garzarelli
###############################################################################

# main.py
# Set the strategy parameters
"""
strat_params_days = {
    'reb_days': 30,  # every month: we rebalance the portfolio
    'lookback_period_short': 30,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 180,  # period to calculate the correlation (Minimum 2)
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

"""
# TODO: change after testing
strat_params_days = {
    'reb_days': 1,  # rebalance the portfolio every year
    'lookback_period_short': 2,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 2,  # period to calculate the correlation (Minimum 2)
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

strat_params_years = {
    'reb_days': 1,  # rebalance the portfolio every year
    'lookback_period_short': 2,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 2,  # period to calculate the correlation (Minimum 2)
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

# report.py
DAYS_IN_YEAR = 260 # 365.2422

report_params = {
    'outfilename': "Aggregated_Report.pdf",
    'user': "Fabio & Federico",
    'memo': "Testing - Report comparing different strategies",
    'riskfree': 0.01,  # Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc
    'targetrate': 0.01,
    # target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio
    'alpha': 0.05,
    # confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio)
    'market_mu': 0.07,  # avg return of the market, to be used in Treynor ratio, Information ratio
    'market_sigma': 0.15,  # std dev of the market, to be used in Treynor ratio, Information ratio
    'fundmode': True,  # Calculate metrics in fund model vs asset mode
    'stddev_sample': True,  # Bessel correction (N-1) when calculating standard deviation from a sample
    'logreturns': True,  # Use logreturns instead of percentage returns when calculating metrics (recommended)
    'annualize': True  # calculate annualized metrics by annualizing returns first

}
