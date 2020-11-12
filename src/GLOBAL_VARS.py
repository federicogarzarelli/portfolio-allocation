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

assetclass_dict = {
    # "medium" term, daily time series
    'GLD': 'gold',
    'COM': 'commodity',
    'SP500': 'equity',
    'SP500TR': 'equity',
    'LTB': 'bond_lt',
    'ITB': 'bond_it',
    'TIP': 'bond_lt',  # can also be classified as commodities due to their inflation hedge
    # "long" term, annual time series
    'GLD_LNG': 'gold',
    'OIL_LNG': 'commodity',
    'EQ_LNG': 'equity',
    'LTB_LNG': 'bond_lt',
    'ITB_LNG': 'bond_it',
    'RE_LNG': 'commodity',  # can also be classified as equities
    'US10YB_LNG': 'bond_it'
}

"""
strat_params_days = {
    'reb_days': 1,  # every month: we rebalance the portfolio
    'lookback_period_short': 2,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 2,  # period to calculate the correlation (Minimum 2)
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}
"""
strat_params_days = {
    'reb_days': 30,  # every month: we rebalance the portfolio
    'lookback_period_short': 20,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 120,  # period to calculate the correlation (Minimum 2)
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

strat_params_years = {
    'reb_days': 1,  # rebalance the portfolio every year
    'lookback_period_short': 5,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 10,  # period to calculate the correlation (Minimum 2)
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spea    rman' # method for the calculation of the correlation matrix
}

strat_params_weights = {
    'reb_days': 30,  #  rebalance the portfolio every month
    'lookback_period_short': 2,  # set the minimum period to the minimum possible value (i.e. 2), when weights are provided
    'lookback_period_long': 2,
    'printlog': True,  # Print log in the console
    'corrmethod': 'pearson'  # 'spea    rman' # method for the calculation of the correlation matrix
}

# report.py
DAYS_IN_YEAR = 260  # 365.2422

report_params = {
    'riskfree': 0.01,  # Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc
    'targetrate': 0.01,
    # target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio
    'alpha': 0.05,
    # confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio)
    'market_mu': 0.07,  # avg return of the market, to be used in Treynor ratio, Information ratio
    'market_sigma': 0.15,  # std dev of the market, to be used in Treynor ratio, Information ratio
    'fundmode': True,  # Calculate metrics in fund model vs asset mode
    'stddev_sample': True,  # Bessel correction (N-1) when calculating standard deviation from a sample
    'logreturns': False,  # Use logreturns instead of percentage returns when calculating metrics (not recommended)
    'annualize': True  # calculate annualized metrics by annualizing returns first

}
