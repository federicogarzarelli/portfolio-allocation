#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
# This file contains a centralized list of global variables used all over the
# project.
#
# Federico Garzarelli
###############################################################################
from datetime import date, timedelta, datetime
import os


# parameters used in main.py
# Set the strategy parameters

APPLY_LEVERAGE_ON_LIVE_STOCKS = False

# DB parameters
DEFAULT_DATE = str(date.today())+ " 00:00:00"
DEFAULT_STARTDATE = "1920-01-01 00:00:00" #"1975-01-01 00:00:00"

DB_NAME = 'myPortfolio.db'

params = {'startdate': '',
          'enddate': '',
          'initial_cash': '',
          'contribution': '',
          'leverage': '',
          'expense_ratio': '',
          'historic': '',
          'shares': '',
          'shareclass': '',
          'weights': '',
          'benchmark': '',
          'indicator': '',
          'riskparity': '',
          'riskparity_nested': '',
          'rotationstrat': '',
          'uniform': '',
          'vanillariskparity': '',
          'onlystocks': '',
          'sixtyforty': '',
          'trend_u': '',
          'absmom_u': '',
          'relmom_u': '',
          'momtrend_u': '',
          'trend_rp': '',
          'absmom_rp': '',
          'relmom_rp': '',
          'momtrend_rp': '',
          'GEM': '',
          'create_report': '',
          'report_name': '',
          'user': '',
          'memo': '',
          # advanced params
          'DAYS_IN_YEAR': '',
          'DAYS_IN_YEAR_BOND_PRICE': '',
          'reb_days_days': '',
          'reb_days_years': '',
          'reb_days_custweights': '',
          'lookback_period_short_days': '',
          'lookback_period_short_years': '',
          'lookback_period_short_custweights': '',
          'lookback_period_long_days': '',
          'lookback_period_long_years': '',
          'lookback_period_long_custweights': '',
          'moving_average_period_days': '',
          'moving_average_period_years': '',
          'moving_average_period_custweights': '',
          'momentum_period_days': '',
          'momentum_period_years': '',
          'momentum_period_custweights': '',
          'momentum_percentile_days': '',
          'momentum_percentile_years': '',
          'momentum_percentile_custweights': '',
          'corrmethod_days': '',
          'corrmethod_years': '',
          'corrmethod_custweights': '',
          'riskfree': '',
          'targetrate': '',
          'alpha': '',
          'market_mu': '',
          'market_sigma': '',
          'stddev_sample': '',
          'annualize': '',
          'logreturns': ''
          }
