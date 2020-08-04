#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2020 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict

from backtrader.utils.py3 import range
from backtrader import Analyzer
from backtrader import TimeFrameAnalyzerBase
from backtrader.utils import AutoOrderedDict


import math

from backtrader.utils.py3 import itervalues

from backtrader import Analyzer, TimeFrame
from backtrader.mathsupport import average, standarddev

import backtrader as bt

class MyAnnualReturn(Analyzer):
    '''
    This analyzer calculates the AnnualReturns by looking at the beginning
    and end of the year
    Params:
      - (None)
    Member Attributes:
      - ``rets``: list of calculated annual returns
      - ``ret``: dictionary (key: year) of annual returns
    **get_analysis**:
      - Returns a dictionary of annual returns (key: year)
    '''

    def stop(self):
        # Must have stats.broker
        cur_year = -1

        value_start = 0.0
        value_cur = 0.0
        value_end = 0.0

        self.rets = list()
        self.ret = OrderedDict()

        for i in range(len(self.data) - 1, -1, -1):
            dt = self.data.datetime.date(-i)
            value_cur = self.strategy.stats.broker.value[-i]
            if dt >= self.strategy.startdate.date():
                if dt.year > cur_year:
                    if cur_year >= 0:
                        annualret = (value_end / value_start) - 1.0
                        self.rets.append(annualret)
                        self.ret[cur_year] = annualret

                        # changing between real years, use last value as new start
                        value_start = value_end
                    else:
                        # No value set whatsoever, use the currently loaded value
                        value_start = value_cur

                    cur_year = dt.year

                # No matter what, the last value is always the last loaded value
                value_end = value_cur

        if cur_year not in self.ret:
            # finish calculating pending data
            annualret = (value_end / value_start) - 1.0
            self.rets.append(annualret)
            self.ret[cur_year] = annualret

        # eliminate first year. The return in that year is set to zero by def. in the code
        del self.ret[self.strategy.startdate.date().year]

    def get_analysis(self):
        return self.ret

class MyTimeReturn(TimeFrameAnalyzerBase):
    '''This analyzer calculates the Returns by looking at the beginning
    and end of the timeframe
    Params:
      - ``timeframe`` (default: ``None``)
        If ``None`` the ``timeframe`` of the 1st data in the system will be
        used
        Pass ``TimeFrame.NoTimeFrame`` to consider the entire dataset with no
        time constraints
      - ``compression`` (default: ``None``)
        Only used for sub-day timeframes to for example work on an hourly
        timeframe by specifying "TimeFrame.Minutes" and 60 as compression
        If ``None`` then the compression of the 1st data of the system will be
        used
      - ``data`` (default: ``None``)
        Reference asset to track instead of the portfolio value.
        .. note:: this data must have been added to a ``cerebro`` instance with
                  ``addata``, ``resampledata`` or ``replaydata``
      - ``firstopen`` (default: ``True``)
        When tracking the returns of a ``data`` the following is done when
        crossing a timeframe boundary, for example ``Years``:
          - Last ``close`` of previous year is used as the reference price to
            see the return in the current year
        The problem is the 1st calculation, because the data has** no
        previous** closing price. As such and when this parameter is ``True``
        the *opening* price will be used for the 1st calculation.
        This requires the data feed to have an ``open`` price (for ``close``
        the standard [0] notation will be used without reference to a field
        price)
        Else the initial close will be used.
      - ``fund`` (default: ``None``)
        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation
        Set it to ``True`` or ``False`` for a specific behavior
    Methods:
      - get_analysis
        Returns a dictionary with returns as values and the datetime points for
        each return as keys
    '''

    params = (
        ('data', None),
        ('firstopen', True),
        ('fund', None),
    )

    def start(self):
        super(MyTimeReturn, self).start()
        if self.p.fund is None:
            self._fundmode = self.strategy.broker.fundmode
        else:
            self._fundmode = self.p.fund

        self._value_start = 0.0
        self._lastvalue = None
        self._deletedFirstVal = None
        if self.p.data is None:
            # keep the initial portfolio value if not tracing a data
            if not self._fundmode:
                self._lastvalue = self.strategy.broker.getvalue()
            else:
                self._lastvalue = self.strategy.broker.fundvalue

    def notify_fund(self, cash, value, fundvalue, shares):
        if not self._fundmode:
            # Record current value
            if self.p.data is None:
                self._value = value  # the portofolio value if tracking no data
            else:
                self._value = self.p.data[0]  # the data value if tracking data
        else:
            if self.p.data is None:
                self._value = fundvalue  # the fund value if tracking no data
            else:
                self._value = self.p.data[0]  # the data value if tracking data

    def on_dt_over(self):
        # next is called in a new timeframe period
        # if self.p.data is None or len(self.p.data) > 1:
        if self.p.data is None or self._lastvalue is not None:
            self._value_start = self._lastvalue  # update value_start to last

        else:
            # The 1st tick has no previous reference, use the opening price
            if self.p.firstopen:
                self._value_start = self.p.data.open[0]
            else:
                self._value_start = self.p.data[0]

    def next(self):
        # Calculate the return
        super(MyTimeReturn, self).next()
        if self.strategy.startdate is not None and self.dtkey >= self.strategy.startdate:
            self.rets[self.dtkey] = (self._value / self._value_start) - 1.0
            self._lastvalue = self._value  # keep last value

    def get_analysis(self):
        # eliminate first return. The return in that year is set to zero by def. in the code
        if self._deletedFirstVal is None:
            del self.rets[list(self.rets.keys())[0]]
            self._deletedFirstVal = True
        return self.rets


class MySharpeRatio(Analyzer):
    '''This analyzer calculates the SharpeRatio of a strategy using a risk free
    asset which is simply an interest rate
    See also:
      - https://en.wikipedia.org/wiki/Sharpe_ratio
    Params:
      - ``timeframe``: (default: ``TimeFrame.Years``)
      - ``compression`` (default: ``1``)
        Only used for sub-day timeframes to for example work on an hourly
        timeframe by specifying "TimeFrame.Minutes" and 60 as compression
      - ``riskfreerate`` (default: 0.01 -> 1%)
        Expressed in annual terms (see ``convertrate`` below)
      - ``convertrate`` (default: ``True``)
        Convert the ``riskfreerate`` from annual to monthly, weekly or daily
        rate. Sub-day conversions are not supported
      - ``factor`` (default: ``None``)
        If ``None``, the conversion factor for the riskfree rate from *annual*
        to the chosen timeframe will be chosen from a predefined table
          Days: 252, Weeks: 52, Months: 12, Years: 1
        Else the specified value will be used
      - ``annualize`` (default: ``False``)
        If ``convertrate`` is ``True``, the *SharpeRatio* will be delivered in
        the ``timeframe`` of choice.
        In most occasions the SharpeRatio is delivered in annualized form.
        Convert the ``riskfreerate`` from annual to monthly, weekly or daily
        rate. Sub-day conversions are not supported
      - ``stddev_sample`` (default: ``False``)
        If this is set to ``True`` the *standard deviation* will be calculated
        decreasing the denominator in the mean by ``1``. This is used when
        calculating the *standard deviation* if it's considered that not all
        samples are used for the calculation. This is known as the *Bessels'
        correction*
      - ``daysfactor`` (default: ``None``)
        Old naming for ``factor``. If set to anything else than ``None`` and
        the ``timeframe`` is ``TimeFrame.Days`` it will be assumed this is old
        code and the value will be used
      - ``legacyannual`` (default: ``False``)
        Use the ``AnnualReturn`` return analyzer, which as the name implies
        only works on years
      - ``fund`` (default: ``None``)
        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation
        Set it to ``True`` or ``False`` for a specific behavior
    Methods:
      - get_analysis
        Returns a dictionary with key "sharperatio" holding the ratio
    '''
    params = (
        ('timeframe', TimeFrame.Years),
        ('compression', 1),
        ('riskfreerate', 0.01),
        ('factor', None),
        ('convertrate', True),
        ('annualize', False),
        ('stddev_sample', False),

        # old behavior
        ('daysfactor', None),
        ('legacyannual', False),
        ('fund', None),
    )

    RATEFACTORS = {
        TimeFrame.Days: 365.2422, #TimeFrame.Days: 252,
        TimeFrame.Weeks: 52,
        TimeFrame.Months: 12,
        TimeFrame.Years: 1,
    }

    def __init__(self):
        if self.p.legacyannual:
            self.anret = MyAnnualReturn()
        else:
            self.timereturn = MyTimeReturn(
                timeframe=self.p.timeframe,
                compression=self.p.compression,
                fund=self.p.fund)

    def stop(self):
        super(MySharpeRatio, self).stop()
        if self.p.legacyannual:
            rate = self.p.riskfreerate
            retavg = average([r - rate for r in self.anret.rets])
            retdev = standarddev(self.anret.rets)

            self.ratio = retavg / retdev
        else:
            # Get the returns from the subanalyzer
            returns = list(itervalues(self.timereturn.get_analysis()))

            rate = self.p.riskfreerate  #

            factor = None

            # Hack to identify old code
            if self.p.timeframe == TimeFrame.Days and \
               self.p.daysfactor is not None:

                factor = self.p.daysfactor

            else:
                if self.p.factor is not None:
                    factor = self.p.factor  # user specified factor
                elif self.p.timeframe in self.RATEFACTORS:
                    # Get the conversion factor from the default table
                    factor = self.RATEFACTORS[self.p.timeframe]

            if factor is not None:
                # A factor was found

                if self.p.convertrate:
                    # Standard: downgrade annual returns to timeframe factor
                    rate = pow(1.0 + rate, 1.0 / factor) - 1.0
                else:
                    # Else upgrade returns to yearly returns
                    returns = [pow(1.0 + x, factor) - 1.0 for x in returns]

            lrets = len(returns) - self.p.stddev_sample
            # Check if the ratio can be calculated
            if lrets:
                # Get the excess returns - arithmetic mean - original sharpe
                ret_free = [r - rate for r in returns]
                ret_free_avg = average(ret_free)
                retdev = standarddev(ret_free, avgx=ret_free_avg,
                                     bessel=self.p.stddev_sample)

                try:
                    ratio = ret_free_avg / retdev

                    if factor is not None and \
                       self.p.convertrate and self.p.annualize:

                        ratio = math.sqrt(factor) * ratio
                except (ValueError, TypeError, ZeroDivisionError):
                    ratio = None
            else:
                # no returns or stddev_sample was active and 1 return
                ratio = None

            self.ratio = ratio

        self.rets['sharperatio'] = self.ratio

class MySharpeRatio_A(MySharpeRatio):
    '''Extension of the SharpeRatio which returns the Sharpe Ratio directly in
    annualized form
    The following param has been changed from ``SharpeRatio``
      - ``annualize`` (default: ``True``)
    '''

    params = (
        ('annualize', True),
    )

class MyReturns(TimeFrameAnalyzerBase):
    '''Total, Average, Compound and Annualized Returns calculated using a
    logarithmic approach
    See:
      - https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
    Params:
      - ``timeframe`` (default: ``None``)
        If ``None`` the ``timeframe`` of the 1st data in the system will be
        used
        Pass ``TimeFrame.NoTimeFrame`` to consider the entire dataset with no
        time constraints
      - ``compression`` (default: ``None``)
        Only used for sub-day timeframes to for example work on an hourly
        timeframe by specifying "TimeFrame.Minutes" and 60 as compression
        If ``None`` then the compression of the 1st data of the system will be
        used
      - ``tann`` (default: ``None``)
        Number of periods to use for the annualization (normalization) of the
        namely:
          - ``days: 252``
          - ``weeks: 52``
          - ``months: 12``
          - ``years: 1``
      - ``fund`` (default: ``None``)
        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation
        Set it to ``True`` or ``False`` for a specific behavior
    Methods:
      - get_analysis
        Returns a dictionary with returns as values and the datetime points for
        each return as keys
        The returned dict the following keys:
          - ``rtot``: Total compound return
          - ``ravg``: Average return for the entire period (timeframe specific)
          - ``rnorm``: Annualized/Normalized return
          - ``rnorm100``: Annualized/Normalized return expressed in 100%
    '''

    params = (
        ('tann', None),
        ('fund', None),
    )

    _TANN = {
        bt.TimeFrame.Days: 365.2422, #bt.TimeFrame.Days: 252.0,
        bt.TimeFrame.Weeks: 52.0,
        bt.TimeFrame.Months: 12.0,
        bt.TimeFrame.Years: 1.0,
    }

    def start(self):
        super(MyReturns, self).start()
        if self.p.fund is None:
            self._fundmode = self.strategy.broker.fundmode
        else:
            self._fundmode = self.p.fund

        if not self._fundmode:
            self._value_start = self.strategy.broker.getvalue()
        else:
            self._value_start = self.strategy.broker.fundvalue

        self._tcount = 0

    def stop(self):
        super(MyReturns, self).stop()

        if not self._fundmode:
            self._value_end = self.strategy.broker.getvalue()
        else:
            self._value_end = self.strategy.broker.fundvalue

        # Compound return
        try:
            nlrtot = self._value_end / self._value_start
        except ZeroDivisionError:
            rtot = float('-inf')
        else:
            if nlrtot < 0.0:
                rtot = float('-inf')
            else:
                rtot = math.log(nlrtot)

        self.rets['rtot'] = rtot

        # Average return
        self.rets['ravg'] = ravg = rtot / (self._tcount-1)

        # Annualized normalized return
        tann = self.p.tann or self._TANN.get(self.timeframe, None)
        if tann is None:
            tann = self._TANN.get(self.data._timeframe, 1.0)  # assign default

        if ravg > float('-inf'):
                self.rets['rnorm'] = rnorm = math.expm1(ravg * tann)
        else:
            self.rets['rnorm'] = rnorm = ravg

        self.rets['rnorm100'] = rnorm * 100.0  # human readable %

    def _on_dt_over(self):
        if self.strategy.startdate is not None and self.dtkey >= self.strategy.startdate:
            self._tcount += 1  # count the subperiod


__all__ = ['DrawDown', 'TimeDrawDown']


class MyDrawDown(bt.Analyzer):
    '''This analyzer calculates trading system drawdowns stats such as drawdown
    values in %s and in dollars, max drawdown in %s and in dollars, drawdown
    length and drawdown max length
    Params:
      - ``fund`` (default: ``None``)
        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation
        Set it to ``True`` or ``False`` for a specific behavior
    Methods:
      - ``get_analysis``
        Returns a dictionary (with . notation support and subdctionaries) with
        drawdown stats as values, the following keys/attributes are available:
        - ``drawdown`` - drawdown value in 0.xx %
        - ``moneydown`` - drawdown value in monetary units
        - ``len`` - drawdown length
        - ``max.drawdown`` - max drawdown value in 0.xx %
        - ``max.moneydown`` - max drawdown value in monetary units
        - ``max.len`` - max drawdown length
    '''

    params = (
        ('fund', None),
    )

    def start(self):
        super(MyDrawDown, self).start()
        if self.p.fund is None:
            self._fundmode = self.strategy.broker.fundmode
        else:
            self._fundmode = self.p.fund

    def create_analysis(self):
        self.rets = AutoOrderedDict()  # dict with . notation

        self.rets.len = 0
        self.rets.drawdown = 0.0
        self.rets.moneydown = 0.0

        self.rets.max.len = 0.0
        self.rets.max.drawdown = 0.0
        self.rets.max.moneydown = 0.0

        self._maxvalue = float('-inf')  # any value will outdo it

    def stop(self):
        self.rets._close()  # . notation cannot create more keys

    def notify_fund(self, cash, value, fundvalue, shares):
        if not self._fundmode:
            self._value = value  # record current value
            self._maxvalue = max(self._maxvalue, value)  # update peak value
        else:
            self._value = fundvalue  # record current value
            self._maxvalue = max(self._maxvalue, fundvalue)  # update peak

    def next(self):
        r = self.rets

        # calculate current drawdown values
        r.moneydown = moneydown = self._maxvalue - self._value
        r.drawdown = drawdown = 100.0 * moneydown / self._maxvalue

        # maxximum drawdown values
        r.max.moneydown = max(r.max.moneydown, moneydown)
        r.max.drawdown = maxdrawdown = max(r.max.drawdown, drawdown)

        r.len = r.len + 1 if drawdown else 0
        r.max.len = max(r.max.len, r.len)


class MyTimeDrawDown(bt.TimeFrameAnalyzerBase):
    '''This analyzer calculates trading system drawdowns on the chosen
    timeframe which can be different from the one used in the underlying data
    Params:
      - ``timeframe`` (default: ``None``)
        If ``None`` the ``timeframe`` of the 1st data in the system will be
        used
        Pass ``TimeFrame.NoTimeFrame`` to consider the entire dataset with no
        time constraints
      - ``compression`` (default: ``None``)
        Only used for sub-day timeframes to for example work on an hourly
        timeframe by specifying "TimeFrame.Minutes" and 60 as compression
        If ``None`` then the compression of the 1st data of the system will be
        used
      - *None*
      - ``fund`` (default: ``None``)
        If ``None`` the actual mode of the broker (fundmode - True/False) will
        be autodetected to decide if the returns are based on the total net
        asset value or on the fund value. See ``set_fundmode`` in the broker
        documentation
        Set it to ``True`` or ``False`` for a specific behavior
    Methods:
      - ``get_analysis``
        Returns a dictionary (with . notation support and subdctionaries) with
        drawdown stats as values, the following keys/attributes are available:
        - ``drawdown`` - drawdown value in 0.xx %
        - ``maxdrawdown`` - drawdown value in monetary units
        - ``maxdrawdownperiod`` - drawdown length
      - Those are available during runs as attributes
        - ``dd``
        - ``maxdd``
        - ``maxddlen``
    '''

    params = (
        ('fund', None),
    )

    def start(self):
        super(MyTimeDrawDown, self).start()
        if self.p.fund is None:
            self._fundmode = self.strategy.broker.fundmode
        else:
            self._fundmode = self.p.fund
        self.dd = 0.0
        self.maxdd = 0.0
        self.maxddlen = 0
        self.peak = float('-inf')
        self.ddlen = 0

    def on_dt_over(self):
        if not self._fundmode:
            value = self.strategy.broker.getvalue()
        else:
            value = self.strategy.broker.fundvalue

        # update the maximum seen peak
        if value > self.peak:
            self.peak = value
            self.ddlen = 0  # start of streak

        # calculate the current drawdown
        self.dd = dd = 100.0 * (self.peak - value) / self.peak
        self.ddlen += bool(dd)  # if peak == value -> dd = 0

        # update the maxdrawdown if needed
        self.maxdd = max(self.maxdd, dd)
        self.maxddlen = max(self.maxddlen, self.ddlen)

    def stop(self):
        self.rets['maxdrawdown'] = self.maxdd
        self.rets['maxdrawdownperiod'] = self.maxddlen
