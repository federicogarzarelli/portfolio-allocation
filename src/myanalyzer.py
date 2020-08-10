# !/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Federico Garzarelli
#
#  Change to the analyzer class to output the date in the same format regardless of
#  the timeframe.
#  This is achived by defining a MyTimeFrameAnalyzerBase, child of TimeFrameAnalyzerBase
#  where the function _get_dt_cmpkey is overriden.
###############################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import calendar
import datetime
from backtrader import TimeFrame, TimeFrameAnalyzerBase

class MyTimeFrameAnalyzerBase(TimeFrameAnalyzerBase):
    def _get_dt_cmpkey(self, dt):
        if self.timeframe == TimeFrame.NoTimeFrame:
            return None, None

        if self.timeframe == TimeFrame.Years:
            dtcmp = dt.year
            dtkey = datetime.datetime(dt.year, 12, 31)  # dtkey = datetime.date(dt.year, 12, 31)

        elif self.timeframe == TimeFrame.Months:
            dtcmp = dt.year * 100 + dt.month
            _, lastday = calendar.monthrange(dt.year, dt.month)
            dtkey = datetime.datetime(dt.year, dt.month, lastday)

        elif self.timeframe == TimeFrame.Weeks:
            isoyear, isoweek, isoweekday = dt.isocalendar()
            dtcmp = isoyear * 100 + isoweek
            sunday = dt + datetime.timedelta(days=7 - isoweekday)
            dtkey = datetime.datetime(sunday.year, sunday.month, sunday.day)

        elif self.timeframe == TimeFrame.Days:
            dtcmp = dt.year * 10000 + dt.month * 100 + dt.day
            dtkey = datetime.datetime(dt.year, dt.month, dt.day)

        else:
            dtcmp, dtkey = self._get_subday_cmpkey(dt)

        return dtcmp, dtkey
