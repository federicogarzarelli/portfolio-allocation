import backtrader as bt

class WeightsObserver(bt.observer.Observer):
    params = (('n_assets', 100),) # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_'+str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        for asset in range(0, self.params.n_assets):
            self.lines[asset][0] = self._owner.weights[asset]


class GetDate(bt.observer.Observer):
    lines = ('year','month', 'day',)

    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.year[0] = self._owner.datas[0].datetime.date(0).year
        self.lines.month[0] = self._owner.datas[0].datetime.date(0).month
        self.lines.day[0] = self._owner.datas[0].datetime.date(0).day