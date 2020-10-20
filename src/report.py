import matplotlib.pyplot as plt
from strategies import *
from utils import timestamp2str
from GLOBAL_VARS import *
from myanalyzers import MyAnnualReturn, MyTimeReturn, MyReturns, MyDrawDown, \
    MyTimeDrawDown, MyLogReturnsRolling, MyDistributionMoments, MyRiskAdjusted_VolBased, \
    MyRiskAdjusted_VaRBased, MyRiskAdjusted_LPMBased, MyRiskAdjusted_DDBased

class PerformanceReport:
    """ Report with performance stats for given backtest run
    """
    def __init__(self, stratbt, system, timeframe):
        self.stratbt = stratbt  # works for only 1 strategy
        self.system = system
        self.timeframe = timeframe

    def get_performance_stats(self):
        """ Return dict with performance stats for given strategy withing backtest
        """
        st = self.stratbt
        dt = self.get_date_index()

        # Total period for the backtesting in days
        bt_period_days = np.busday_count(dt[0].date().isoformat(), dt[-1].date().isoformat())
        bt_period_years = bt_period_days / DAYS_IN_YEAR

        # Import analyzers results
        # Returns
        annualreturns = st.analyzers.myAnnualReturn.get_analysis()  # For total and annual returns in asset mode
        timeret = st.analyzers.myTimeReturn.get_analysis()
        logret = st.analyzers.myLogReturnsRolling.get_analysis()
        returns = st.analyzers.myReturns.get_analysis()  # For the annual return in fund mode

        # Drawdowns
        drawdown = st.analyzers.myDrawDown.get_analysis()
        timedd = st.analyzers.myTimeDrawDown.get_analysis()

        # Distribution
        ret_distrib = st.analyzers.MyDistributionMoments.get_analysis()

        # Risk-adjusted return based on Volatility
        RiskAdjusted_VolBased = st.analyzers.MyRiskAdjusted_VolBased.get_analysis()

        # Risk-adjusted return based on Value at Risk
        RiskAdjusted_VaRBased = st.analyzers.MyRiskAdjusted_VaRBased.get_analysis()

        # Risk-adjusted return based on Lower Partial Moments
        RiskAdjusted_LPMBased = st.analyzers.MyRiskAdjusted_LPMBased.get_analysis()

        # Risk-adjusted return based on Drawdown risk
        RiskAdjusted_DDBased = st.analyzers.MyRiskAdjusted_DDBased.get_analysis()

        # Calculate end value and total return (portfolio asset mode)
        endValue = st.observers.broker.lines[1].get(size=len(dt))[-1]

        tot_return = 1
        for key, value in timeret.items():
            tot_return = tot_return * (1 + value)
        tot_return = tot_return - 1

        if self.timeframe == bt.TimeFrame.Days:
            annual_return_asset = 100 * ((1 + tot_return) ** (DAYS_IN_YEAR / bt_period_days) - 1)
        elif self.timeframe == bt.TimeFrame.Years:
            annual_return_asset = 100 * ((1 + tot_return) ** (1 / bt_period_years) - 1)

        kpi = {  # PnL
            'Starting cash': self.get_startcash(),
            'End value': endValue,
            'Total return': 100 * tot_return,
            'Annual return': 100 * returns['rnorm'],
            'Annual return (asset mode)': annual_return_asset,
            'Max money drawdown': drawdown['max']['moneydown'],
            'Max percentage drawdown': drawdown['max']['drawdown'],
            # Distribution
            'Returns volatility': ret_distrib['std'],
            'Returns skewness': ret_distrib['skewness'],
            'Returns kurtosis': ret_distrib['kurtosis'],
            # Risk-adjusted return based on Volatility
            'Treynor ratio': RiskAdjusted_VolBased['treynor_ratio'],
            'Sharpe ratio': RiskAdjusted_VolBased['sharpe_ratio'],
            'Information ratio': RiskAdjusted_VolBased['information_ratio'],
            # Risk-adjusted return based on Value at Risk
            'VaR': RiskAdjusted_VaRBased['var'],
            'Expected Shortfall': RiskAdjusted_VaRBased['cvar'],
            'Excess var': RiskAdjusted_VaRBased['excess_var'],
            'Conditional sharpe ratio': RiskAdjusted_VaRBased['conditional_sharpe_ratio'],
            # Risk-adjusted return based on Lower Partial Moments
            'Omega ratio': RiskAdjusted_LPMBased['omega_ratio'],
            'Sortino ratio': RiskAdjusted_LPMBased['sortino_ratio'],
            'Kappa three ratio': RiskAdjusted_LPMBased['kappa_three_ratio'],
            'Gain loss ratio': RiskAdjusted_LPMBased['gain_loss_ratio'],
            'Upside potential ratio': RiskAdjusted_LPMBased['upside_potential_ratio'],
            # Risk-adjusted return based on Drawdown risk
            'Calmar ratio': RiskAdjusted_DDBased['calmar_ratio']
        }
        return kpi

    def get_assets(self):
        st = self.stratbt
        dt = self.get_date_index()
        n_assets = self.get_strategy_params().get('n_assets')

        dt_df = pd.DataFrame(data=dt, columns=["date"])

        for i in range(0, n_assets):
            thisasset = st.datas[i]._dataname[["close"]]
            thisasset = thisasset.rename(columns={"close": st.assets[i]._name})
            dt_df = pd.merge(left=dt_df, right=thisasset, how='left', left_on='date', right_on='date')

        dt_df = dt_df.set_index("date", drop=True)

        return dt_df.div(dt_df.iloc[0])

    def get_equity_curve(self):
        """ Return series containing equity curve
        """
        st = self.stratbt
        dt = self.get_date_index()
        value = st.observers.broker.lines[1].get(size=len(dt))
        vv = np.asarray(value)
        vv = vv[~np.isnan(vv)]

        curve = pd.Series(data=vv, index=dt)
        #return 100 * curve / curve.iloc[0]
        return curve

    def get_weights(self):
        st = self.stratbt
        n_assets = self.get_strategy_params().get('n_assets')

        # Asset targetweights
        size_weights = 100  # get targetweights for the last 100 days
        if len(self.get_date_index()) > size_weights:
            idx = self.get_date_index()[len(self.get_date_index()) - size_weights:len(self.get_date_index())]
        else:
            size_weights = len(self.get_date_index())
            idx = self.get_date_index()[0:len(self.get_date_index())]

        targetweights_df = pd.DataFrame(index=idx)
        effectiveweights_df = pd.DataFrame(index=idx)

        for i in range(0, n_assets):
            targetweights_df[st.assets[i]._name] = st.observers.targetweightsobserver.lines[i].get(size=size_weights)
            effectiveweights_df[st.assets[i]._name] = st.observers.effectiveweightsobserver.lines[i].get(
                size=size_weights)
        return targetweights_df, effectiveweights_df

    def get_strategy_name(self):
        return self.stratbt.__class__.__name__

    def get_strategy_params(self):
        return self.stratbt.cerebro.strats[0][0][-1]

    def get_date_index(self):
        st = self.stratbt
        # Get dates from the observer
        year = st.observers.getdate.lines[0].array
        month = st.observers.getdate.lines[1].array
        day = st.observers.getdate.lines[2].array

        # Put all together and drop na
        df = pd.DataFrame(data=list(zip(day.tolist(), month.tolist(), year.tolist())),
                          columns=["day", "month", "year"])
        df = df.dropna()

        # Transform into DatetimeIndex
        df = pd.to_datetime(df[["day", "month", "year"]])
        df.index = df
        df = df[df >= st.startdate]
        return df.index

    def get_start_date(self):
        """ Return first datafeed datetime
        """
        dt = self.get_date_index()
        return timestamp2str(dt[0])

    def get_end_date(self):
        """ Return first datafeed datetime
        """
        dt = self.get_date_index()
        return timestamp2str(dt[-1])

    def get_startcash(self):
        return self.stratbt.broker.startingcash

    def get_aggregated_data(self):
        kpis = self.get_performance_stats()
        kpis_df = pd.DataFrame.from_dict(kpis, orient='index')
        kpis_df.loc['Total return'] = kpis_df.loc['Total return'] / 100
        kpis_df.loc['Annual return'] = kpis_df.loc['Annual return'] / 100
        kpis_df.loc['Annual return (asset mode)'] = kpis_df.loc['Annual return (asset mode)'] / 100
        kpis_df.loc['Max percentage drawdown'] = kpis_df.loc['Max percentage drawdown'] / 100

        kpis_df['Category'] = ['P&L', 'P&L', 'P&L', 'P&L', 'P&L',
                    'Risk-adjusted return based on Drawdown', 'Risk-adjusted return based on Drawdown',
                    'Distribution moments', 'Distribution moments', 'Distribution moments',
                    'Risk-adjusted return based on Volatility', 'Risk-adjusted return based on Volatility',
                    'Risk-adjusted return based on Volatility',
                    'Risk-adjusted return based on Value at Risk', 'Risk-adjusted return based on Value at Risk',
                    'Risk-adjusted return based on Value at Risk', 'Risk-adjusted return based on Value at Risk',
                    'Risk-adjusted return based on Lower Partial Moments',
                    'Risk-adjusted return based on Lower Partial Moments',
                    'Risk-adjusted return based on Lower Partial Moments',
                    'Risk-adjusted return based on Lower Partial Moments',
                    'Risk-adjusted return based on Lower Partial Moments',
                    'Risk-adjusted return based on Drawdown']

        kpis_df['Metrics'] = kpis_df.index

        all_stats = kpis_df.set_index(['Category', 'Metrics'])

        all_stats.columns = [self.get_strategy_name()]
        return all_stats

    def output_all_data(self):
        prices = self.get_equity_curve()
        prices.index = prices.index.date
        returns = prices.diff() / prices
        assetprices = self.get_assets()

        prices = pd.DataFrame(data=prices, columns=[self.get_strategy_name()])
        returns = pd.DataFrame(data=returns, columns=[self.get_strategy_name()])
        returns = returns.dropna()

        perf_data = self.get_aggregated_data()
        targetweights, effectiveweights = self.get_weights()
        targetweights = targetweights.tail(1).T
        effectiveweights = effectiveweights.tail(1).T
        targetweights.columns = [self.get_strategy_name()]
        effectiveweights.columns = [self.get_strategy_name()]

        params = self.get_strategy_params()

        return prices, returns, perf_data, targetweights, effectiveweights, params, assetprices

class Cerebro(bt.Cerebro):
    def __init__(self, timeframe=None, **kwds):
        super().__init__(**kwds)
        self.timeframe = timeframe
        self.add_report_analyzers(riskfree=report_params['riskfree'], targetrate=report_params['targetrate'],
                                  alpha=report_params['alpha'], market_mu=report_params['market_mu'],
                                  market_sigma=report_params['market_sigma'])
        self.add_report_observers()

    def add_report_observers(self):
        self.addobserver(GetDate)

    def add_report_analyzers(self, riskfree=0.01, targetrate=0.01, alpha=0.05, market_mu=0.07, market_sigma=0.15):
        """ Adds performance stats, required for report
            """
        if self.timeframe == bt.TimeFrame.Years:
            scalar = 1
        elif self.timeframe == bt.TimeFrame.Days:
            scalar = DAYS_IN_YEAR

        # Returns
        self.addanalyzer(MyAnnualReturn, _name="myAnnualReturn")
        self.addanalyzer(MyTimeReturn, _name="myTimeReturn",
                         fund=report_params['fundmode'])
        self.addanalyzer(MyLogReturnsRolling, _name="myLogReturnsRolling",
                         fund=report_params['fundmode'])
        self.addanalyzer(MyReturns, _name="myReturns",
                         fund=report_params['fundmode'],
                         tann=scalar)
        # Drawdowns
        self.addanalyzer(MyDrawDown, _name="myDrawDown",
                         fund=report_params['fundmode'])
        self.addanalyzer(MyTimeDrawDown, _name="myTimeDrawDown",
                         fund=report_params['fundmode'])
        # Distribution
        self.addanalyzer(MyDistributionMoments, _name="MyDistributionMoments",
                         timeframe=self.timeframe,
                         compression=1,
                         annualize=report_params['annualize'],
                         factor=scalar,
                         stddev_sample=report_params['stddev_sample'],
                         logreturns=report_params['logreturns'],
                         fund=report_params['fundmode'])
        # Risk-adjusted return based on Volatility
        self.addanalyzer(MyRiskAdjusted_VolBased, _name="MyRiskAdjusted_VolBased",
                         timeframe=self.timeframe,
                         compression=1,
                         annualize=report_params['annualize'],
                         stddev_sample=report_params['stddev_sample'],
                         logreturns=report_params['logreturns'],
                         fund=report_params['fundmode'],
                         riskfreerate=riskfree,
                         market_mu=market_mu,
                         market_sigma=market_sigma,
                         factor=scalar)
        # Risk-adjusted return based on Value at Risk
        self.addanalyzer(MyRiskAdjusted_VaRBased, _name="MyRiskAdjusted_VaRBased",
                         timeframe=self.timeframe,
                         compression=1,
                         annualize=report_params['annualize'],
                         stddev_sample=report_params['stddev_sample'],
                         logreturns=report_params['logreturns'],
                         fund=report_params['fundmode'],
                         riskfreerate=riskfree,
                         targetrate=targetrate,
                         factor=scalar,
                         alpha=alpha)
        # Risk-adjusted return based on Lower Partial Moments
        self.addanalyzer(MyRiskAdjusted_LPMBased, _name="MyRiskAdjusted_LPMBased",
                         timeframe=self.timeframe,
                         compression=1,
                         annualize=report_params['annualize'],
                         stddev_sample=report_params['stddev_sample'],
                         logreturns=report_params['logreturns'],
                         fund=report_params['fundmode'],
                         riskfreerate=riskfree,
                         targetrate=targetrate,
                         factor=scalar)

        # Risk-adjusted return based on Drawdown risk
        self.addanalyzer(MyRiskAdjusted_DDBased, _name="MyRiskAdjusted_DDBased",
                         timeframe=self.timeframe,
                         compression=1,
                         annualize=report_params['annualize'],
                         stddev_sample=report_params['stddev_sample'],
                         logreturns=report_params['logreturns'],
                         fund=report_params['fundmode'],
                         riskfreerate=riskfree,
                         factor=scalar)

        # Pyfolio
        self.addanalyzer(bt.analyzers.PyFolio,
                         timeframe=self.timeframe,
                         _name="myPyFolio")

    def get_strategy_backtest(self):
        return self.runstrats[0][0]

    def report(self, system=None):
        bt = self.get_strategy_backtest()
        rpt = PerformanceReport(bt, system=system, timeframe=self.timeframe)

        prices, returns, perf_data, targetweights, effectiveweights, params, assetprices = rpt.output_all_data()
        return prices, returns, perf_data, targetweights, effectiveweights, params, assetprices
