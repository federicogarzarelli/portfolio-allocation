import streamlit as st
from datetime import date, datetime, timedelta
from scipy import stats
import os, sys
import plotly.express as px
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import SessionState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from main import main
from GLOBAL_VARS import params

session_state = SessionState.get(startdate=datetime.strptime('2010-01-01', '%Y-%m-%d'), enddate = datetime.strptime('2021-01-01', '%Y-%m-%d'),
                                initial_cash=1000000.0, contribution=0.0, leverage=1.0, expense_ratio=0.01,
                                 historic="Historical DB (daily prices)", shares='SP500,ZB.F,ZN.F,BM.F,GC.C',
                                 shareclass='equity,bond_lt,bond_it,commodity,gold',
                                 weights='', benchmark='', indicator=False,
                                 riskparity=True, riskparity_nested=False, rotationstrat=False, uniform=True, vanillariskparity=False, onlystocks=False, sixtyforty=False,
                                 trend_u=False, absmom_u=False, relmom_u=False, momtrend_u=False, trend_rp=False, absmom_rp=False, relmom_rp=False, momtrend_rp=False, GEM=False,
                                 create_report=True, report_name='backtest report', user='FG', memo='backtest report',
                                 # advanced parameters
                                 DAYS_IN_YEAR=252, DAYS_IN_YEAR_BOND_PRICE=360,
                                 reb_days_days=21,reb_days_years=1,reb_days_custweights=21,
                                 lookback_period_short_days=20, lookback_period_short_years=5, lookback_period_short_custweights=2,
                                 lookback_period_long_days=120, lookback_period_long_years=10, lookback_period_long_custweights=2,
                                 moving_average_period_days=252,moving_average_period_years=5, moving_average_period_custweights=2,
                                 momentum_period_days=252, momentum_period_years=5, momentum_period_custweights=2,
                                 momentum_percentile_days=0.5, momentum_percentile_years=0.5,
                                 momentum_percentile_custweights=0.5,
                                 corrmethod_days='pearson', corrmethod_years='pearson',
                                 corrmethod_custweights='pearson',
                                 riskfree=0.01, targetrate=0.01, alpha=0.05, market_mu=0.07, market_sigma=0.15,
                                 stddev_sample=True, annualize=True, logreturns=False,
                                 assets_startdate=datetime.strptime('2010-01-01', '%Y-%m-%d'),assets_enddate = datetime.strptime('2021-01-01', '%Y-%m-%d'),
                                 assets_multiselect=['SP500','ZB.F','ZN.F','BM.F','GC.C'])

def app():
    st.title('Home')

    st.write('First adjust the backtest parameters on the left, then launch the backtest by pressing the button below.')

    st.sidebar.header("Backtest parameters")

    with st.form("input_params"):

        session_state.startdate = st.sidebar.date_input('start date', value=session_state.startdate,
                                                        min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                                                        max_value=date.today(), key='startdate',
                                                        help='start date of the backtest')
        session_state.enddate = st.sidebar.date_input('end date', value=session_state.enddate,
                                                      min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                                                      max_value=date.today(), key='enddate',
                                                      help='end date of the backtest')

        session_state.initial_cash = st.sidebar.number_input("initial cash", min_value=0.0, max_value=None,
                                                             value=session_state.initial_cash, step=1000.0, format='%f',
                                                             key='initial_cash', help='initial cash')
        session_state.contribution = st.sidebar.number_input("contribution or withdrawal", min_value=None, max_value=None,
                                                             value=session_state.contribution, format='%f',step=0.01,
                                                             key='contribution',
                                                             help='contribution or withdrawal. Can be specified as % of the portfolio value or in absolute terms.')
        session_state.leverage = st.sidebar.number_input("leverage", min_value=1.0, max_value=None,step=0.01,
                                                         value=session_state.leverage, format='%f', key='leverage',
                                                         help='daily leverage to apply to assets returns')
        session_state.expense_ratio = st.sidebar.number_input("expense ratio", min_value=0.0, max_value=1.0, step=0.01,
                                                         value=session_state.expense_ratio, format='%f', key='expense_ratio',
                                                         help='annual expense ratio')

        st.sidebar.subheader("Assets")

        if session_state.historic == "Yahoo Finance (daily prices)":
            idx = 0
        elif session_state.historic == "Historical DB (daily prices)":
            idx = 1
        else:
            idx = 2

        session_state.historic = st.sidebar.radio('data source', (
        "Yahoo Finance (daily prices)", "Historical DB (daily prices)", "Historical DB (yearly prices)"), index=idx,
                                                  key='historic', help='choose the data source')
        if session_state.historic == "Yahoo Finance (daily prices)":
            historic_cd = None
        elif session_state.historic == "Historical DB (daily prices)":
            historic_cd = 'medium'
        elif session_state.historic == "Historical DB (yearly prices)":
            historic_cd = 'long'

        session_state.shares = st.sidebar.text_area("assets to backtest", value=session_state.shares, height=None,
                                                    max_chars=None, key="shares",
                                                    help='tickers in a comma separated list (e.g. "SPY,TLT,GLD")')
        session_state.shareclass = st.sidebar.text_area("assets class (for Yahoo Finance only)",
                                                        value=session_state.shareclass, height=None, max_chars=None,
                                                        key="shareclass",
                                                        help='class of each asset (e.g. `equity,bond_lt,gold`). Possibilities are `equity, bond_lt, bond_it, gold, commodity`, where "bond_lt" and "bond_it" are long and intermediate duration bonds, respectively. __This argument is mandatory when the data source is Yahoo Finance.')
        session_state.weights = st.sidebar.text_area("asset weights", value=session_state.weights, height=None,
                                                     max_chars=None, key="weights",
                                                     help='list of portfolio weights for each asset specified (e.g. `0.35,0.35,0.30`). The weights need to sum to 1. When weights are specified a custom weights strategy is used that simply loads the weights specified. Alternative is to use a pre-defined strategy.')

        session_state.benchmark = st.sidebar.text_input("benchmark", value=session_state.benchmark, max_chars=None,
                                                        key='benchmark', help='ticker of a benchmark')
        session_state.indicator = st.sidebar.checkbox("signal assets", value=session_state.indicator, key='indicators',
                                                      help='load the signal assets needed for the rotation strategy')

        st.sidebar.subheader("Strategies")

        session_state.riskparity = st.sidebar.checkbox('risk parity', value=session_state.riskparity, key='riskparity',
                                                       help='Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run at portfolio level.')
        session_state.riskparity_nested = st.sidebar.checkbox('risk parity nested', value=session_state.riskparity_nested,
                                                              key='riskparity_nested',
                                                              help='Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run first at asset classe level (for assets belonging to the same asset class) and then at portfolio level.')
        session_state.rotationstrat = st.sidebar.checkbox('asset rotation', value=session_state.rotationstrat,
                                                          key='rotationstrat',
                                                          help='Asset rotation strategy that buy either gold, bonds or equities based on a signal (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy). To use this strategy tick the box signal assets.')
        session_state.uniform = st.sidebar.checkbox('uniform', value=session_state.uniform, key='uniform',
                                                    help='Static allocation uniform across asset classes. Assets are allocated uniformly within the same asset class.')
        session_state.vanillariskparity = st.sidebar.checkbox('static risk parity', value=session_state.vanillariskparity,
                                                              key='vanillariskparity',
                                                              help='Static allocation to asset classes where weights are taken from https://www.theoptimizingblog.com/leveraged-all-weather-portfolio/ (see section "True Risk Parity").')
        session_state.onlystocks = st.sidebar.checkbox('only equity', value=session_state.onlystocks, key='onlystocks',
                                                       help='Static allocation only to the equity class. Assets are allocated uniformly within the equity class.')
        session_state.sixtyforty = st.sidebar.checkbox('60% equities 40% bonds', value=session_state.sixtyforty,
                                                       key='sixtyforty',
                                                       help='Static allocation 60% to the equity class, 20% to the Long Term Bonds class and 20% to the Short Term Bonds class. Assets are allocated uniformly within the asset classes.')
        session_state.trend_u = st.sidebar.checkbox('trend uniform', value=session_state.trend_u, key='trend_u',
                                                    help='First weights are assigned according to the "uniform" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        session_state.absmom_u = st.sidebar.checkbox('absolute momentum uniform', value=session_state.absmom_u,
                                                     key='absmom_u',
                                                     help='First weights are assigned according to the "uniform" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).')
        session_state.relmom_u = st.sidebar.checkbox('relative momentum uniform', value=session_state.relmom_u,
                                                     key='relmom_u',
                                                     help='First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "uniform" strategy.')
        session_state.momtrend_u = st.sidebar.checkbox('relative momentum & trend uniform', value=session_state.momtrend_u,
                                                       key='momtrend_u',
                                                       help='First weights are assigned according to the "uniform" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        session_state.trend_rp = st.sidebar.checkbox('trend risk parity', value=session_state.trend_rp, key='trend_rp',
                                                     help='First weights are assigned according to the "riskparity" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        session_state.absmom_rp = st.sidebar.checkbox('absolute momentum risk parity', value=session_state.absmom_rp,
                                                      key='absmom_rp',
                                                      help='First weights are assigned according to the "riskparity" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).')
        session_state.relmom_rp = st.sidebar.checkbox('relative momentum risk parity', value=session_state.relmom_rp,
                                                      key='relmom_rp',
                                                      help='First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "risk parity" strategy.')
        session_state.momtrend_rp = st.sidebar.checkbox('relative momentum & trend risk parity',
                                                        value=session_state.momtrend_rp, key='momtrend_rp',
                                                        help='First weights are assigned according to the "riskparity" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        session_state.GEM = st.sidebar.checkbox('Global equity momentum', value=session_state.GEM, key='GEM',
                                                help='Global equity momentum strategy. Needs only 4 assets of classes equity, equity_intl, bond_lt, money_market. example: `--shares VEU,IVV,BIL,AGG --shareclass equity_intl,equity,money_market,bond_lt`. See https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/')

        st.sidebar.subheader("PDF Report")
        # session_state.create_report = st.sidebar.checkbox('create PDF report', value=session_state.create_report,
        #                                                   key='create_report', help=None)
        session_state.report_name = st.sidebar.text_input("report name", value=session_state.report_name, max_chars=None,
                                                          key='report_name', help=None)
        session_state.user = st.sidebar.text_input("user name", value=session_state.user, max_chars=None, key='user',
                                                   help='user generating the report')
        session_state.memo = st.sidebar.text_input("report memo", value=session_state.memo, max_chars=None, key='memo',
                                                   help='description of the report')

        #launch_btn = st.button("Launch backtest")
        launch_btn = st.form_submit_button("Launch backtest")

    params['startdate'] = session_state.startdate
    params['enddate'] = session_state.enddate
    params['initial_cash'] = session_state.initial_cash
    params['contribution'] = session_state.contribution
    params['leverage'] = session_state.leverage
    params['expense_ratio'] = session_state.expense_ratio
    params['historic'] = historic_cd
    params['shares'] = session_state.shares
    params['shareclass'] = session_state.shareclass
    params['weights'] = session_state.weights
    params['benchmark'] = session_state.benchmark
    params['indicator'] = session_state.indicator
    params['riskparity'] = session_state.riskparity
    params['riskparity_nested'] = session_state.riskparity_nested
    params['rotationstrat'] = session_state.rotationstrat
    params['uniform'] = session_state.uniform
    params['vanillariskparity'] = session_state.vanillariskparity
    params['onlystocks'] = session_state.onlystocks
    params['sixtyforty'] = session_state.sixtyforty
    params['trend_u'] = session_state.trend_u
    params['absmom_u'] = session_state.absmom_u
    params['relmom_u'] = session_state.relmom_u
    params['momtrend_u'] = session_state.momtrend_u
    params['trend_rp'] = session_state.trend_rp
    params['absmom_rp'] = session_state.absmom_rp
    params['relmom_rp'] = session_state.relmom_rp
    params['momtrend_rp'] = session_state.momtrend_rp
    params['GEM'] = session_state.GEM
    params['create_report'] = session_state.create_report
    params['report_name'] = session_state.report_name
    params['user'] = session_state.user
    params['memo'] = session_state.memo
    # advanced params
    params['DAYS_IN_YEAR'] = session_state.DAYS_IN_YEAR
    params['DAYS_IN_YEAR_BOND_PRICE'] = session_state.DAYS_IN_YEAR_BOND_PRICE
    params['reb_days_days'] = session_state.reb_days_days
    params['reb_days_years'] = session_state.reb_days_years
    params['reb_days_custweights'] = session_state.reb_days_custweights
    params['lookback_period_short_days'] = session_state.lookback_period_short_days
    params['lookback_period_short_years'] = session_state.lookback_period_short_years
    params['lookback_period_short_custweights'] = session_state.lookback_period_short_custweights
    params['lookback_period_long_days'] = session_state.lookback_period_long_days
    params['lookback_period_long_years'] = session_state.lookback_period_long_years
    params['lookback_period_long_custweights'] = session_state.lookback_period_long_custweights
    params['moving_average_period_days'] = session_state.moving_average_period_days
    params['moving_average_period_years'] = session_state.moving_average_period_years
    params['moving_average_period_custweights'] = session_state.moving_average_period_custweights
    params['momentum_period_days'] = session_state.momentum_period_days
    params['momentum_period_years'] = session_state.momentum_period_years
    params['momentum_period_custweights'] = session_state.momentum_period_custweights
    params['momentum_percentile_days'] = session_state.momentum_percentile_days
    params['momentum_percentile_years'] = session_state.momentum_percentile_years
    params['momentum_percentile_custweights'] = session_state.momentum_percentile_custweights
    params['corrmethod_days'] = session_state.corrmethod_days
    params['corrmethod_years'] = session_state.corrmethod_years
    params['corrmethod_custweights'] = session_state.corrmethod_custweights
    params['riskfree'] = session_state.riskfree
    params['targetrate'] = session_state.targetrate
    params['alpha'] = session_state.alpha
    params['market_mu'] = session_state.market_mu
    params['market_sigma'] = session_state.market_sigma
    params['stddev_sample'] = session_state.stddev_sample
    params['annualize'] = session_state.annualize
    params['logreturns'] = session_state.logreturns

    if launch_btn:
        main(params)

    # get the output and show it in the page
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    folder = os.path.join(basedir, 'output')
    today_str = datetime.today().strftime('%Y-%m-%d')
    outputfilename = ["/Fund_Prices_", "/Returns_", "/PerformanceMetrics_", "/Target_Weights_",
                      "/Effective_Weights_", "/Portfolio_Drawdown_", "/Asset_Prices_", "/Assets_drawdown_"]

    input_df = []
    outputfiles_exist = True
    for name in outputfilename:
        inputfilepath = folder + name + today_str + '.csv'
        if os.path.isfile(inputfilepath):
            input_df.append(pd.read_csv(inputfilepath, index_col=0))
        else:
            outputfiles_exist = False

    # Placeholder for all the charts and analysis
    #show_output = st.empty()
    #with show_output.beta_container():
    # with st.form(key='my_form'):

    if outputfiles_exist == True: # Only make charts if ALL the output exists.

        # Portfolio value
        idx = 0
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='strategy', value_name='price')

        fig = px.line(input_df_long, x="date", y="price", color="strategy")

        st.markdown("### Portfolio value")
        st.plotly_chart(fig, use_container_width=True)

        # Portfolio drawdowns
        idx = 5 # find a smarter way later
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='strategy', value_name='drawdown')

        fig = px.line(input_df_long, x="date", y="drawdown", color="strategy")

        st.markdown("### Portfolio drawdown")
        st.plotly_chart(fig, use_container_width=True)

        # Portfolio metrics
        st.markdown("### Portfolio metrics")
        st.dataframe(input_df[2])

        # Portfolio weights
        st.markdown("### Portfolio weights")
        col1, col2 = st.beta_columns(2)

        idx = 3
        for column in input_df[idx]:
            input_df[idx][column] = input_df[idx][column].astype(float).map(lambda n: '{:.2%}'.format(n))
        col1.subheader("Target weights")
        col1.dataframe(input_df[idx])

        idx = 4
        for column in input_df[idx]:
            input_df[idx][column] = input_df[idx][column].astype(float).map(lambda n: '{:.2%}'.format(n))
        col2.subheader("Effective weights")
        col2.dataframe(input_df[idx])

        # Asset value
        idx = 6
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='asset', value_name='price')

        fig = px.line(input_df_long, x="date", y="price", color="asset")

        st.markdown("### Assets value")
        st.plotly_chart(fig, use_container_width=True)

        # Assets drawdowns
        idx = 7 # find a smarter way later
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='asset', value_name='drawdown')

        fig = px.line(input_df_long, x="date", y="drawdown", color="asset")

        st.markdown("### Assets drawdown")
        st.plotly_chart(fig, use_container_width=True)

        # # Portfolio Returns
        idx = 1
        # Determine the price frequency
        dates=[]
        for i in range(1, len(input_df[idx].index)):
            dates.append(datetime.strptime(str(input_df[idx].index[i]), '%Y-%m-%d'))
        datediff = stats.mode(np.diff(dates))[0][0]
        if datediff > timedelta(days=250):
            frequency = "Years"
        elif datediff < timedelta(days=2):
            frequency = "Days"

        rolling_ret_period = st.slider("rolling returns period (in years)", min_value=1, max_value=30,
                                       value=1, step=1, format='%i', key='rolling_ret_period',
                                       help='period of rolling annual return (in years)')

        if frequency == "Days": # plot the rolling return (annualized)
            for column in input_df[idx]:
                if params['logreturns']:
                    input_df[idx][column] = (input_df[idx][column]).rolling(window=params['DAYS_IN_YEAR']*rolling_ret_period).sum()/rolling_ret_period
                else:
                    input_df[idx][column] = (1 + input_df[idx][column]).rolling(window=params['DAYS_IN_YEAR']*rolling_ret_period).apply(np.prod) ** (1 / rolling_ret_period) - 1
        elif frequency == "Years": # plot the rolling 5 years return
            for column in input_df[idx]:
                if params['logreturns']:
                    input_df[idx][column] = (input_df[idx][column]).rolling(window=rolling_ret_period).mean()
                else:
                    input_df[idx][column] = (1 + input_df[idx][column]).rolling(window=rolling_ret_period).apply(np.prod) - 1


        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='strategy', value_name='rolling return')

        fig = px.line(input_df_long, x="date", y="rolling return", color="strategy")

        st.markdown("### Portfolio returns")
        st.plotly_chart(fig, use_container_width=True)

