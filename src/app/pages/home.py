import streamlit as st
from datetime import date, datetime

def app():
    st.title('Home')

    st.write('First adjust the backtest parameters on the left, then launch the backtest by pressing the button below.')

    st.sidebar.header("Backtest parameters")


    startdate=st.sidebar.date_input('start date', value=datetime.strptime('1990-01-01', '%Y-%m-%d'), min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                            max_value=date.today() , key='startdate', help='start date of the backtest')
    enddate = st.sidebar.date_input('end date', value=datetime.strptime('1990-01-01', '%Y-%m-%d'),
                              min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'), max_value=date.today(), key='enddate',
                              help='end date of the backtest')

    initial_cash=st.sidebar.number_input("initial cash", min_value=0.0, max_value=None, value=1000000.0, step=1000.0, format='%f',
                 key='initial_cash', help='initial cash')
    contribution=st.sidebar.number_input("contribution or withdrawal", min_value=None, max_value=None, value=0.0, format='%f',
                 key='contribution', help='contribution or withdrawal. Can be specified as % of the portfolio value or in absolute terms.')
    leverage=st.sidebar.number_input("leverage", min_value=1.0, max_value=None, value=1.0, format='%f',
                 key='leverage', help='daily leverage to apply to assets returns')

    st.sidebar.subheader("Assets")

    historic=st.sidebar.radio('data source', ("Yahoo Finance (daily prices)","Historical DB (daily prices)","Historical DB (yearly prices)"), index=0, key='historic', help='choose the data source')
    if historic=="Yahoo Finance (daily prices)":
        historic=None
    elif historic=="Historical DB (daily prices)":
        historic='medium'
    elif historic=="Historical DB (yearly prices)":
        historic = 'long'

    shares=st.sidebar.text_area("assets to backtest", value='', height=None, max_chars=None, key="shares", help='tickers in a comma separated list (e.g. "SPY,TLT,GLD")')
    shareclass=st.sidebar.text_area("assets class (for Yahoo Finance only)", value='', height=None, max_chars=None, key="shareclass", help='class of each asset (e.g. `equity,bond_lt,gold`). Possibilities are `equity, bond_lt, bond_it, gold, commodity`, where "bond_lt" and "bond_it" are long and intermediate duration bonds, respectively. __This argument is mandatory when the data source is Yahoo Finance.')
    weights=st.sidebar.text_area("asset weights", value='', height=None, max_chars=None,
                          key="weights",
                          help='list of portfolio weights for each asset specified (e.g. `0.35,0.35,0.30`). The weights need to sum to 1. When weights are specified a custom weights strategy is used that simply loads the weights specified. Alternative is to use a pre-defined strategy.')

    benchmark=st.sidebar.text_input("benchmark", value='', max_chars=None, key='benchmark', help='ticker of a benchmark')
    indicator = st.sidebar.checkbox("signal assets", value=False, key='indicators', help='load the signal assets needed for the rotation strategy')

    st.sidebar.subheader("Strategies")

    riskparity = st.sidebar.checkbox('risk parity', value=True, key='riskparity', help='Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run at portfolio level.')
    riskparity_nested = st.sidebar.checkbox('risk parity nested', value=False, key='riskparity_nested', help='Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run first at asset classe level (for assets belonging to the same asset class) and then at portfolio level.')
    rotationstrat = st.sidebar.checkbox('asset rotation', value=False, key='rotationstrat', help='Asset rotation strategy that buy either gold, bonds or equities based on a signal (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy). To use this strategy tick the box signal assets.')
    uniform=st.sidebar.checkbox('uniform', value=True, key='uniform', help='Static allocation uniform across asset classes. Assets are allocated uniformly within the same asset class.')
    vanillariskparity=st.sidebar.checkbox('static risk parity', value=False, key='vanillariskparity', help='Static allocation to asset classes where weights are taken from https://www.theoptimizingblog.com/leveraged-all-weather-portfolio/ (see section "True Risk Parity").')
    onlystocks=st.sidebar.checkbox('only equity', value=False, key='onlystocks', help='Static allocation only to the equity class. Assets are allocated uniformly within the equity class.')
    sixtyforty=st.sidebar.checkbox('60% equities 40% bonds', value=False, key='sixtyforty', help='Static allocation 60% to the equity class, 20% to the Long Term Bonds class and 20% to the Short Term Bonds class. Assets are allocated uniformly within the asset classes.')
    trend_u=st.sidebar.checkbox('trend uniform', value=False, key='trend_u', help='First weights are assigned according to the "uniform" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
    absmom_u=st.sidebar.checkbox('absolute momentum uniform', value=False, key='absmom_u', help='First weights are assigned according to the "uniform" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).')
    relmom_u = st.sidebar.checkbox('relative momentum uniform', value=False, key='relmom_u', help='First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "uniform" strategy.')
    momtrend_u = st.sidebar.checkbox('relative momentum & trend uniform', value=False, key='momtrend_u', help='First weights are assigned according to the "uniform" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
    trend_rp= st.sidebar.checkbox('trend risk parity', value=False, key='trend_rp', help='First weights are assigned according to the "riskparity" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
    absmom_rp = st.sidebar.checkbox('absolute momentum risk parity', value=False, key='absmom_rp', help='First weights are assigned according to the "riskparity" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).')
    relmom_rp = st.sidebar.checkbox('relative momentum risk parity', value=False, key='relmom_rp', help='First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "risk parity" strategy.')
    momtrend_rp = st.sidebar.checkbox('relative momentum & trend risk parity', value=False, key='momtrend_rp', help='First weights are assigned according to the "riskparity" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
    GEM = st.sidebar.checkbox('Global equity momentum', value=False, key='GEM', help='Global equity momentum strategy. Needs only 4 assets of classes equity, equity_intl, bond_lt, money_market. example: `--shares VEU,IVV,BIL,AGG --shareclass equity_intl,equity,money_market,bond_lt`. See https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/')

    strategies=[]
    if riskparity:
        strategies.append('riskparity')
    if riskparity_nested:
        strategies.append('riskparity_nested')
    if rotationstrat:
        strategies.append('rotationstrat')
    if uniform:
        strategies.append('uniform')
    if vanillariskparity:
        strategies.append('vanillariskparity')
    if onlystocks:
        strategies.append('onlystocks')
    if sixtyforty:
        strategies.append('sixtyforty')
    if trend_u:
        strategies.append('trend_u')
    if absmom_u:
        strategies.append('absmom_u')
    if relmom_u:
        strategies.append('relmom_u')
    if momtrend_u:
        strategies.append('momtrend_u')
    if trend_rp:
        strategies.append('trend_rp')
    if absmom_rp:
        strategies.append('absmom_rp')
    if relmom_rp:
        strategies.append('relmom_rp')
    if momtrend_rp:
        strategies.append('momtrend_rp')
    if GEM:
        strategies.append('GEM')

    launch_btn=st.button("Launch backtest",key="launch_btn")