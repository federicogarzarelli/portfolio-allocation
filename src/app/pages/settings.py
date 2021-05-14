import streamlit as st

def app():
    st.title('Advanced Settings')

    st.write('Here you can modify advanced settings for the backtesting.')

    st.markdown('## General parameters')
    DAYS_IN_YEAR=st.number_input("Days in year", min_value=1, max_value=366, value=260, format='%i', key='DAYS_IN_YEAR', help='Number of days in a year. Default is 260.')
    DAYS_IN_YEAR_BOND_PRICE=st.number_input("Days in year for bonds", min_value=1, max_value=366, value=360, format='%i', key='DAYS_IN_YEAR_BOND_PRICE', help='Number of days in a year used for calculating bond prices from yields. Default is 360.')

    st.markdown('## Strategy parameters')

    col1, col2, col3 = st.beta_columns(3)
    col1.subheader("daily frequency")
    col2.subheader("yearly frequency")
    col3.subheader("custom weights")

    tooltip = 'Number of days (of bars) every which the portfolio is rebalanced. Default is 21 for daily data and 1 for yearly data.'
    reb_days_days=col1.number_input("rebalance days", min_value=1, max_value=366, value=21, format='%i', key='reb_days_days', help=tooltip)
    reb_days_years=col2.number_input("rebalance days", min_value=1, max_value=366, value=1, format='%i', key='reb_days_years', help=tooltip)
    reb_days_custweights=col3.number_input("rebalance days", min_value=1, max_value=366, value=21, format='%i', key='reb_days_custweights', help=tooltip)

    tooltip = "Window to calculate the standard deviation of assets returns. Applies to strategy `riskparity` and derived strategies. Default is 20 for daily data and 10 for yearly data."
    lookback_period_short_days=col1.number_input("lookback period volatility", min_value=2, max_value=366, value=20, format='%i', key='lookback_period_short_days', help=tooltip)
    lookback_period_short_years=col2.number_input("lookback period volatility", min_value=2, max_value=366, value=5, format='%i', key='lookback_period_short_years', help=tooltip)
    lookback_period_short_custweights=col3.number_input("lookback period volatility", min_value=2, max_value=366, value=2, format='%i', key='lookback_period_short_custweights', help=tooltip)

    tooltip = "Window to calculate the correlation matrix of assets returns. Applies to strategies `riskparity` and derived strategies. Default is 120 for daily data and 10 for yearly data."
    lookback_period_long_days=col1.number_input("lookback period correlation", min_value=2, max_value=366, value=120, format='%i', key='lookback_period_long_days', help=tooltip)
    lookback_period_long_years=col2.number_input("lookback period correlation", min_value=2, max_value=366, value=10, format='%i', key='lookback_period_long_years', help=tooltip)
    lookback_period_long_custweights=col3.number_input("lookback period correlation", min_value=2, max_value=366, value=2, format='%i', key='lookback_period_long_custweights', help=tooltip)

    tooltip = "Window to calculate the simple moving average. Applies to strategies `trend_uniform`, `trend_riskparity`, `momentumtrend_uniform`  and `momentumtrend_riskparity`. Default is 252 for daily data and 5 for yearly data."
    moving_average_period_days=col1.number_input("trend period", min_value=2, max_value=366, value=252, format='%i', key='moving_average_period_days', help=tooltip)
    moving_average_period_years=col2.number_input("trend period", min_value=2, max_value=366, value=5, format='%i', key='moving_average_period_years', help=tooltip)
    moving_average_period_custweights=col3.number_input("trend period", min_value=2, max_value=366, value=2, format='%i', key='moving_average_period_custweights', help=tooltip)

    tooltip = "Window to calculate the momentum. Applies to strategies `absolutemomentum_uniform`, `relativemomentum_uniform`, `momentumtrend_uniform`, `absolutemomentum_riskparity`, `relativemomentum_riskparity`  and `momentumtrend_riskparity`. Default is 252 for daily data and 5 for yearly data."
    momentum_period_days=col1.number_input("momentum period", min_value=2, max_value=366, value=252, format='%i', key='momentum_period_days', help=tooltip)
    momentum_period_years=col2.number_input("momentum period", min_value=2, max_value=366, value=5, format='%i', key='momentum_period_years', help=tooltip)
    momentum_period_custweights=col3.number_input("momentum period", min_value=2, max_value=366, value=2, format='%i', key='momentum_period_custweights', help=tooltip)

    tooltip = "Percentile of assets with the highest return in a period to form the relative momentum portfolio. The higher the percentile, the higher the return quantile."
    momentum_percentile_days=col1.number_input("momentum percentile", min_value=0.0, max_value=1.0, value=0.5, format='%f', key='momentum_percentile_days', help=tooltip)
    momentum_percentile_years=col2.number_input("momentum percentile", min_value=0.0, max_value=1.0, value=0.5, format='%f', key='momentum_percentile_years', help=tooltip)
    momentum_percentile_custweights=col3.number_input("momentum percentile", min_value=0.0, max_value=1.0, value=0.5, format='%f', key='momentum_percentile_custweights', help=tooltip)

    tooltip = "Method for the calculation of the correlation matrix. Applies to strategies `riskparity` and `riskparity_nested`. Default is 'pearson'. Alternative is 'spearman'."
    corrmethod_days=col1.selectbox("correlation coefficient", ('pearson', 'spearman'), index=0, key='corrmethod_days', help=tooltip)
    corrmethod_years=col2.selectbox("correlation coefficient", ('pearson', 'spearman'), index=0, key='corrmethod_years', help=tooltip)
    corrmethod_custweights=col3.selectbox("correlation coefficient", ('pearson', 'spearman'), index=0, key='corrmethod_custweights', help=tooltip)

    st.markdown('## Report parameters')

    riskfree=st.number_input("risk free rate", min_value=0.0, max_value=1.0, value=0.01, format='%f', key='riskfree', help="Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc. Default is 0.01.")
    targetrate=st.number_input("target rate", min_value=0.0, max_value=1.0, value=0.01, format='%f', key='targetrate',
                    help="Target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio. Default is 0.01.")
    alpha=st.number_input("alpha (VaR)", min_value=0.0, max_value=1.0, value=0.05, format='%f', key='alpha',
                    help="Confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio). Default is 0.05.")
    market_mu=st.number_input("market mu", min_value=0.0, max_value=1.0, value=0.07, format='%f', key='market_mu',
                    help="Average annual return of the market, to be used in Treynor ratio, Information ratio. Default is 0.07.")
    market_sigma=st.number_input("market sigma", min_value=0.0, max_value=1.0, value=0.15, format='%f', key='market_sigma',
                    help="Annual standard deviation of the market, to be used in Treynor ratio, Information ratio. Default is  0.15.")
    stddev_sample=st.checkbox("standard deviation sample adjustment", value=True, key="stddev_sample",
                help='Bessel correction (N-1) when calculating standard deviation from a sample')
    annualize=st.checkbox("annualize first", value=True, key="annualize",
                help='calculate annualized metrics by annualizing returns first')
    logreturns=st.checkbox("logreturns", value=False, key="logreturns",
                help='Use logreturns instead of percentage returns when calculating metrics (not recommended). Default is False.')

    params = {'DAYS_IN_YEAR':DAYS_IN_YEAR,
              'DAYS_IN_YEAR_BOND_PRICE':DAYS_IN_YEAR_BOND_PRICE,
    }
    return params
