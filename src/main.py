from GLOBAL_VARS import *
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro
import report_aggregator
import sys
pd.options.mode.chained_assignment = None  # default='warn'
from PortfolioDB import PortfolioDB
import platform
import streamlit as st


# Strategy parameters not passed
from strategies import customweights

def runOneStrat(strategy=None):

    # startdate = datetime.datetime.strptime(params['startdate'], "%Y-%m-%d")
    # enddate = datetime.datetime.strptime(params['enddate'], "%Y-%m-%d")
    startdate = params['startdate']
    enddate = params['enddate']

    # Add the data
    data = []
    if params['historic'] == 'medium':
        timeframe = bt.TimeFrame.Days

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        # cerebro = Cerebro()
        # cerebro.broker.set_coo(True)
        # cerebro.broker.set_coc(True)
        cerebro.broker.set_cash(params['initial_cash'])
        cerebro.broker.set_checksubmit(False)
        cerebro.broker.set_shortcash(True) # Can short the cash

        # Import the historical assets
        shares_list = params['shares'].split(',') # GLD,COM,SP500,LTB,ITB,TIP
        for share in shares_list:
            #df = import_process_hist(share, args)
            df = import_histprices_db(share)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_leverage(df[column], leverage=params['leverage'], expense_ratio=params['expense_ratio'], timeframe=timeframe)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            #data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))
            data.append(df)

        if params['shareclass'] == '':
            db = PortfolioDB(databaseName=DB_NAME)
            shareclass = []
            for share in shares_list:
                thisshareinfo = db.getStockInfo(share)
                shareclass.append(thisshareinfo['asset_class'].values[0])
            #shareclass = [assetclass_dict[x] for x in shares_list]
        else:
            shareclass = params['shareclass'].split(',')

    elif params['historic'] == 'long':
        timeframe = bt.TimeFrame.Years

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        cerebro.broker.set_cash(params['initial_cash'])

        # Import the historical assets
        shares_list = params['shares'].split(',') # GLD_LNG,OIL_LNG,EQ_LNG,LTB_LNG,ITB_LNG,RE_LNG
        for share in shares_list:
            #df = import_process_hist(share, args)
            df = import_histprices_db(share)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_leverage(df[column], leverage=params['leverage'], expense_ratio=params['expense_ratio'], timeframe=timeframe)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            #data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))
            data.append(df)

        if params['shareclass'] == '':
            db = PortfolioDB(databaseName=DB_NAME)
            shareclass = []
            for share in shares_list:
                thisshareinfo = db.getStockInfo(share)
                shareclass.append(thisshareinfo['asset_class'].values[0])
        else:
            shareclass = params['shareclass'].split(',')

    else:
        shares_list = params['shares'].split(',')
        timeframe = bt.TimeFrame.Days

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        cerebro.broker.set_cash(params['initial_cash'])

        # download the datas
        for i in range(len(shares_list)):
            this_assets = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]

            if APPLY_LEVERAGE_ON_LIVE_STOCKS == True:
                this_assets = add_leverage(this_assets, leverage=params['leverage'],
                                                          expense_ratio=params['expense_ratio'],timeframe=timeframe).to_frame("close")
            else:
                this_assets = this_assets.to_frame("close")

            for column in ['open', 'high', 'low']:
                this_assets[column] = this_assets['close']

            this_assets['volume'] = 0

            #data.append(bt.feeds.PandasData(dataname=this_assets, fromdate=startdate, todate=enddate, timeframe=timeframe))
            data.append(this_assets)

        shareclass = params['shareclass'].split(',')

    # Set the minimum periods, lookback period (and other parameters) depending on the data used (daily or yearly) and
    # if weights are used instead of a strategy
    if params['weights'] != '' or strategy == 'customweights':
        corrmethod = params['corrmethod_custweights']
        reb_days = params['reb_days_custweights']
        lookback_period_short = params['lookback_period_short_custweights']
        lookback_period_long = params['lookback_period_long_custweights']
        moving_average_period = params['moving_average_period_custweights']
        momentum_period = params['momentum_period_custweights']
        momentum_percentile = params['momentum_percentile_custweights']
    elif timeframe == bt.TimeFrame.Days:
        corrmethod = params['corrmethod_days']
        reb_days = params['reb_days_days']
        lookback_period_short = params['lookback_period_short_days']
        lookback_period_long = params['lookback_period_long_days']
        moving_average_period = params['moving_average_period_days']
        momentum_period = params['momentum_period_days']
        momentum_percentile = params['momentum_percentile_days']
    elif timeframe == bt.TimeFrame.Years:
        corrmethod = params['corrmethod_years']
        reb_days = params['reb_days_years']
        lookback_period_short = params['lookback_period_short_years']
        lookback_period_long = params['lookback_period_long_years']
        moving_average_period = params['moving_average_period_years']
        momentum_period = params['momentum_period_years']
        momentum_percentile = params['momentum_percentile_years']

    if timeframe != bt.TimeFrame.Days and params['indicator']:
        sys.exit('Error: Indicators can only added in backtest with daily frequency')

    if params['indicator']:
        # now import the non-tradable indexes for the rotational strategy
        indicatorLabels = ['DFII20', 'T10Y2Y', 'T10YIE_T10Y2Y']
        all_indicators = load_economic_curves(startdate, enddate)
        for indicatorLabel in indicatorLabels:
            df = all_indicators[[indicatorLabel]]
            for column in ['open', 'high', 'low', 'close']:
                df[column] = df[indicatorLabel]

            df['volume'] = 0
            df = df[['open', 'high', 'low', 'close', 'volume']]
            #data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))
            data.append(df)

        shareclass = shareclass + ['non-tradable', 'non-tradable', 'non-tradable']
        shares_list = shares_list + indicatorLabels

    if params['benchmark'] != '':
        # look for the benchmark in the database
        #benchmark_df = import_process_hist(params['benchmark, args) # First look for the benchmark in the historical "database"
        benchmark_df = import_histprices_db(params['benchmark'])

        if benchmark_df is None: # if not, download it
            benchmark_df = web.DataReader(params['benchmark'], "yahoo", startdate, enddate)["Adj Close"]
            benchmark_df = benchmark_df.to_frame("close")

        if benchmark_df is not None:
            for column in ['open', 'high', 'low']:
                benchmark_df[column] = benchmark_df['close']

            benchmark_df['volume'] = 0

        #data.append(bt.feeds.PandasData(dataname=benchmark_df, fromdate=startdate, todate=enddate, timeframe=timeframe))
        data.append(benchmark_df)

        shareclass = shareclass + ['benchmark']
        shares_list = shares_list + [params['benchmark']]

    data = common_dates(data=data, fromdate=startdate, todate=enddate, timeframe=timeframe)

    i = 0
    for dt in data:
        dt_feed = bt.feeds.PandasData(dataname=dt, fromdate=startdate, todate=enddate, timeframe=timeframe)
        cerebro.adddata(dt_feed, name=shares_list[i])
        i = i + 1

    n_assets = len([x for x in shareclass if x not in ['non-tradable', 'benchmark']])
    cerebro.addobserver(targetweightsobserver, n_assets=n_assets)
    cerebro.addobserver(effectiveweightsobserver, n_assets=n_assets)

    # if you provide the weights, use them
    if params['weights'] != '' and strategy == 'customweights':
        weights_list = params['weights'].split(',')
        weights_listt = [float(i) for i in weights_list]

        cerebro.addstrategy(customweights,
                            n_assets=n_assets,
                            initial_cash=params['initial_cash'],
                            contribution=params['contribution'],
                            assetweights=weights_listt,
                            shareclass=shareclass,
                            printlog=True,
                            corrmethod=corrmethod,
                            reb_days=reb_days,
                            lookback_period_short=lookback_period_short,
                            lookback_period_long=lookback_period_long,
                            moving_average_period=moving_average_period,
                            momentum_period=momentum_period,
                            momentum_percentile=momentum_percentile
                            )

    # otherwise, rely on the weights of a strategy
    else:
        cerebro.addstrategy(eval(strategy),
                            n_assets=n_assets,
                            initial_cash=params['initial_cash'],
                            contribution=params['contribution'],
                            shareclass=shareclass,
                            printlog=True,
                            corrmethod=corrmethod,
                            reb_days=reb_days,
                            lookback_period_short=lookback_period_short,
                            lookback_period_long=lookback_period_long,
                            moving_average_period=moving_average_period,
                            momentum_period=momentum_period,
                            momentum_percentile=momentum_percentile
                            )

    # Run backtest
    cerebro.run()
    #cerebro.plot(volume=False)

    # Create report
    if params['create_report']:
        OutputList = cerebro.report(system=platform.system())

        return OutputList

@st.cache(suppress_st_warning=True)
def main(params):
    # Fund mode if contribution is 0 otherwise, asset mode
    if params['contribution'] == 0:
        params["fundmode"] = True
    else:
        params["fundmode"] = False

    # Clear the output folder
    outputdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputdir = find("output", outputdir)
    delete_in_dir(outputdir)

    strategy_list = []
    if params['riskparity']:
        strategy_list.append('riskparity')
    if params['riskparity_nested']:
        strategy_list.append('riskparity_nested')
    if params['rotationstrat']:
        strategy_list.append('rotationstrat')
    if params['uniform']:
        strategy_list.append('uniform')
    if params['vanillariskparity']:
        strategy_list.append('vanillariskparity')
    if params['onlystocks']:
        strategy_list.append('onlystocks')
    if params['sixtyforty']:
        strategy_list.append('sixtyforty')
    if params['trend_u']:
        strategy_list.append('trend_u')
    if params['absmom_u']:
        strategy_list.append('absmom_u')
    if params['relmom_u']:
        strategy_list.append('relmom_u')
    if params['momtrend_u']:
        strategy_list.append('momtrend_u')
    if params['trend_rp']:
        strategy_list.append('trend_rp')
    if params['absmom_rp']:
        strategy_list.append('absmom_rp')
    if params['relmom_rp']:
        strategy_list.append('relmom_rp')
    if params['momtrend_rp']:
        strategy_list.append('momtrend_rp')
    if params['GEM']:
        strategy_list.append('GEM')
    if not strategy_list:
        strategy_list = ["customweights"]
    if params['benchmark'] != '':
        strategy_list = strategy_list + ['benchmark']

    print_header(params,strategy_list)

    # Output list description:
    # list index, content
    # 0, prices
    # 1, returns
    # 2, performance data
    # 3, target weights
    # 4, effective weights
    # 5, portfolio drawdown
    # 6, assetprices
    # 7, assets drawdown
    # 8, parameters
    InputList = []
    stratIndependentOutput = [6, 7, 8] # these indexes correspond to the strategy independent outputs

    for i in range(0,9):
        InputList.append(pd.DataFrame())

    for strat in strategy_list:
        print_section_divider(strat)
        st.write("Backtesting strategy: " + strat)

        ThisOutputList = runOneStrat(strat)

        for i in range(0, len(ThisOutputList)):
            if strat == strategy_list[0] or i in stratIndependentOutput:
                InputList[i] = ThisOutputList[i]
            else:
                InputList[i][strat] = ThisOutputList[i]

    if params['report_name'] != '':
        outfilename = params['report_name'] + "_" + get_now() + ".pdf"
    else:
        outfilename = "Report_" + get_now() + ".pdf"

    ReportAggregator = report_aggregator.ReportAggregator(outfilename, outputdir, params['user'], params['memo'], params['leverage'], platform.system(), InputList)
    ReportAggregator.report()

