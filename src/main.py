from GLOBAL_VARS import *
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro
from report_aggregator import ReportAggregator
import sys
pd.options.mode.chained_assignment = None  # default='warn'


# Strategy parameters not passed
from strategies import customweights


def parse_args():
    now = datetime.datetime.now().strftime("%Y_%m_%d")  # string to be used after
    parser = argparse.ArgumentParser(description='main class to run strategies')
    parser.add_argument('--historic', type=str, default=None, required=False,
                        help='"Long" for yearly data from 1900, "medium" for daily data from the 1970')
    parser.add_argument('--shares', type=str, default='SPY,TLT', required=False,
                        help='string corresponding to list of shares')
    parser.add_argument('--shareclass', type=str, default=None, required=False,
                        help='string corresponding to list of asset classes, if not using historic')
    parser.add_argument('--weights', type=str, default='', required=False,
                        help='string corresponding to list of weights. if no values, strategy weights are taken')
    parser.add_argument('--indicators', action='store_true', default=False, required=False,
                        help='include indicators for rotational strategy, if true')
    parser.add_argument('--benchmark', type=str, required=False, help='Benchmark index')
    parser.add_argument('--initial_cash', type=int, default=100000, required=False, help='initial_cash to start with')
    parser.add_argument('--contribution', type=float, default=0, required=False, help='investment or withdrawal')
    parser.add_argument('--strategy', type=str, required=False,
                        help='Specify the strategies for which a backtest is run')
    parser.add_argument('--startdate', type=str, default='2017-01-01', required=False,
                        help='starting date of the simulation')
    parser.add_argument('--enddate', type=str, default=now, required=False, help='end date of the simulation')
    parser.add_argument('--system', type=str, default='windows', help='operating system, to deal with different paths')
    parser.add_argument('--leverage', type=float, default=1.0, help='leverage to consider')
    parser.add_argument('--create_report', action='store_true', default=False, required=False,
                        help='creates a report if true')
    parser.add_argument('--report_name', type=str, default=None, required=False,
                        help='if create_report is True, it is better to have a specific name')
    parser.add_argument('--user', type=str, default="Federico & Fabio", required=False, help='user generating the report')
    parser.add_argument('--memo', type=str, default="Backtest", required=False, help='Description of the report')

    return parser.parse_args()

def runOneStrat(strategy=None):
    args = parse_args()
    if strategy is None:
        strategy = args.strategy

    startdate = datetime.datetime.strptime(args.startdate, "%Y-%m-%d")
    enddate = datetime.datetime.strptime(args.enddate, "%Y-%m-%d")

    # Add the data
    data = []
    if args.historic == 'medium':
        timeframe = bt.TimeFrame.Days

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        # cerebro = Cerebro()
        # cerebro.broker.set_coo(True)
        # cerebro.broker.set_coc(True)
        cerebro.broker.set_cash(args.initial_cash)
        cerebro.broker.set_checksubmit(False)
        cerebro.broker.set_shortcash(True) # Can short the cash

        # Import the historical assets
        shares_list = args.shares.split(',') # GLD,COM,SP500,LTB,ITB,TIP
        for share in shares_list:
            df = import_process_hist(share, args)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_leverage(df[column], leverage=args.leverage, expense_ratio=expense_ratio, timeframe=timeframe)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            #data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))
            data.append(df)

        if args.shareclass is None:
            shareclass = [assetclass_dict[x] for x in shares_list]
        else:
            shareclass = args.shareclass.split(',')

    elif args.historic == 'long':
        timeframe = bt.TimeFrame.Years

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        cerebro.broker.set_cash(args.initial_cash)

        # Import the historical assets
        shares_list = args.shares.split(',') # GLD_LNG,OIL_LNG,EQ_LNG,LTB_LNG,ITB_LNG,RE_LNG
        for share in shares_list:
            df = import_process_hist(share, args)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_leverage(df[column], leverage=args.leverage, expense_ratio=expense_ratio, timeframe=timeframe)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            #data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))
            data.append(df)

        if args.shareclass is None:
            shareclass = [assetclass_dict[x] for x in shares_list]
        else:
            shareclass = args.shareclass.split(',')

    else:
        shares_list = args.shares.split(',')
        timeframe = bt.TimeFrame.Days

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        cerebro.broker.set_cash(args.initial_cash)

        # download the datas
        for i in range(len(shares_list)):
            this_assets = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]

            if APPLY_LEVERAGE_ON_LIVE_STOCKS == True:
                this_assets = add_leverage(this_assets, leverage=args.leverage,
                                                          expense_ratio=expense_ratio,timeframe=timeframe).to_frame("close")
            else:
                this_assets = this_assets.to_frame("close")

            for column in ['open', 'high', 'low']:
                this_assets[column] = this_assets['close']

            this_assets['volume'] = 0

            #data.append(bt.feeds.PandasData(dataname=this_assets, fromdate=startdate, todate=enddate, timeframe=timeframe))
            data.append(this_assets)

        shareclass = args.shareclass.split(',')

    # Set the minimum periods, lookback period (and other parameters) depending on the data used (daily or yearly) and
    # if weights are used instead of a strategy
    if args.weights != '' or strategy is None:
        strat_params = strat_params_weights
    elif timeframe == bt.TimeFrame.Days:
        strat_params = strat_params_days
    elif timeframe == bt.TimeFrame.Years:
        strat_params = strat_params_years

    if timeframe != bt.TimeFrame.Days and args.indicators:
        sys.exit('Error: Indicators can only added in backtest with daily frequency')

    if args.indicators:
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

    if args.benchmark is not None:
        # download the datas
        benchmark_df = import_process_hist(args.benchmark, args) # First look for the benchmark in the historical "database"

        if benchmark_df is None: # if not, download it
            benchmark_df = web.DataReader(args.benchmark, "yahoo", startdate, enddate)["Adj Close"]
            benchmark_df = benchmark_df.to_frame("close")

        if benchmark_df is not None:
            for column in ['open', 'high', 'low']:
                benchmark_df[column] = benchmark_df['close']

            benchmark_df['volume'] = 0

        #data.append(bt.feeds.PandasData(dataname=benchmark_df, fromdate=startdate, todate=enddate, timeframe=timeframe))
        data.append(benchmark_df)

        shareclass = shareclass + ['benchmark']
        shares_list = shares_list + [args.benchmark]

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
    if args.weights != '' and strategy == 'customweights':
        weights_list = args.weights.split(',')
        weights_listt = [float(i) for i in weights_list]

        cerebro.addstrategy(customweights,
                            n_assets=n_assets,
                            initial_cash=args.initial_cash,
                            contribution=args.contribution,
                            assetweights=weights_listt,
                            shareclass=shareclass,
                            printlog=strat_params.get('printlog'),
                            corrmethod=strat_params.get('corrmethod'),
                            reb_days=strat_params.get('reb_days'),
                            lookback_period_short=strat_params.get('lookback_period_short'),
                            lookback_period_long=strat_params.get('lookback_period_long')
                            )

    # otherwise, rely on the weights of a strategy
    else:
        strategy.split(',')
        cerebro.addstrategy(eval(strategy),
                            n_assets=n_assets,
                            initial_cash=args.initial_cash,
                            contribution=args.contribution,
                            shareclass=shareclass,
                            printlog=strat_params.get('printlog'),
                            corrmethod=strat_params.get('corrmethod'),
                            reb_days=strat_params.get('reb_days'),
                            lookback_period_short=strat_params.get('lookback_period_short'),
                            lookback_period_long=strat_params.get('lookback_period_long')
                            )

    # Run backtest
    cerebro.run()
    cerebro.plot(volume=False)

    # Create report
    if args.create_report:
        OutputList = cerebro.report(system=args.system)

        return OutputList


if __name__ == '__main__':
    args = parse_args()

    # Fund mode if contribution is 0 otherwise, asset mode
    if args.contribution == 0:
        report_params["fundmode"] = True
    else:
        report_params["fundmode"] = False

    # Clear the output folder
    outputdir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "output")
    delete_in_dir(outputdir)

    print_header(args)

    if args.strategy is None:
        strategy_list = ["customweights"]
    else:
        strategy_list = args.strategy.split(',')

    if args.benchmark is not None:
        strategy_list = strategy_list + ['benchmark']

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

        ThisOutputList = runOneStrat(strat)

        for i in range(0, len(ThisOutputList)):
            if strat == strategy_list[0] or i in stratIndependentOutput:
                InputList[i] = ThisOutputList[i]
            else:
                InputList[i][strat] = ThisOutputList[i]

    if args.report_name is not None:
        outfilename = args.report_name + "_" + get_now() + ".pdf"
    else:
        outfilename = "Report_" + get_now() + ".pdf"

    ReportAggregator = ReportAggregator(outfilename, outputdir, args.user, args.memo, args.leverage, args.system, InputList)
    ReportAggregator.report()
