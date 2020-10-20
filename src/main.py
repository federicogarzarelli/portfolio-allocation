import os
from GLOBAL_VARS import *
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro
from report_aggregator import ReportAggregator

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
    parser.add_argument('--initial_cash', type=int, default=100000, required=False, help='initial_cash to start with')
    parser.add_argument('--monthly_cash', type=float, default=10000, required=False, help='monthly cash invested')
    parser.add_argument('--strategy', type=str, required=False,
                        help='Specify the strategies for which a backtest is run')
    parser.add_argument('--startdate', type=str, default='2017-01-01', required=False,
                        help='starting date of the simulation')
    parser.add_argument('--enddate', type=str, default=now, required=False, help='end date of the simulation')
    parser.add_argument('--system', type=str, default='windows', help='operating system, to deal with different paths')
    parser.add_argument('--leverage', type=int, default=1, help='leverage to consider')
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
        # cerebro.broker.set_shortcash(True) # Can short the cash

        # Import the historical assets
        shares_list = args.shares.split(',') # GLD,COM,SP500,LTB,ITB,TIP
        for share in shares_list:
            df = import_process_hist(share, args)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_leverage(df[column], leverage=args.leverage, expense_ratio=0.0, timeframe=timeframe)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))

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
                df[column] = add_leverage(df[column], leverage=args.leverage, expense_ratio=0.0, timeframe=timeframe)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))

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
        assets_dic = {}
        for i in range(len(shares_list)):
            assets_dic[shares_list[i]] = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
            assets_dic[shares_list[i]] = add_leverage(assets_dic[shares_list[i]], leverage=args.leverage,
                                                      expense_ratio=0.0,timeframe=timeframe).to_frame("close")

            for column in ['open', 'high', 'low']:
                assets_dic[shares_list[i]][column] = assets_dic[shares_list[i]]['close']

            assets_dic[shares_list[i]]['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=assets_dic[shares_list[i]], fromdate=startdate, todate=enddate,
                                            timeframe=timeframe))

        shareclass = args.shareclass.split(',')

    # Set the minimum periods, lookback period (and other parameters) depending on the data used (daily or yearly) and
    # if weights are used instead of a strategy
    if args.weights != '' or strategy is None:
        strat_params = strat_params_weights
    elif timeframe == bt.TimeFrame.Days:
        strat_params = strat_params_days
    elif timeframe == bt.TimeFrame.Years:
        strat_params = strat_params_years

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
            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))

        shareclass = shareclass + ['non-tradable', 'non-tradable', 'non-tradable']
        shares_list = shares_list + indicatorLabels

    i = 0
    for dt in data:
        cerebro.adddata(dt, name=shares_list[i])
        i = i + 1

    n_assets = len([x for x in shareclass if x != 'non-tradable'])
    cerebro.addobserver(targetweightsobserver, n_assets=n_assets)
    cerebro.addobserver(effectiveweightsobserver, n_assets=n_assets)

    # if you provide the weights, use them
    if args.weights != '' or strategy is None:
        weights_list = args.weights.split(',')
        weights_listt = [float(i) for i in weights_list]

        cerebro.addstrategy(customweights,
                            n_assets=n_assets,
                            initial_cash=args.initial_cash,
                            monthly_cash=args.monthly_cash,
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
                            monthly_cash=args.monthly_cash,
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
        res_prices, res_returns, res_perfdata, res_targetweights, res_effectiveweights, res_params, res_assetprices = cerebro.report(system=args.system)

        return res_prices, res_returns, res_perfdata, res_targetweights, res_effectiveweights, res_params, res_assetprices


if __name__ == '__main__':
    args = parse_args()

    # Clear the output folder
    outputdir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "output")
    delete_in_dir(outputdir)

    print_header(args)

    if args.strategy is None:
        strategy_list = ["Custom Weights"]
    else:
        strategy_list = args.strategy.split(',')

    prices = pd.DataFrame()
    returns = pd.DataFrame()
    perf_data = pd.DataFrame()
    targetweights = pd.DataFrame()
    effectiveweights = pd.DataFrame()
    params = pd.DataFrame()

    for strat in strategy_list:
        print_section_divider(strat)

        ThisStrat_prices, ThisStrat_returns, ThisStrat_perf_data, ThisStrat_targetweight, \
            ThisStrat_effectiveweight, ThisStrat_params, ThisStrat_assetprices = runOneStrat(strat)

        if prices.empty:
            prices = ThisStrat_prices
        else:
            prices[strat] = ThisStrat_prices

        if returns.empty:
            returns = ThisStrat_returns
        else:
            returns[strat] = ThisStrat_returns

        if perf_data.empty:
            perf_data = ThisStrat_perf_data
        else:
            perf_data[strat] = ThisStrat_perf_data

        if targetweights.empty:
            targetweights = ThisStrat_targetweight
        else:
            targetweights[strat] = ThisStrat_targetweight

        if effectiveweights.empty:
            effectiveweights = ThisStrat_effectiveweight
        else:
            effectiveweights[strat] = ThisStrat_effectiveweight

        params = ThisStrat_params
        assetprices = ThisStrat_assetprices

    if args.report_name is not None:
        outfilename = args.report_name + "_" + get_now() + ".pdf"
    else:
        outfilename = "Report_" + get_now() + ".pdf"
    user = args.user
    memo = args.memo

    ReportAggregator = ReportAggregator(outfilename, outputdir, user, memo, args.system, prices, returns,
                                        perf_data, targetweights, effectiveweights, params, assetprices)
    ReportAggregator.report()
