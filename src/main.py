import os
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro
from report_aggregator import ReportAggregator

def parse_args():
    now = datetime.datetime.now().strftime("%Y_%m_%d")  # string to be used after
    parser = argparse.ArgumentParser(description='main class to run strategies')
    parser.add_argument('--historic', action='store_true', default=False, required=False,
                        help='would you like to use the historical manual data')
    parser.add_argument('--shares', type=str, default='SPY,TLT', required=False,
                        help='string corresponding to list of shares if not using historic')
    parser.add_argument('--shareclass', type=str, default='equity,bond', required=False,
                        help='string corresponding to list of asset classes, in not using historic (needed for static strategies)')
    parser.add_argument('--weights', type=str, default='', required=False,
                        help='string corresponding to list of weights. if no values, risk parity weights are taken')
    parser.add_argument('--indicators', action='store_true', default=False, required=False,
                        help='include indicators for rotational strategy, if true')
    parser.add_argument('--initial_cash', type=int, default=100000, required=False, help='initial_cash to start with')
    parser.add_argument('--monthly_cash', type=float, default=10000, required=False, help='monthly cash invested')
    parser.add_argument('--create_report', action='store_true', default=False, required=False,
                        help='creates a report if true')
    parser.add_argument('--report_name', type=str, default='Testf_Report', required=False,
                        help='if create_report is True, it is better to have a specific name')
    parser.add_argument('--strategy', type=str, required=False,
                        help='Specify the strategies for which a backtest is run')
    parser.add_argument('--startdate', type=str, default='2017-01-01', required=False,
                        help='starting date of the simulation')
    parser.add_argument('--enddate', type=str, default=now, required=False, help='end date of the simulation')
    parser.add_argument('--system', type=str, default='windows', help='operating system, to deal with different paths')
    parser.add_argument('--leverage', type=int, default=1, help='leverage to consider')

    return parser.parse_args()

def runOneStrat(strategy=None):
    args = parse_args()
    if strategy is None:
        strategy = args.strategy

    startdate = datetime.datetime.strptime(args.startdate, "%Y-%m-%d")
    enddate = datetime.datetime.strptime(args.enddate, "%Y-%m-%d")

    # Initialize the engine
    cerebro = Cerebro()
    cerebro.broker.set_cash(args.initial_cash)
    # cerebro.broker.set_checksubmit(checksubmit=False)  # Do not check if there is enough margin or cash before executing the order
    # cerebro.broker.set_shortcash(True) # Can short the cash

    # Add the data
    data = []
    if args.historic:

        # Import the historical assets
        assetLabels = ['GLD', 'COM', 'SP500', 'LTB', 'ITB']
        for assetLabel in assetLabels:
            df = import_process_hist(assetLabel, args)
            for column in ['open', 'high', 'low', 'close']:
                df[column]=add_leverage(df[column], leverage=args.leverage, expense_ratio=0.0)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))

        shareclass = ['gold', 'commodity', 'equity', 'bond_lt', 'bond_it']

    else:
        shares_list = args.shares.split(',')

        # download the datas
        assets_dic = {}
        for i in range(len(shares_list)):
            assets_dic[shares_list[i]] = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"] # might not always work
            assets_dic[shares_list[i]] = add_leverage(assets_dic[shares_list[i]], leverage=args.leverage, expense_ratio=0.0).to_frame("close")

            for column in ['open', 'high', 'low']:
                assets_dic[shares_list[i]][column] = assets_dic[shares_list[i]]['close']
                
            assets_dic[shares_list[i]]['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=assets_dic[shares_list[i]], fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))

        shareclass = args.shareclass.split(',')

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
            data.append(
                bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))

        shareclass = shareclass+['non-tradable', 'non-tradable', 'non-tradable']

    for dt in data:
        cerebro.adddata(dt)

    n_assets = len([x for x in shareclass if x != 'non-tradable'])
    cerebro.addobserver(WeightsObserver, n_assets=n_assets)

    # if you provide the weights, use them
    if args.weights != '' or strategy is None:
        weights_list = args.weights.split(',')
        weights_listt = [float(i) for i in weights_list]

        cerebro.addstrategy(customweights, n_assets=n_assets,
                            monthly_cash=args.monthly_cash,
                            assetweights=weights_listt,
                            shareclass=shareclass
                            )
    # otherwise, rely on the weights of a strategy
    else:
        """
        NOT WORKING. IT WOULD BE A BETTER SOLUTION.
        cerebro.optstrategy(StFetcher, idx=StFetcher.COUNT())
        results = cerebro.run()
        print(results)
        """
        strategy.split(',')
        cerebro.addstrategy(eval(strategy), n_assets=n_assets,
                    monthly_cash=args.monthly_cash,
                    shareclass=shareclass)
        # Run backtest
        cerebro.run()
        cerebro.plot(volume=False)

        # Create report
        if args.create_report:
            prices, returns, perf_data, weight = cerebro.report('reports/', system=args.system)
            if strategy is None:
                strat_name = "StratNotSpecified"
            else:
                strat_name = eval(strategy).strategy_name

            os.rename('reports/report.pdf',
                      'reports/%s_%s_%s.pdf' % (
                      args.report_name, strat_name, startdate.isoformat().replace(':', '')))
            return prices, returns, perf_data, weight

if __name__=='__main__':
    args = parse_args()

    # Clear the output folder
    outputdir = 'reports/'
    delete_in_dir(outputdir)

    print_header(args)
    if args.strategy is None:
        print_section_divider(args.strategy)
        runOneStrat()
    else:
        strategy_list = args.strategy.split(',')
        if len(strategy_list) == 1:
            print_section_divider(args.strategy)

            runOneStrat()
        elif len(strategy_list) > 1:

            prices = pd.DataFrame()
            returns = pd.DataFrame()
            perf_data = pd.DataFrame()
            weight = pd.DataFrame()
            for strat in strategy_list:
                print_section_divider(strat)

                ThisStrat_prices, ThisStrat_returns, ThisStrat_perf_data, ThisStrat_weight = runOneStrat(strat)
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

                if weight.empty:
                    weight = ThisStrat_weight
                else:
                    weight[strat] = ThisStrat_weight

            outfilename = "Aggregated_Report.pdf"
            user = "Fabio & Federico"
            memo = "Testing - Report comparing different strategies"

            ReportAggregator = ReportAggregator(outfilename, outputdir, user, memo, args.system, prices, returns, perf_data, weight)
            ReportAggregator.report()