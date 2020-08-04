import os
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro
from report_aggregator import ReportAggregator

# Strategy parameters not passed
from strategies import customweights

# Set the strategy parameters
strat_params_days = {
    'reb_days': 30,  # every month: we rebalance the portfolio
    'lookback_period_short': 30,  # period to calculate the variance
    'lookback_period_long': 180,  # period to calculate the correlation
    'printlog': True,
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
}

strat_params_years = {
    'reb_days': 1,  # rebalance the portfolio every year
    'lookback_period_short': 2,  # period to calculate the variance (Minimum 2)
    'lookback_period_long': 2,  # period to calculate the correlation (Minimum 2)
    'printlog': True,
    'corrmethod': 'pearson'  # 'spearman' # method for the calculation of the correlation matrix
    }


def parse_args():
    now = datetime.datetime.now().strftime("%Y_%m_%d")  # string to be used after
    parser = argparse.ArgumentParser(description='main class to run strategies')
    parser.add_argument('--historic', type=str, default=None, required=False,
                        help='would you like to use the historical manual data. Long for yearly data from 1900, medium for daily from the 1970')
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
    parser.add_argument('--report_name', type=str, default='Test_Report', required=False,
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
        shares_list = ['GLD', 'COM', 'SP500', 'LTB', 'ITB']
        for share in shares_list:
            df = import_process_hist(share, args)
            for column in ['open', 'high', 'low', 'close']:
                df[column]=add_leverage(df[column], leverage=args.leverage, expense_ratio=0.0)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))

        shareclass = ['gold', 'commodity', 'equity', 'bond_lt', 'bond_it']

    elif args.historic == 'long':
        timeframe = bt.TimeFrame.Years

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        # cerebro = Cerebro()
        # cerebro.broker.set_coo(True)
        # cerebro.broker.set_coc(True)
        cerebro.broker.set_cash(args.initial_cash)
        # cerebro.broker.set_shortcash(True) # Can short the cash


        # Import the historical assets
        shares_list = ['GLD_LNG', 'OIL_LNG', 'EQ_LNG', 'LTB_LNG', 'ITB_LNG']
        for share in shares_list:
            df = import_process_hist(share, args)
            for column in ['open', 'high', 'low', 'close']:
                df[column]=add_leverage(df[column], leverage=args.leverage, expense_ratio=0.0)

            for column in ['open', 'high', 'low']:
                df[column] = df['close']

            df['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=timeframe))

        shareclass = ['gold', 'commodity', 'equity', 'bond_lt', 'bond_it']

    else:
        shares_list = args.shares.split(',')
        timeframe = bt.TimeFrame.Days

        # Initialize the engine
        cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
        # cerebro = Cerebro()
        # cerebro.broker.set_coo(True)
        # cerebro.broker.set_coc(True)
        cerebro.broker.set_cash(args.initial_cash)
        # cerebro.broker.set_shortcash(True) # Can short the cash

        # download the datas
        assets_dic = {}
        for i in range(len(shares_list)):
            assets_dic[shares_list[i]] = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"] # might not always work
            assets_dic[shares_list[i]] = add_leverage(assets_dic[shares_list[i]], leverage=args.leverage, expense_ratio=0.0).to_frame("close")

            for column in ['open', 'high', 'low']:
                assets_dic[shares_list[i]][column] = assets_dic[shares_list[i]]['close']
                
            assets_dic[shares_list[i]]['volume'] = 0

            data.append(bt.feeds.PandasData(dataname=assets_dic[shares_list[i]], fromdate=startdate, todate=enddate, timeframe=timeframe))

        shareclass = args.shareclass.split(',')

    if timeframe == bt.TimeFrame.Days:
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
            data.append(
                bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))

        shareclass = shareclass+['non-tradable', 'non-tradable', 'non-tradable']
        shares_list = shares_list+indicatorLabels

    i = 0
    for dt in data:
        cerebro.adddata(dt, name=shares_list[i])
        i = i+1

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
        """
        NOT WORKING. IT WOULD BE A BETTER SOLUTION.
        cerebro.optstrategy(StFetcher, idx=StFetcher.COUNT())
        results = cerebro.run()
        print(results)
        """
        strategy.split(',')
        cerebro.addstrategy(eval(strategy),
                            n_assets=n_assets,
                            initial_cash=args.initial_cash,
                            monthly_cash=args.monthly_cash,
                            shareclass=shareclass,
                            printlog = strat_params.get('printlog'),
                            corrmethod = strat_params.get('corrmethod'),
                            reb_days = strat_params.get('reb_days'),
                            lookback_period_short = strat_params.get('lookback_period_short'),
                            lookback_period_long = strat_params.get('lookback_period_long')
                           )


    # Run backtest
    cerebro.run()
    cerebro.plot(volume=False)

    # Create report
    if args.create_report:
        outputdir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "output")
        outfile = os.path.join(outputdir, args.report_name)

        prices, returns, perf_data, targetweights, effectiveweights = cerebro.report(outfile, system=args.system)
        return prices, returns, perf_data, targetweights, effectiveweights

if __name__=='__main__':
    args = parse_args()

    # Clear the output folder
    outputdir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "output")
    delete_in_dir(outputdir)

    print_header(args)
    if args.strategy is None:
        print_section_divider("Custom Weights")
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
            targetweights = pd.DataFrame()
            effectiveweights  = pd.DataFrame()
            for strat in strategy_list:
                print_section_divider(strat)

                ThisStrat_prices, ThisStrat_returns, ThisStrat_perf_data, ThisStrat_targetweight, ThisStrat_effectiveweight = runOneStrat(strat)
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

            outfilename = "Aggregated_Report.pdf"
            user = "Fabio & Federico"
            memo = "Testing - Report comparing different strategies"

            ReportAggregator = ReportAggregator(outfilename, outputdir, user, memo, args.system, prices, returns,
                                                perf_data, targetweights, effectiveweights)
            ReportAggregator.report()