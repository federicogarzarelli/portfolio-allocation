import os
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro


def parse_args():
    now = datetime.datetime.now().strftime("%Y_%m_%d")# string to be used after
    parser = argparse.ArgumentParser(description='main class to run strategies')
    parser.add_argument('--historic', action='store_true', default=False, required=False,help='would you like to use the historical manual data')
    parser.add_argument('--shares', type=str, default='SPY,TLT', required=False,help='string corresponding to list of shares if not using historic')
    parser.add_argument('--shareclass', type=str, default='equity,bond', required=False,help='string corresponding to list of asset classes, in not using historic (needed for static strategies)')
    parser.add_argument('--weights', type=str, default='', required=False,help='string corresponding to list of weights. if no values, risk parity weights are taken')
    parser.add_argument('--indicators', action='store_true', default=False, required=False,help='include indicators for rotational strategy, if true')
    parser.add_argument('--initial_cash', type=int, default=100000, required=False,help='initial_cash to start with')
    parser.add_argument('--monthly_cash', type=float, default=10000, required=False,help='monthly cash invested')
    parser.add_argument('--create_report', action='store_true', default=False, required=False,help='creates a report if true')
    parser.add_argument('--report_name', type=str, default=now, required=False,help='if create_report is True, it is better to have a specific name')
    parser.add_argument('--report_type', type=str, default='OneStrategyPDF', required=False,help='if create_report is True, specify the type of report between OneStrategyPDF, StrategiesComparison')
    parser.add_argument('--strategy', type=str, default='uniform', required=False,help='if report_type = OneStrategyPDF, specify the strategy')
    parser.add_argument('--startdate', type=str, default='2017-01-01', required=True,help='starting date of the simulation')
    parser.add_argument('--enddate', type=str, default=now, required=True,help='end date of the simulation')
    parser.add_argument('--system', type=str, default='windows', help='operating system, to deal with different paths')
    parser.add_argument('--leverage', type=int, default=1, help='leverage to consider')
    
    return parser.parse_args()

if __name__=='__main__':

    args = parse_args()

    startdate = datetime.datetime.strptime(args.startdate,"%Y-%m-%d")
    enddate = datetime.datetime.strptime(args.enddate,"%Y-%m-%d")

    # Initialize the engine
    cerebro = Cerebro()
    cerebro.broker.set_cash(args.initial_cash)
    cerebro.broker.set_checksubmit(checksubmit=False)  # Do not check if there is enough margin or cash before executing the order
    cerebro.broker.set_shortcash(True) # Can short the cash

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
            #assets_dic[shares_list[i]] = assets_dic[shares_list[i]].rename(columns={"High":"high", "Low":"low","Open":"open", "Adj Close":"close"})
            #assets_dic[shares_list[i]] = assets_dic[shares_list[i]].drop(columns=[0,"Close"])

            for column in ['open', 'high', 'low']:
                assets_dic[shares_list[i]][column] = assets_dic[shares_list[i]]['close']
                
            assets_dic[shares_list[i]]['volume'] = 0

            
            #print(assets_dic[shares_list[0]])
            #break
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
            df = df[['open', 'high', 'low', 'close','volume']]
            data.append(
                bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))

        shareclass = shareclass+['non-tradable', 'non-tradable', 'non-tradable']

    for dt in data:
        cerebro.adddata(dt)

    # Add the strategy
    n_assets = len([x for x in shareclass if x != 'non-tradable'])
    # if you provide the weights, use them
    if args.weights != '':
        weights_list = args.weights.split(',')
        weights_listt = [float(i) for i in weights_list]

        cerebro.addstrategy(customweights, n_assets=n_assets,
                            monthly_cash=args.monthly_cash,
                            assetweights=weights_listt,
                            shareclass=shareclass
                            )
    # otherwise, rely on the weights of a strategy
    else:
        cerebro.addstrategy(eval(args.strategy), n_assets=n_assets,
                            monthly_cash=args.monthly_cash,
                            shareclass=shareclass)

    # Run and create report
    cerebro.run()
    cerebro.plot(volume=False)
        
    if args.create_report:
        cerebro.report('reports/', system=args.system)
        os.rename('reports/report.pdf', 'reports/%s_%s_%s.pdf' %(args.report_name, eval(args.strategy).strategy_name, startdate.isoformat().replace(':', '')))
