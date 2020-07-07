import os
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
import riskparityportfolio as rp
from report import Cerebro


def parse_args():
    now = datetime.datetime.now().strftime("%Y_%m_%d")# string to be used after
    parser = argparse.ArgumentParser(description='main class to run strategies')
    parser.add_argument('--historic', action='store_true', default=False, required=False,help='would you like to use the historical manual data')
    parser.add_argument('--shares', type=str, default='SPY,TLT', required=False,help='string corresponding to list of shares if not using historic')
    parser.add_argument('--weights', type=str, default='', required=False,help='string corresponding to list of weights. if no values, risk parity weights are taken')
    parser.add_argument('--initial_cash', type=int, default=100000, required=False,help='initial_cash to start with')
    parser.add_argument('--create_report', action='store_true', default=False, required=False,help='creates a report if true')
    parser.add_argument('--report_name', type=str, default=now, required=False,help='if create_report is True, it is better to have a specific name')
    parser.add_argument('--startdate', type=str, default='2017-01-01', required=True,help='starting date of the simulation')
    parser.add_argument('--enddate', type=str, default=now, required=True,help='end date of the simulation')
    parser.add_argument('--system', type=str, default='microsoft', help='operating system, to deal with different paths')
    parser.add_argument('--leverage', type=int, default=1, help='leverage to consider')
    
    return parser.parse_args()



if __name__=='__main__':

    args = parse_args()

    startdate = datetime.datetime.strptime(args.startdate,"%Y-%m-%d")
    enddate = datetime.datetime.strptime(args.enddate,"%Y-%m-%d")
    
    data = []
    if args.historic:
        
        assetLabels = ['GLD','COM', 'SP500', 'LTB', 'ITB']

        params = (('alloc_gld', 0.12),
                  ('alloc_com',0.13),
                  ('alloc_spy',0.25),
                  ('alloc_ltb',0.15),
                  ('alloc_itb',0.45))
        
        for assetLabel in assetLabels:
            df = import_process_hist(assetLabel, args)
            for column in ['open','high', 'low', 'close']:
                df[column]=add_leverage(df[column], leverage=args.leverage, expense_ratio=0.0)

            for column in ['open','high', 'low']:
                df[column] = df['close']
                
            df['volume'] = 0
                
            data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate,timeframe=bt.TimeFrame.Days))
        
    else:
        shares_list = args.shares.split(',')

        # download the datas
        assets_dic = {}
        for i in range(len(shares_list)):
            assets_dic[shares_list[i]] = web.DataReader(shares_list[i],"yahoo",startdate, enddate)["Adj Close"] # might not always work
            assets_dic[shares_list[i]] = add_leverage(assets_dic[shares_list[i]], leverage=args.leverage, expense_ratio=0.0).to_frame("close")
            #assets_dic[shares_list[i]] = assets_dic[shares_list[i]].rename(columns={"High":"high", "Low":"low","Open":"open", "Adj Close":"close"})
            #assets_dic[shares_list[i]] = assets_dic[shares_list[i]].drop(columns=[0,"Close"])

            for column in ['open','high', 'low']:
                assets_dic[shares_list[i]][column] = assets_dic[shares_list[i]]['close']
                
            assets_dic[shares_list[i]]['volume'] = 0

            
            #print(assets_dic[shares_list[0]])
            #break
            data.append(bt.feeds.PandasData(dataname=assets_dic[shares_list[i]],fromdate=startdate, todate=enddate,timeframe=bt.TimeFrame.Days))
        
        # if you provide the weights, use them
        if args.weights != '':
            weights_list = args.weights.split(',')
            weights_listt = [int(i) for i in weights_list]
              
        # otherwise, calculate the risk parity weights
        else:
            cov = covariances(shares=shares_list, start=startdate,end=enddate)
            target = np.array([1/len(shares_list)]*len(shares_list))
            port = rp.RiskParityPortfolio(covariance=cov, budget=target)
            port.design()
            weights_list = port.weights

            
        # allocate the weights to the right spots
        params = ()
        for i in range(len(shares_list)):
            params += (('alloc_%s' %shares_list[i],weights_list[i]),)



    cerebro = Cerebro()
    cerebro.broker.set_cash(args.initial_cash)

    for dt in data:
        cerebro.adddata(dt)

    cerebro.addstrategy(uniform, n_assets=len(data))
    cerebro.run()
    cerebro.plot(volume=False)
        
    if args.create_report:
        cerebro.report('reports/')
        os.rename('reports/report.pdf', 'reports/%s_%s.pdf' %(args.report_name, startdate))
