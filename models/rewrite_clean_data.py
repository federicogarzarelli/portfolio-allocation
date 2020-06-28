import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import numpy as np
import os
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--system',  type=str, help='operating system, to deal with different paths', default='microsoft')
    args = parser.parse_args()

    
    wd = os.path.dirname(os.getcwd())

    if args.system == 'linux':
        datapath = (wd + '/data/Gold.csv')
    else:
        datapath = (wd + '\data\Gold.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    df = df.rename(columns={"Gold USD": "close"}, index={'Date': 'date'})
    df['close'] = df['close'].str.replace(',', '').astype(float)

    df = df[['close']]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_gld.csv')

    
    if args.system == 'linux':
        datapath = (wd + '/data/^GSPC.csv')
    else:
        datapath = (wd + '\data\^GSPC.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Adj Close": "close"}, index={'Date': 'date'})
    df = df[['close']]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_gspc.csv')


    if args.system == 'linux':
        datapath = (wd + '/data/SPGSCITR_IND.csv')
    else:
        datapath = (wd + '\data\SPGSCITR_IND.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Close": "close"}, index={'Date': 'date'})
    df['close'] = df['close'].str.replace(',', '').astype(float)
    # Eliminate the flash crashes between Apr 08 - Jul 08 from the data
    df['log_ret'] = (np.log(df.close) - np.log(df.close.shift(1)))
    df = df.dropna()
    df = df[(abs(df['log_ret']) < 1)] # filter out the flash crash!
    df['close'] = df['close'].iloc[0]*np.exp(np.cumsum(df['log_ret']))
    df = df[['close']]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_spgscitr.csv')


    
    if args.system == 'linux':
        datapath = (wd + '/data/^TYX.csv')
    else:
        datapath = (wd + '\data\^TYX.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    df = df.rename(columns={"Adj Close": "yield"}, index={'Date': 'date'})
    df = df[df['yield'] != 'null']
    df['close'] =  100/np.power(1+df['yield']/100,30)
    df = df.dropna()
    df = df[['close']]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_tyx.csv')



    if args.system == 'linux':
        datapath = (wd + '/data/^FVX.csv')
    else:
        datapath = (wd + '\data\^FVX.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Adj Close": "yield"}, index={'Date': 'date'})
    df = df[df['yield'] != 'null']
    df['close'] =  100/np.power(1+df['yield']/100,5)
    df = df.dropna()

    df = df[['close']]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_fvx.csv')


    if args.system == 'linux':
        datapath = (wd + '/data/DFII10.csv')
    else:
        datapath = (wd + '\data\DFII10.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"DFII10": "yield"}, index={'DATE': 'date'})
    df = df[df['yield'] != '.']
    df['yield'] = df['yield'].astype(float)
    df['close'] =  100/np.power(1+df['yield']/100,10)
    df = df.dropna()
    df = df[['close']]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_dfii10.csv')
