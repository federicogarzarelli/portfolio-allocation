import pandas as pd
import numpy as np
import os
import argparse
from utils import bond_total_return
from GLOBAL_VARS import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--system',  type=str, help='operating system, to deal with different paths', default='microsoft')
    args = parser.parse_args()

    
    wd = os.path.dirname(os.getcwd())

    # OLD VERSION OF THE GOLD FILE
    # # Gold
    # if args.system == 'linux':
    #     datapath = (wd + '/data_raw/Gold.csv')
    # else:
    #     datapath = (wd + '\data_raw\Gold.csv')
    # df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    # df = df.rename(columns={"Gold USD": "close"}, index={'Date': 'Date'})
    # df['close'] = df['close'].str.replace(',', '').astype(float)
    # # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    # for column in ["open", "high", "low"]:
    #     df[column] = df["close"]
    # df['volume'] = 0
    # df = df[["open", "high", "low", "close", "volume"]]
    # df.index = df.index.rename('Date')
    # # save the modified csv
    # df.to_csv(wd+'/modified_data/clean_gld.csv')

    # Gold
    if args.system == 'linux':
        datapath = (wd + '/data_raw/GOLDAMGBD228NLBM.csv')
    else:
        datapath = (wd + '\data_raw\GOLDAMGBD228NLBM.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    df = df.rename(columns={"Gold USD": "close"}, index={'Date': 'Date'})
    #df['close'] = df['close'].str.replace(',', '').astype(float)
    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/GLD.csv')

    # S&P 500
    if args.system == 'linux':
        datapath = (wd + '/data_raw/^GSPC.csv')
    else:
        datapath = (wd + '\data_raw\^GSPC.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Adj Close": "close"}, index={'Date': 'Date'})
    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/SP500.csv')

    # S&P 500 Total return
    if args.system == 'linux':
        datapath = (wd + '/data_raw/^SP500TR.csv')
    else:
        datapath = (wd + '\data_raw\^SP500TR.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Adj Close": "close"}, index={'Date': 'Date'})
    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/SP500TR.csv')


    # GSCI Commodity Index
    if args.system == 'linux':
        datapath = (wd + '/data_raw/SPGSCITR_IND.csv')
    else:
        datapath = (wd + '\data_raw\SPGSCITR_IND.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Close": "close"}, index={'Date': 'Date'})
    df['close'] = df['close'].str.replace(',', '').astype(float)
    # Eliminate the flash crashes between Apr 08 - Jul 08 from the data
    df['log_ret'] = (np.log(df.close) - np.log(df.close.shift(1)))
    df = df.dropna()
    df = df[(abs(df['log_ret']) < 1)] # filter out the flash crash!
    df['close'] = df['close'].iloc[0]*np.exp(np.cumsum(df['log_ret']))
    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/COM.csv')

    # 30 years treasury yield
    if args.system == 'linux':
        datapath = (wd + '/data_raw/^TYX.csv')
    else:
        datapath = (wd + '\data_raw\^TYX.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    df = df.rename(columns={"Adj Close": "yield"}, index={'Date': 'Date'})
    df = df[df['yield'] != 'null']
    df = df.dropna()

    total_return = bond_total_return(ytm = df[['yield']], dt = 1/DAYS_IN_YEAR_BOND_PRICE, maturity = 30)
    df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
    df["close"].iloc[0] = 100
    df = df.dropna()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/LTB.csv')

    # 5 years treasury yield
    if args.system == 'linux':
        datapath = (wd + '/data_raw/^FVX.csv')
    else:
        datapath = (wd + '\data_raw\^FVX.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, index_col=0)
    df = df.rename(columns={"Adj Close": "yield"}, index={'Date': 'Date'})
    df = df[df['yield'] != 'null']
    df = df.dropna()

    total_return = bond_total_return(ytm = df[['yield']], dt = 1/DAYS_IN_YEAR_BOND_PRICE, maturity = 5)
    df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
    df["close"].iloc[0] = 100
    df = df.dropna()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/ITB.csv')

    # TIPS 10 years
    if args.system == 'linux':
        datapath = (wd + '/data_raw/DFII10.csv')
    else:
        datapath = (wd + '\data_raw\DFII10.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    df = df.rename(columns={"DFII10": "yield"}, index={'DATE': 'Date'})
    df = df[df['yield'] != '.']
    df['yield'] = df['yield'].astype(float)
    df = df.dropna()

    total_return = bond_total_return(ytm = df[['yield']], dt = 1/DAYS_IN_YEAR_BOND_PRICE, maturity = 10)
    df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
    df["close"].iloc[0] = 100
    df = df.dropna()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/TIP.csv')


    # US Bonds 20 years
    if args.system == 'linux':
        datapath = (wd + '/data_raw/DGS20.csv')
    else:
        datapath = (wd + '\data_raw\DGS20.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x:pd.datetime.strptime(x, '%d/%m/%Y'), index_col=0)
    df = df.rename(columns={"DGS20": "yield"}, index={'DATE': 'Date'})
    df = df[df['yield'] != '.']
    df['yield'] = df['yield'].astype(float)
    df = df.dropna()

    total_return = bond_total_return(ytm = df[['yield']], dt = 1/DAYS_IN_YEAR_BOND_PRICE, maturity = 20)
    df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
    df["close"].iloc[0] = 100
    df = df.dropna()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = df.index.rename('Date')
    # save the modified csv
    df.to_csv(wd+'/modified_data/US20YB.csv')
