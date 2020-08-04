import pandas as pd
import pandas_datareader.data as web
import datetime
from datetime import datetime as dt
import backtrader as bt
import backtrader.feeds as btfeeds
import numpy as np
import os
import argparse
from utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--system',  type=str, help='operating system, to deal with different paths', default='microsoft')
    args = parser.parse_args()

    
    wd = os.path.dirname(os.getcwd())

    if args.system == 'linux':
        datapath = (wd + '/data_raw/GOLD_1800-2019.csv')
    else:
        datapath = (wd + '\data_raw\GOLD_1800-2019.csv')
    df = pd.read_csv(datapath, skiprows=1, header=0, parse_dates=True, date_parser=lambda x:dt.strptime(x, '%Y'), index_col=0)
    df = df.rename(columns={"New York Market Price (U.S. dollars per fine ounce)":"close_USD",
                            "London Market Price (British &pound; [1718-1949] or U.S. $ [1950-2011] per fine ounce)":"close_GBP"})
    df.index = df.index.rename('date')

    df['close_USD'] = df['close_USD'].str.replace(',', '').astype(float)
    df['close_GBP'] = df['close_GBP'].str.replace(',', '').astype(float)

    # Select just the US price
    df = df.drop(['close_GBP'], axis=1)

    df = df.rename(columns={"close_USD": "close"})

    # Go from yearly from day frequency. Fill the gaps with the previous value.

#    r = pd.date_range(start=df.index.min(), end=df.index.max())
#    df = df.reindex(r)
#    df = df.fillna(method='backfill')

    # Fill the gap with a brownian bridge
    data_filled = pd.DataFrame(columns=['close'])

    # First calculate the yearly standard deviation
    logret = np.log(df.close) - np.log(df.close.shift(1))
    sigma = logret.std()

    for j in range(0, len(df)-1): # loop on the yearly prices
        a = df.close.iloc[j]
        b = df.close.iloc[j+1]
        r = pd.date_range(start=df.index[j], end=df.index[j+1])
        N = len(r)
        sigma_day = sigma/np.sqrt(N) # daily standard deviation

        B = brownian_bridge(N, a, b)
        BB = pd.DataFrame(data=B.T, columns=['close'], index=r)
        logret = np.log(BB.close) - np.log(BB.close.shift(1))
        thisstd = logret.std()
        scaling = sigma_day / thisstd
        # Scale the returns to have the desired standard deviation
        logret_scaled = (logret-logret.mean())*scaling+logret.mean()
        BB_scaled = BB.close.iloc[0] * np.exp(np.cumsum(logret_scaled))
        BB_scaled = pd.DataFrame(data=BB_scaled, columns=['close'], index=r)
        BB_scaled.close.iloc[0] = a
        #B_df = pd.DataFrame(B.T, columns=['close'], index=r)
        data_filled = data_filled.append(BB_scaled)
    df = data_filled
    df = df.drop_duplicates()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df.loc[:, column] = df["close"]
    df.loc[:, 'volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_gld_yearly.csv')

    if args.system == 'linux':
        datapath = (wd + '/data_raw/F000000__3a.xls')
    else:
        datapath = (wd + '\data_raw\F000000__3a.xls')

    df = pd.read_excel(datapath, sheet_name='Data 1', skiprows=2, header=0, index_col=0)
    df = df.rename(columns={"U.S. Crude Oil First Purchase Price (Dollars per Barrel)": "close"})
    df.index = df.index.rename('date')

    # Go from yearly from day frequency. Fill the gaps with the previous value.
    #r = pd.date_range(start=df.index.min(), end=df.index.max())
    #df = df.reindex(r)
    #df = df.fillna(method='backfill')

    # Fill the gap with a brownian bridge
    data_filled = pd.DataFrame(columns=['close'])

    # First calculate the yearly standard deviation
    logret = np.log(df.close) - np.log(df.close.shift(1))
    sigma = logret.std()

    for j in range(0, len(df)-1): # loop on the yearly prices
        a = df.close.iloc[j]
        b = df.close.iloc[j+1]
        r = pd.date_range(start=df.index[j], end=df.index[j+1])
        N = len(r)
        sigma_day = sigma/np.sqrt(N) # daily standard deviation

        B = brownian_bridge(N, a, b)
        BB = pd.DataFrame(data=B.T, columns=['close'], index=r)
        logret = np.log(BB.close) - np.log(BB.close.shift(1))
        thisstd = logret.std()
        scaling = sigma_day / thisstd
        # Scale the returns to have the desired standard deviation
        logret_scaled = (logret-logret.mean())*scaling+logret.mean()
        BB_scaled = BB.close.iloc[0] * np.exp(np.cumsum(logret_scaled))
        BB_scaled = pd.DataFrame(data=BB_scaled, columns=['close'], index=r)
        BB_scaled.close.iloc[0] = a
        #B_df = pd.DataFrame(B.T, columns=['close'], index=r)
        data_filled = data_filled.append(BB_scaled)
    df = data_filled
    df = df.drop_duplicates()


    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df.loc[:, column] = df["close"]
    df.loc[:, 'volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    # save the modified csv
    df.to_csv(wd+'/modified_data/clean_oil_yearly.csv')

    if args.system == 'linux':
        datapath = (wd + '/data_raw/JSTdatasetR4.xlsx')
    else:
        datapath = (wd + '\data_raw\JSTdatasetR4.xlsx')

    df = pd.read_excel(datapath, sheet_name='Data', skiprows=0, header=0, index_col=0,
                       parse_dates=True, date_parser=lambda x:dt.strptime(x, '%Y'))

    # For now take only US data
    df = df[df["iso"] == "USA"]

    # Get the following metrics:
    # 1. Equity total return, nominal. r[t] = [[p[t] + d[t]] / p[t-1] ] - 1 (column eq_tr)
    # 2. Housing total return, nominal. r[t] = [[p[t] + d[t]] / p[t-1] ] - 1 (column housing_tr)
    # 3. Government bond total return, nominal. r[t] = [[p[t] + coupon[t]] / p[t-1] ] - 1 (column bond_tr)
    # 4. Bill rate, nominal. r[t] = coupon[t] / p[t-1] (column bill_rate)
    P0 = 100

    outname = ['clean_equity_yearly', 'clean_housing_yearly', 'clean_bond_yearly', 'clean_bill_yearly']
    i = 0
    for asset in ["eq_tr", "housing_tr", "bond_tr", "bill_rate"]:
        data = df[[asset]]
        data = data.dropna()
        # 1. reconstruct prices from returns, P0 = 100
        if asset != 'bill_rate': # for bill rate the formula is different
            data['close'] = P0 * np.cumprod(1 + data[asset])
        elif asset == 'bill_rate':
            data['close'] = P0 / (1 + data[asset])
        data = data.drop([asset], axis=1)

        # 2. go from yearly to day frequency and backfill prices
        #r = pd.date_range(start=data.index.min(), end=data.index.max())
        #data = data.reindex(r)
        #data = data.fillna(method='backfill')

        # Fill the gap with a brownian bridge
        data_filled = pd.DataFrame(columns=['close'])

        # First calculate the yearly standard deviation
        logret = np.log(data.close) - np.log(data.close.shift(1))
        sigma = logret.std()

        for j in range(0, len(data) - 1):  # loop on the yearly prices
            a = data.close.iloc[j]
            b = data.close.iloc[j + 1]
            r = pd.date_range(start=data.index[j], end=data.index[j + 1])
            N = len(r)
            sigma_day = sigma / np.sqrt(N)  # daily standard deviation

            B = brownian_bridge(N, a, b)
            BB = pd.DataFrame(data=B.T, columns=['close'], index=r)
            logret = np.log(BB.close) - np.log(BB.close.shift(1))
            thisstd = logret.std()
            scaling = sigma_day / thisstd
            # Scale the returns to have the desired standard deviation
            logret_scaled = (logret - logret.mean()) * scaling + logret.mean()
            BB_scaled = BB.close.iloc[0] * np.exp(np.cumsum(logret_scaled))
            BB_scaled = pd.DataFrame(data=BB_scaled, columns=['close'], index=r)
            BB_scaled.close.iloc[0] = a
            # B_df = pd.DataFrame(B.T, columns=['close'], index=r)
            data_filled = data_filled.append(BB_scaled)
        data = data_filled
        data = data.drop_duplicates()

        # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
        for column in ["open", "high", "low"]:
            data.loc[:, column] = data["close"]
        data.loc[:, 'volume'] = 0
        data = data[["open", "high", "low", "close", "volume"]]
        # save the modified csv
        data.to_csv(wd+'/modified_data/'+outname[i]+'.csv')
        i = i+1
