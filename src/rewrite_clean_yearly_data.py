import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--system', type=str, help='operating system, to deal with different paths',
                        default='microsoft')
    args = parser.parse_args()

    wd = os.path.dirname(os.getcwd())

    # Gold
    """
    if args.system == 'linux':
        datapath = (wd + '/data_raw/GOLD_1800-2019.csv')
    else:
        datapath = (wd + '\data_raw\GOLD_1800-2019.csv')
    df = pd.read_csv(datapath, skiprows=1, header=0, parse_dates=True, date_parser=lambda x: dt.strptime(x, '%Y'),
                     index_col=0)
    df = df.rename(columns={"New York Market Price (U.S. dollars per fine ounce)": "close_USD",
                            "London Market Price (British &pound; [1718-1949] or U.S. $ [1950-2011] per fine ounce)": "close_GBP"})
    df.index = df.index.rename('date')

    df['close_USD'] = df['close_USD'].str.replace(',', '').astype(float)
    df['close_GBP'] = df['close_GBP'].str.replace(',', '').astype(float)

    # Select just the US price
    df = df.drop(['close_GBP'], axis=1)

    df = df.rename(columns={"close_USD": "close"})

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df.loc[:, column] = df["close"]
    df.loc[:, 'volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    # save the modified csv
    df.to_csv(wd + '/modified_data/clean_gld_yearly.csv')
    """

    if args.system == 'linux':
        datapath = (wd + '/data_raw/GOLD_PIKETTY_1850-2011.csv')
    else:
        datapath = (wd + '\data_raw\GOLD_PIKETTY_1850-2011.csv')
    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x: dt.strptime(x, '%Y'),
                     index_col=0)
    df = df.rename(columns={"Price": "close"}, index={"Year": "Date"})
    df.index = df.index.rename('Date')

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df.loc[:, column] = df["close"]
    df.loc[:, 'volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    # save the modified csv
    df.to_csv(wd + '/modified_data/GLD_LNG.csv')

    # Oil
    if args.system == 'linux':
        datapath = (wd + '/data_raw/F000000__3a.xls')
    else:
        datapath = (wd + '\data_raw\F000000__3a.xls')

    df = pd.read_excel(datapath, sheet_name='Data 1', skiprows=2, header=0, index_col=0)
    df = df.rename(columns={"U.S. Crude Oil First Purchase Price (Dollars per Barrel)": "close"})
    df.index = df.index.rename('Date')
    df.index = df.index - pd.offsets.YearBegin(1) # assign the yearly price to year start. The yearly price is
    # actually an average of the price over a year (with monthly frequency)

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df.loc[:, column] = df["close"]
    df.loc[:, 'volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    # save the modified csv
    df.to_csv(wd + '/modified_data/OIL_LNG.csv')

    # Gov bond 10 years (from https://stooq.com/q/d/?s=10usy.b)
    if args.system == 'linux':
        datapath = (wd + '/data_raw/10usy_b_y.csv')
    else:
        datapath = (wd + '\\data_raw\\10usy_b_y.csv')

    df = pd.read_csv(datapath, skiprows=0, header=0, parse_dates=True, date_parser=lambda x: dt.strptime(x, '%d/%m/%Y'),
                     index_col=0)
    df = df.rename(columns={"Close": "yield"}, index={"Date": "Date"})
    df.index = df.index.rename('Date')
    df.index = df.index + pd.Timedelta(days=1)
    df = df[["yield"]]

    total_return = bond_total_return(ytm = df[['yield']], dt = 1, maturity = 10)
    df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
    df["close"].iloc[0] = 100
    df = df.dropna()

    # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
    for column in ["open", "high", "low"]:
        df[column] = df["close"]
    df['volume'] = 0
    df = df[["open", "high", "low", "close", "volume"]]

    # save the modified csv
    df.to_csv(wd+'/modified_data/US10YB_LNG.csv')

    # Equity, Housing, Gov Bond
    if args.system == 'linux':
        datapath = (wd + '/data_raw/JSTdatasetR4.xlsx')
    else:
        datapath = (wd + '\data_raw\JSTdatasetR4.xlsx')

    df = pd.read_excel(datapath, sheet_name='Data', skiprows=0, header=0, index_col=0,
                       parse_dates=True, date_parser=lambda x: dt.strptime(x, '%Y'))
    df.index = df.index.rename('Date')
    # For now take only US data
    df = df[df["iso"] == "USA"]

    # Get the following metrics:
    # 1. Equity total return, nominal. r[t] = [[p[t] + d[t]] / p[t-1] ] - 1 (column eq_tr)
    # 2. Housing total return, nominal. r[t] = [[p[t] + d[t]] / p[t-1] ] - 1 (column housing_tr)
    # 3. Government bond total return, nominal. r[t] = [[p[t] + coupon[t]] / p[t-1] ] - 1 (column bond_tr)
    # 4. Bill rate, nominal. r[t] = coupon[t] / p[t-1] (column bill_rate)
    P0 = 0.01

    outname = ['EQ_LNG', 'RE_LNG', 'LTB_LNG', 'ITB_LNG']
    i = 0
    for asset in ["eq_tr", "housing_tr", "bond_tr", "bill_rate"]:
        data = df[[asset]]
        data = data.dropna()
        # 1. reconstruct prices from returns, P0 = 0.01
        data['close'] = P0 * np.cumprod(1 + data[asset])
        data = data.drop([asset], axis=1)

        # Add columns open, high, low and set them  equal to close. Add column volume and set it equal to 0
        for column in ["open", "high", "low"]:
            data.loc[:, column] = data["close"]
        data.loc[:, 'volume'] = 0
        data = data[["open", "high", "low", "close", "volume"]]
        # save the modified csv
        data.to_csv(wd + '/modified_data/' + outname[i] + '.csv')
        i = i + 1
