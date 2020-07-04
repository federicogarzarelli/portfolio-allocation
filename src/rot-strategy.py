import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from subprocess import call

path = '/home/newuser/Desktop/AlgoTrading/Research/'

def get_today():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')


def load_economic_curves(start, end):
    list_fundamental = ['T10YIE','DFII20','T10Y2Y']
    df_fundamental = web.DataReader(list_fundamental,"fred", start=start, end=end)
    df_fundamental = df_fundamental.dropna()
    df_fundamental['T10YIE_T10Y2Y'] = df_fundamental['T10YIE'] - df_fundamental['T10Y2Y'] 
    df_fundamental = df_fundamental.drop(['T10YIE'], axis=1)
    df_fundamental = df_fundamental.rename(columns={'T10YIE_T10Y2Y':'GLD', 'DFII20':'TLT', 'T10Y2Y': 'SPY',
                                                    'DATE':'Date'})
    df_fundamental['Max'] = df_fundamental.idxmax(axis=1)
    df_fundamental.index.name = 'Date'
    df_fundamental.plot()
    plt.savefig(path+'results/drivers.png')
    return df_fundamental[['Max']]


def load_shares(start, end):
    shares = "SPY TLT GLD"
    data = yf.download(shares, start=start, end=end, group_by='ticker',auto_adjust=True)
    data = data.dropna()
    data = data.iloc[:,data.columns.get_level_values(1) == 'Close']
    data = data.pct_change()
    return data.dropna()



if __name__ == '__main__':
    start = '2005-01-01'
    end = get_today()
    dff = load_economic_curves(start=start, end=end) # the two are not in a one to one correspondance
    dfs = load_shares(start, end)
    df_final = pd.merge(dff, dfs, left_index=True, right_index=True)
    df_final = df_final.rename(columns={("GLD",'Close'):'GLD',("TLT",'Close'):'TLT',("SPY",'Close'):'SPY'})
    df_final['rotret'] = df_final.lookup(df_final.index, df_final['Max'])
    df_final['CumRot'] = (1+df_final.rotret).cumprod()
    df_final['CumGLD'] = (1+df_final.GLD).cumprod()
    df_final['CumSPY'] = (1+df_final.SPY).cumprod()
    df_final['CumTLT'] = (1+df_final.TLT).cumprod()
    df_final[['CumRot','CumGLD','CumSPY','CumTLT']].plot()
    plt.show()
    plt.savefig(path+'results/results.png')
    
    df_final.to_csv(path+'results/rot_strat.csv')

    f1 = open(path+'results/rot_strat.csv')
    last_line = f1.readlines()[-1]
    f1.close()
    
