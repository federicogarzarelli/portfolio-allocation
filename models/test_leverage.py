import pandas as pd
from utils import *
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--system', type=str, help='operating system, to deal with different paths',
                        default='microsoft')
    args = parser.parse_args()

    
    asset = 'GLD'
    df = import_process_hist(asset, args)
    leveraged_df = pd.DataFrame()
    leveraged_df['Date'] = df.index
    leveraged_df = leveraged_df.set_index('Date')
    leveraged_df['high'] = add_leverage(df['high'], leverage = 2, expense_ratio=0.0)
    df['high'].plot(label='normal gld')
    leveraged_df['high'].plot(label='leveraged')
    plt.legend()
    plt.show()
