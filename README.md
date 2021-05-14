# portfolio-allocation
Backtest portfolio allocation strategies.   

# Install Requirements

A requirements file has been added to be able to install the required libraries. To install them, you need to run:

```bash
pip install -r requirements.txt
```

A virtual environment should normally also be setup.


# Install Backtrader Reports

```bash
$ pip3 install jinja2
$ pip install WeasyPrint
```

or run the bash script **install_reports.sh**

# Usage of main

```bash
python main.py [--historic [medium | long]] [--shares SHARES --shareclass SHARECLASS] [--weights WEIGHTS | --strategy STRATEGY] [--indicators] [--benchmark BENCHMARK]
               [--initial_cash INITIAL_CASH] [--contribution CONTRIBUTION] 
               [--startdate STARTDATE] [--enddate ENDDATE]
               [--system SYSTEM] [--leverage LEVERAGE] 
               [--create_report [--report_name REPORT_NAME] [--user USER] [--memo MEMO]] 
```

## DESCRIPTION
Running main.py with the options below a backtest is performed on the assets specified following a specified strategy. It is recommended to run the code in a jupyter notebook. Jupyter notebooks are available in the 'src' folder.

### Strategies

#### Passive strategies
* __riskparity__ Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run at portfolio level.
* __riskparity_nested__ Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run first at asset classe level (for assets belonging to the same asset class) and then at portfolio level.
* __rotationstrat__ Asset rotation strategy that buy either gold, bonds or equities based on a signal (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy). To use this strategy specify the parameter `--indicators`.
* __uniform__ Static allocation uniform across asset classes. Assets are allocated uniformly within the same asset class.
* __vanillariskparity__ Static allocation to asset classes where weights are taken from https://www.theoptimizingblog.com/leveraged-all-weather-portfolio/ (see section "True Risk Parity").
* __onlystocks__ Static allocation only to the equity class. Assets are allocated uniformly within the equity class.
* __sixtyforty__ Static allocation 60% to the equity class, 20% to the Long Term Bonds class and 20% to the Short Term Bonds class. Assets are allocated uniformly within the asset classes.

#### Semi-passive strategies
* __trend_u__ First weights are assigned according to the "uniform" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash). 
* __absmom_u__ First weights are assigned according to the "uniform" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).
* __relmom_u__ First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "uniform" strategy.
* __momtrend_u__ First weights are assigned according to the "uniform" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash). 
* __trend_rp__ First weights are assigned according to the "riskparity" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).
* __absmom_rp__ First weights are assigned according to the "riskparity" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).
* __relmom_rp__ First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "risk parity" strategy.
* __momtrend_rp__ First weights are assigned according to the "riskparity" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).
* __GEM__ Global equity momentum strategy. Needs only 4 assets of classes equity, equity_intl, bond_lt, money_market. example: `--shares VEU,IVV,BIL,AGG --shareclass equity_intl,equity,money_market,bond_lt`. See https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/
__Note__: the asset classes (`--shareclass` argument) used in the strategies are: Gold, Commodities, Equities, Long Term Bonds, Short Term Bonds (see "OPTIONS" section below). When `--historic` is __not__ specified, every asset after `--shares` must be assigned to one of these 

### Taking into account the minimum period

Please note that the backtest starts after the periods used to calculate the covariance matrix and variance of assets, necessary to compute the weights of `riskparity` and `riskparity_nested` strategies and the periods to calculate the moving average and the momentum.
The minimum period is defined as the maximum between `lookback_period_short`, `lookback_period_long`, `moving_average_period` and `moving_average_period` (see `GLOBAL_VARS.py`)

For example, if the minimum period is 120 days for daily data and to 10 years for yearly data:
* __Years__ i.e. when `--historic` is `long`, if startdate is between "1916-01-02" and "1917-01-01" the backtest starts on the "1926-01-01"
* __Days__ e.g. when `--historic` is `medium` or when `--shares` are specified, if startdate is "1999-01-01" the backtest starts on the "1999-06-18"

## OPTIONS
* `--historic`             use historical asset data, already downloaded manually. Alternative is using assets downloaded automatically from the Yahoo API. If `--historic = "medium"` assets from about 1970 at daily frequency are loaded (`'GLD', 'COM', 'SP500', 'SP500TR', 'LTB', 'ITB','TIP'`). If `--historic = "long"` assets from 1900 at annual frequency are loaded (`'GLD_LNG', 'OIL_LNG', 'EQ_LNG', 'LTB_LNG', 'ITB_LNG', '10YB_LNG', 'RE_LNG''`). The specific assets to be loaded need to be specified after `--shares`.
* `--shares`               if `--historic` is not specified, use downloaded asset data of the tickers specified in a comma separated list (e.g. "SPY,TLT,GLD"). If `--historic` is specified, load asset data of the tickers specified in a comma separated list.
* `--shareclass`           class of each share specified after `--shares` (e.g. `equity,bond_lt,gold`). Possibilities are `equity, bond_lt, bond_it, gold, commodity`, where "bond_lt" and "bond_it" are long and intermediate duration bonds, respectively. __This argument is mandatory when `--historic` is not chosen__
* `--weights`              list of portfolio weights for each share specified after `--shares` (e.g. `0.35,0.35,0.30`). The weights need to sum to 1. When weights are specified a custom weights strategy is used that simply loads the weights specified. Alternative is `--strategy`. __Either this argument or `--strategy` is mandatory__
* `--strategy`             name of one of the strategy to run for the PDF report. Possibilities are `riskparity, riskparity_nested, rotationstrat, uniform, vanillariskparity, onlystocks, sixtyforty`. Alternative is --weights. __Either this argument or `--weights` is mandatory__
* `--indicators`           include the indicator assets (no backtest will be run on these) that are used to decide which assets are used in the strategy. At present these are used only in the asset rotation strategy.  __This argument is mandatory when `--strategy rotationstrat` is chosen__
* `--benchmark`            name of a benchmark to compare against the portfolio allocations. The benchmark is bought using the money available in the portfolio. No leverage is applied to the benchmark. 
* `--initial_cash`         initial_cash to start with. Default is 100000.
* `--contribution`         cash invested or withdrawn in a given year. If the data frequency is daily cash is added or removed on the 20th of each month; if the data frequency is yearly, cash is added or removed each year. If the amount is between (-1,1) the amount is considered to be a % of the portfolio value (e.g. if the amount is -0.04, the 4% of the portfolio is withdrawn). Default is 0.
* `--startdate`            starting date of the simulation. If not specified backtrader will take the earliest possible date. (To test) 
* `--enddate`              end date of the simulation.  If not specified backtrader will take the latest possible date. (To test)
* `--system`               operating system, to deal with different path. Default is windows. If not specified windows is not chosen.
* `--leverage`             leverage to consider. Leverage is applied both with historical (`--historic`) and automatic (`--shares`). data Default is 1. 
* `--create_report`        creates a report if true
* `--report_name`          report name. Default is "Report_DATE". __This argument should be specified only when `--create_report` is chosen__ 
* `--user`                 user generating the report. Default is "Federico & Fabio". __This argument should be specified ony when `--create_report` is chosen__ 
* `--memo`                 description of the report. Default is "Backtest". __This argument should be specified ony when `--create_report` is chosen__ 

### Hidden parameters 
The parameters below are hardcoded in the `GLOBAL_VARS.py` file. 

#### General parameters
* __DAYS_IN_YEAR__ Number of days in a year. Default is 260.
* __DAYS_IN_YEAR_BOND_PRICE__ Number of days in a year used for calculating bond prices from yields. Default is 360.
* __APPLY_LEVERAGE_ON_LIVE_STOCKS__ Flag to apply leverage to downloaded stock prices or not

#### Strategy parameters
* __reb_days__ Number of days (of bars) every which the portfolio is rebalanced. Default is 30 for daily data and 1 for yearly data.
* __lookback_period_short__ Window to calculate the standard deviation of assets returns. Applies to strategy `riskparity` and derived strategies. Default is 20 for daily data and 10 for yearly data. 
* __lookback_period_long__ Window to calculate the correlation matrix of assets returns. Applies to strategies `riskparity` and derived strategies. Default is 120 for daily data and 10 for yearly data.
* __moving_average_period__ Window to calculate simple moving average. Applies to strategies `trend_uniform`, `trend_riskparity`, `momentumtrend_uniform`  and `momentumtrend_riskparity`. Default is 252 for daily data and 5 for yearly data.
* __momentum_period__ Window to calculate the momentum. Applies to strategies `absolutemomentum_uniform`, `relativemomentum_uniform`, `momentumtrend_uniform`, `absolutemomentum_riskparity`, `relativemomentum_riskparity`  and `momentumtrend_riskparity`. Default is 252 for daily data and 5 for yearly data.
* __printlog__ If true a log is output in the terming. Default is True.
* __corrmethod__ Method for the calculation of the correlation matrix. Applies to strategies `riskparity` and `riskparity_nested`. Default is 'pearson'. Alternative is 'spearman'. 
* __momentum_percentile__ Percentile of assets with the highest return in a period to form the relative momentum portfolio. The higher the percentile, the higher the return quantile. 

#### Report parameters
* __riskfree__ Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc. Default is 0.01.
* __targetrate__ Target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio. Default is 0.01.
* __alpha__ Confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio). Default is 0.05.
* __market_mu__ Average annual return of the market, to be used in Treynor ratio, Information ratio. Default is 0.07.
* __market_sigma__ Annual standard deviation of the market, to be used in Treynor ratio, Information ratio. Default is  0.15.
* __fundmode__ Calculate metrics in fund model vs asset mode. Default is True.
* __stddev_sample__ Bessel correction (N-1) when calculating standard deviation from a sample. Default is True.
* __logreturns__ Use logreturns instead of percentage returns when calculating metrics (not recommended). Default is False.
* __annualize__  Calculate annualized metrics by annualizing returns first. Default is True.



## EXAMPLES
1. Historical data, uniform strategy

```bash
python main.py --historic "medium" --shares GLD,COM,SP500,LTB,ITB  --strategy uniform --initial_cash 100000 --contribution 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

2. Historical data, custom weights

```bash
python main.py --historic "medium" --shares GLD,COM,SP500,LTB,ITB --weights "0.2, 0.3, 0.1, 0.1, 0.3" --initial_cash 100000 --contribution 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

 3. Automatically downloaded data, custom weights

```bash
python main.py --shares SPY,IWM,TLT,GLD --shareclass "equity,equity,bond_lt,gold" --weights "0.2, 0.3, 0.1, 0.4" --initial_cash 100000 --contribution 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

4. Automatically downloaded data, 60-40 strategy

```bash
python main.py --shares SPY,IWM,TLT,GLD --shareclass "equity,equity,bond_lt,gold" --strategy sixtyforty --initial_cash 100000 --contribution 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

5. Multiple strategies backtest
```bash
python main.py --shares UPRO,UGLD,TYD,TMF,UTSL --shareclass "equity,gold,bond_it,bond_lt,commodity" --strategy riskparity_nested,riskparity,riskparity_pylib --initial_cash 100000 --contribution 0 --create_report --report_name MyCurrentPortfolio --startdate "2019-01-01" --enddate "2020-06-30" --system windows --leverage 1
```
https://clio-infra.eu/Indicators/LongTermGovernmentBondYield.html

6. GEM
```bash
python main.py --shares VEU,IVV,BIL,AGG --shareclass equity_intl,equity,money_market,bond_lt --strategy GEM --initial_cash 10000000 --contribution 0 --create_report --report_name MyCurrentPortfolio --startdate "2019-01-01" --enddate "2020-06-30" --system windows --leverage 1
```

# Dataset 

Data are stored in a sqlite3 database "myPortfolio.db" with the following structure: 

* __DIM_STOCKS__
   * "NAME"  TEXT NOT NULL,
   * "TICKER"        TEXT,
   * "EXCHANGE"      TEXT,
   * "CURRENCY"      TEXT,
   * "ISIN"  TEXT,
   * "SOURCE"        TEXT,
   * "FREQUENCY"     TEXT,
   * "ASSET_CLASS"   TEXT,
   * "TREATMENT_TYPE"        TEXT,
   * PRIMARY KEY("TICKER")
* __FACT_HISTPRICES__
   * "DATE"  DATETIME NOT NULL,
   * "TICKER"        TEXT NOT NULL,
   * "OPEN"  REAL NOT NULL,
   * "HIGH"  REAL,
   * "LOW"   REAL,
   * "CLOSE" REAL,
   * "VOLUME"        REAL,
   * PRIMARY KEY("DATE","TICKER"),
   * FOREIGN KEY("TICKER") REFERENCES "DIM_STOCKS"("TICKER")
* __FACT_DIVIDENDS__
   * "DIVIDEND_DATE" DATETIME NOT NULL,
   * "TICKER"        TEXT NOT NULL,
   * "DIVIDEND_AMOUNT"       REAL NOT NULL,
   * PRIMARY KEY("DIVIDEND_DATE","TICKER"),
   * FOREIGN KEY("TICKER") REFERENCES "DIM_STOCKS"("TICKER")
* __DIM_STOCK_DATES__ (view) 
    * TICKER
    * MIN_DT
    * MAX_DT

The database is populated using two main sources of financial data:
* stooq (https://stooq.com/db/h/)
* manual (dowloaded from various online sources and cleaned)

Below the data which have been manually downloaded from various online sources. 

| Symbol Name                       | File name                  | Start date   |Used (Y/N)| Used for                                                           | Frequency |                 Description                            																       | Source                                                                              |
|-----------------------------------|----------------------------|--------------|----------|--------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| TIP                               | DFII10.csv                 | 02/01/2003   | Y | Medium term backtest                                               | Daily     | 10 Year Treasury Inflation-Indexed Security																				   | https://fred.stlouisfed.org/series/DFII10                                           |
| GLD                               | Gold.csv                   | 29/12/1978   | N | Medium term backtest                                               | Daily     | Gold Prices                    																					           | ???                                                                                 |
| GLD                               | GOLDAMGBD228NLBM.csv       | 02/04/1968   | Y | Medium term backtest                                               | Daily     | Gold Prices                    																					           | https://fred.stlouisfed.org/series/GOLDAMGBD228NLBM                                |
| COM                               | SPGSCITR_IND.csv           | Dec 31, 1969 | Y | Medium term backtest                                               | Daily     | SP500 GSCI Total Return Index (commodity and infl.)  																	   | https://tradingeconomics.com/commodity/gsci (Not sure)                              |
| ITB                               | ^FVX.csv                   | 02/01/1962   | Y | Medium term backtest                                               | Daily     | Treasury Yield 5 Years                     																				   | https://finance.yahoo.com/quote/%5EFVX/history?p=%5EFVX                             |
| SP500                             | ^GSPC.csv                  | 30/12/1927   | Y | Medium term backtest                                               | Daily     | SP500 Index                           																					   | https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC                           |
| SP500TR                           | ^SP500TR.csv               | 04/01/1988   | Y | Medium term backtest                                               | Daily     | SP500 Index total return                          																		   | https://finance.yahoo.com/quote/%5ESP500TR/history?p=%5ESP500TR                     |
| US20YB                            | DGS20.csv                  | 04/01/1962   | Y | Medium term backtest                                               | Daily     | Treasury Yield 20 Years                    																				   | https://fred.stlouisfed.org/series/DGS20                                            |
| LTB                               | ^TYX.csv                   | 15/02/1977   | Y | Medium term backtest                                               | Daily     | Treasury Yield 30 Years                    																				   | https://finance.yahoo.com/quote/%5ETYX/history?p=%5ETYX                             |
|                                   | T10Y2Y                     |              | Y | Indicator for rotational strategy                                  | Daily     | 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity          									   | https://fred.stlouisfed.org/series/T10Y2Y                                           |
|                                   | DFII20                     |              | Y | Indicator for rotational strategy                                  | Daily     | 20-Year Treasury Inflation-Indexed Security, Constant Maturity                     										   | https://fred.stlouisfed.org/series/DFII20                                           |
|                                   | T10YIE                     |              | Y | Indicator for rotational strategy                                  | Daily     | 10-Year Breakeven Inflation Rate (T10YIE)                                         										   | https://fred.stlouisfed.org/series/T10YIE                                           |
| OIL_LNG                           | F000000__3a.xls            | 1900         | Y | Long term backtest                                                 | Yearly    | U.S. Crude Oil First Purchase Price (Dollars per Barrel) from 1900 to 2019 (annual frequency)                              | http://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=F000000__3&f=A            |
| EQ_LNG, RE_LNG, LTB_LNG, ITB_LNG  | JSTdatasetR4.xlsx          | 1891         | Y | Long term backtest (US equity, bond, bills, housing total return)  | Yearly    | Macroeconomic data from 1870 to 2019                                          											   | http://www.macrohistory.net/data/#DownloadData                                      |
| US10YB_LNG                        | 10usy_b_y.csv              | 1900         | Y | Long term backtest                                                 | Yearly    | US 10Y Treasury yield   																									   | https://stooq.com/q/d/?s=10usy.b													 |
|                                   | GOLD_1800-2019.csv         |              | N | Replaced. It was used for the long term backtest.                  | Yearly    | Gold prices                                                                     										   | https://www.measuringworth.com/datasets/gold/                                       |
| GLD_LNG                           | GOLD_PIKETTY_1850-2011.csv | 1850         | Y | Long term backtest                                                 | Yearly    | Gold prices   																											   | http://piketty.pse.ens.fr/files/capital21c/xls/RawDataFiles/GoldPrices17922012.pdf  |


# Todo List
- [ ] Implement and test momentum and trending strategies
- [ ] Assign weight to money market, instead of cash, for assets excluded by trend or momentum
- [ ] Integrate the database in the backtesting engine
- [ ] Create a GUI (Googlesheet or dash)  
- [X] Add drawdown plots (for portfolio and assets)
- [X] Add money withdrawal functionality
- [X] Create a script to create and execute orders on IBKR (paper trading and live) __Separate repository__
- [ ] Integrate asset rotation strategy with risk parity (comparison with RP) __Implemented: results are wrong__
- [X] Check money drawdown in report that is probably wrong
- [X] Clean yearly data and add functionality to run backtest on them, regression testing 
- [X] ~~Scan galaxy of assets that are uncorrelated by buckets and save them~~ See Uncorrelated asset Jupyter notebook
- [X] ~~Report: add max time in drawdown, VaR.~~ Added Pyfolio report. Added multistrategy report. 
- [X] Create simple vanilla risk parity strategy
- [X] Add the function to automatically calculate the different weights
- [X] Make the format of the data csv files homogeneous among the different sources
- [X] Define the bonds proxy based on yields
- [X] Create a small function to emulate leverage
- [X] Bucketing by asset classes and put into place the strategy for the weights of each bucket
- [X] Implement asset rotation strategy


