# portfolio-allocation
Backtest portfolio allocation strategies and live trade them on Interactive Brokers  

# Install Requirements

A requirements file has been added to be able to install the required libraries. To install them, you need to run:

```bash
pip install -r requirements
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
python main.py [--historic [medium | long] | --shares SHARES_LIST --shareclass SHARES_CLASS_LIST] [--weights ASSET_WEIGHTS | --strategy STRATEGY] [--indicators] 
               [--initial_cash CASH] [--monthly_cash CASH] 
               [--create_report [--report_name NAME]] 
               [--startdate DATE] [--enddate DATE]
               [--system SYSTEM] [--leverage LEV]
```

## DESCRIPTION
Running main.py with the options below a backtest is performed on the assets specified following a specified strategy. It is recommended to run the code in a jupyter notebook to obtain the pyfolio reports in the right format. A jupyter notebook is created for this purpose in 'src/apply_jupyter.ipynb'.

### Strategies
* __riskparity__ Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run at portfolio level.
* __riskparity_nested__ Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run first at asset classe level (for assets belonging to the same asset class) and then at portfolio level.
* __rotationstrat__ Asset rotation strategy that buy either gold, bonds or equities based on a signal (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy). To use this strategy specify the parameter `--indicators`.
* __uniform__ Static allocation uniform across asset classes. Assets are allocated uniformly within the same asset class.
* __vanillariskparity__ Static allocation to asset classes where weights are taken from https://www.theoptimizingblog.com/leveraged-all-weather-portfolio/ (see section "True Risk Parity").
* __onlystocks__ Static allocation only to the equity class. Assets are allocated uniformly within the equity class.
* __sixtyforty__ Static allocation 60% to the equity class, 20% to the Long Term Bonds class and 20% to the Short Term Bonds class. Assets are allocated uniformly within the asset classes.

__Note__: the asset classes (`--shareclass` argument) used in the strategies are: Gold, Commodities, Equities, Long Term Bonds, Short Term Bonds (see "OPTIONS" section below). When `--shares` is specified, every asset in `SHARES_LIST` must be assigned to one of these 

### Taking into account the minimum period

Please note that the backtest starts after the periods used to calculate the covariance matrix and variance of assets, necessary to compute the weights of `riskparity` and `riskparity_nested` strategies.

The number of periods is set to 120 days for daily data and to 10 years for yearly data (see `GLOBAL_VARS.py`).

For example:
* __Years__ i.e. when `--historic` is `long`, if startdate is between "1916-01-02" and "1917-01-01" the backtest starts on the "1926-01-01"
* __Days__ e.g. when `--historic` is `medium` or when `--shares` are specified, if startdate is "1999-01-01" the backtest starts on the "1999-06-18"

## OPTIONS
* `--historic`             use historical asset data, already downloaded manually. Alternative is `--shares`. If `--historic = "medium"` assets from about 1970 at daily frequency are loaded (`'GLD', 'COM', 'SP500', 'LTB', 'ITB'`). If `--historic = "long"` assets from 1900 at annual frequency are loaded (`'GLD_LNG', 'OIL_LNG', 'EQ_LNG', 'LTB_LNG', 'ITB_LNG'`). `--historic = "long"` cannot be used with the following strategies `riskparity, riskparity_nested, rotationstrat`.
* `--shares`               use downloaded asset data of the tickers specified in comma separated list (e.g. "SPY,TLT,GLD"). Alternative is `--historic`.
* `--shareclass`           class of each share specified after --shares (e.g. `equity,bond_lt,gold`). Possibilities are `equity, bond_lt, bond_it, gold, commodity`, where "bond_lt" and "bond_it" are long and intermediate duration bonds, respectively. __This argument is mandatory when `--shares` is chosen__
* `--weights`              list of portfolio weights for each share specified after `--shares` (e.g. `0.35,0.35,0.30`). The weights need to sum to 1. When weights are specified a custom weights strategy is used that simply loads the weights specified. Alternative is `--strategy`. __Either this argument or `--strategy` is mandatory__
* `--strategy`             name of one of the strategy to run for the PDF report. Possibilities are `riskparity, riskparity_nested, rotationstrat, uniform, vanillariskparity, onlystocks, sixtyforty`. Alternative is --weights. __Either this argument or `--weights` is mandatory__
* `--indicators`           include the indicator assets (no backtest will be run on these) that are used to decide which assets are used in the strategy. At present these are used only in the asset rotation strategy.  __This argument is mandatory when `--strategy rotationstrat` is chosen__
* `--initial_cash`         initial_cash to start with. Default is 100000.
* `--monthly_cash`         monthly cash invested. Default is 10000.
* `--create_report`        creates a report if true
* `--report_name`          report name. __This argument should be specified ony when `--create_report` is chosen__ 
* `--startdate`            starting date of the simulation. If not specified backtrader will take the earliest possible date. (To test) 
* `--enddate`              end date of the simulation.  If not specified backtrader will take the latest possible date. (To test)
* `--system`               operating system, to deal with different path. Default is windows. If not specified windows is not chosen.
* `--leverage`             leverage to consider. Leverage is applied both with historical (`--historic`) and automatic (`--shares`). data Default is 1. 

### Hidden parameters 

The parameters below are hardcoded in the `GLOBAL_VARS.py` file. 

#### General parameters
* __DAYS_IN_YEAR__ Number of days in a year. Default is 260

#### Strategy parameters

* __reb_days__ Number of days (of bars) every which the portfolio is rebalanced. Default is 30 for daily data and 1 for yearly data.
* __lookback_period_short__ Window to calculate the standard deviation of assets returns. Applies to strategies `riskparity` and `riskparity_nested`. Default is 20 for daily data and 10 for yearly data. 
* __lookback_period_long__ Window to calculate the correlation matrix of assets returns. Applies to strategies `riskparity` and `riskparity_nested`. Default is 120 for daily data and 10 for yearly data.
* __printlog__ If true a log is output in the terming. Default is True.
* __corrmethod__ Method for the calculation of the correlation matrix. Applies to strategies `riskparity` and `riskparity_nested`. Default is 'pearson'. Alternative is 'spearman'. 

#### Report parameters

* __outfilename__ File name of the aggregated report. Default is "Aggregated_Report.pdf".
* __user__ username shown in the report. Default is "Fabio & Federico",
* __memo__ notes displayed in the report. Default is "Testing - Report comparing different strategies",
* __riskfree__ Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc. Default is 0.01.
* __targetrate__ Target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio. Default is 0.01.
* __alpha__ Confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio). Default is 0.05.
* __market_mu__ Average annual return of the market, to be used in Treynor ratio, Information ratio. Default is 0.07.
* __market_sigma__ Annual standard deviation of the market, to be used in Treynor ratio, Information ratio. Default is  0.15.
* __fundmode__ Calculate metrics in fund model vs asset mode. Default is True.
* __stddev_sample__ Bessel correction (N-1) when calculating standard deviation from a sample. Default is True.
* __logreturns__ Use logreturns instead of percentage returns when calculating metrics (recommended). Default is True.
* __annualize__  Calculate annualized metrics by annualizing returns first. Default is True.



## EXAMPLES
1. Historical data, uniform strategy

```bash
python main.py --historic "medium" --strategy uniform --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

2. Historical data, custom weights

```bash
python main.py --historic "medium" --weights "0.2, 0.3, 0.1, 0.1, 0.3" --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

 3. Automatically downloaded data, custom weights

```bash
python main.py --shares "SPY,IWM,TLT,GLD," --shareclass "equity,equity,bond_lt,gold" --weights "0.2, 0.3, 0.1, 0.4" --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

4. Automatically downloaded data, 60-40 strategy

```bash
python main.py --shares "SPY,IWM,TLT,GLD," --shareclass "equity,equity,bond_lt,gold" --strategy sixtyforty --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```
5. Multiple strategies backtest
```bash
python main.py --shares "UPRO,UGLD,TYD,TMF,UTSL" --shareclass "equity,gold,bond_it,bond_lt,commodity" --strategy riskparity_nested,riskparity,riskparity_pylib --initial_cash 100000 --monthly_cash 0 --create_report --report_name MyCurrentPortfolio --startdate "2019-01-01" --enddate "2020-06-30" --system windows --leverage 1
```

# Dataset explanation
| Symbol         | Frequency |                 Meaning                            																		    |
|----------------|-----------|------------------------------------------------------------------------------------------------------------------------------|
| DFII10         | Daily     | 10 Year Treasury Inflation-Indexed Security																				    |
| Gold           | Daily     | Gold Prices                    																					            |
| SPGSCITR       | Daily     | SP500 GSCI Total Return Index (commodity and infl.)  																		|
| FVX            | Daily     | Treasury Yield 5 Years                     																					|
| GSPC           | Daily     | SP500 Index                           																					    |
| TYX            | Daily     | Treasury Yield 30 Years                    																					|
| T10Y2Y         | Daily     | 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity https://fred.stlouisfed.org/series/T10Y2Y         |
| DFII20         | Daily     | 20-Year Treasury Inflation-Indexed Security, Constant Maturity https://fred.stlouisfed.org/series/DFII20                     |
| T10YIE         | Daily     |10-Year Breakeven Inflation Rate (T10YIE) https://fred.stlouisfed.org/series/T10YIE                                           |
| F000000__3a    | Yearly    | U.S. Crude Oil First Purchase Price (Dollars per Barrel) from 1900 to 2019 (annual frequency)                                |
| JSTdatasetR4   | Yearly    | Macroeconomic data from 1870 to 2019 http://www.macrohistory.net/data/#DownloadData                                          |
| GOLD_1800-2019 | Yearly    | Gold prices https://www.measuringworth.com/datasets/gold/                                                                    |

# Todo List
- [ ] Create a script to create and execute orders on IBKR (paper trading and live)
- [ ] Integrate asset rotation strategy with risk parity (comparison with RP)
- [ ] think about alarms if something is going wrong (e.g. Telegram)
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


