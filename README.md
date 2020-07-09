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
python main.py [--historic | --shares SHARES_LIST --shareclass SHARES_CLASS_LIST] [--weights ASSET_WEIGHTS | --strategy STRATEGY] [--indicators] 
               [--initial_cash CASH] [--monthly_cash CASH] 
               [--create_report [--report_name NAME --report_type REPORTYPE]] 
               [--startdate DATE] [--enddate DATE]
               [--system SYSTEM] [--leverage LEV]
```
               

DESCRIPTION
Running main.py with the options below a backtest is performed on the assets specified following a specified strategy. 

OPTIONS
* `--historic`             use historical asset data ('GLD', 'COM', 'SP500', 'LTB', 'ITB'), already downloaded manually. Alternative is --shares
* `--shares`               use downloaded asset data of the tickers specified in comma separated list (e.g. "SPY,TLT,GLD"). Alternative is --historic.
* `--shareclass`           class of each share specified after --shares (e.g. "equity,bond_lt,gold"). Possibilities are "equity, bond_lt, bond_it, gold, commodity", where "bond_lt" and "bond_it" are long and intermediate duration bonds, respectively. __This argument is mandatory when --shares is chosen__
* `--weights`              list of portfolio weights for each share specified after --shares (e.g. "0.35,0.35,0.30"). The weights need to sum to 1. When weights are specified a custom weights strategy is used that simply loads the weights specified. Alternative is --strategy. __Either this argument or --strategy is mandatory__
* `--strategy`             name of one of the strategy to run for the PDF report. Possibilities are "riskparity_pylib, riskparity, rotationstrat, uniform, vanillariskparity, onlystocks, sixtyforty", where "riskparity_pylib" is the dynamic weights risk parity allocation from the package riskparityportfolio and riskparity is our implementation.  Alternative is --weights. __Either this argument or --weights is mandatory__
* `--indicators`           include the indicator assets (no backtest will be run on these) that are used to decide which assets are used in the strategy. At present these are used only in the asset rotation strategy.  __This argument is mandatory when --strategy rotationstrat is chosen__
* `--initial_cash`         initial_cash to start with. Default is 100000.
* `--monthly_cash`         monthly cash invested. Default is 10000.
* `--create_report`        creates a report if true
* `--report_name`          report name. __This argument should be specified ony when --create_report is chosen__ 
* `--report_type`          __NOT USED - WORK IN PROGRESS__ Type of report. For now only a report that analyses one strategy at a time is possible. When another report e.g. for comparing different strategies will be implemented, one can choose which report to run by specifying this argument. Default is OneStrategyPDF. __This argument should be specified ony when --create_report is chosen__
* `--startdate`            starting date of the simulation. If not specified backtrader will take the earliest possible date. (To test) 
* `--enddate`              end date of the simulation.  If not specified backtrader will take the latest possible date. (To test)
* `--system`               operating system, to deal with different path. Default is windows. If not specified windows is not chosen.
* `--leverage`             leverage to consider. Leverage is applied both with historical (--historical) and automatic (--shares). data Default is 1. 

EXAMPLES
1. Historical data, uniform strategy

```bash
python main.py --historical --strategy uniform --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --report_type OneStrategyPDF
 --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

2. Historical data, custom weights

```bash
python main.py --historical --weights "0.2, 0.3, 0.1, 0.1, 0.3" --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --report_type OneStrategyPDF
 --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

 3. Automatically downloaded data, custom weights

```bash
python main.py --shares "SPY,QQQ,TLT,GLD," --shareclass "equity,equity,bond_lt,gold" --weights "0.2, 0.3, 0.1, 0.4" --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --report_type OneStrategyPDF --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

4. Automatically downloaded data, 60-40 strategy

```bash
python main.py --shares "SPY,QQQ,TLT,GLD," --shareclass "equity,equity,bond_lt,gold" --strategy sixtyforty --initial_cash 100000 --monthly_cash 10000 --create_report --report_name example --report_type OneStrategyPDF --startdate "2015-01-01" --enddate "2020-01-01" --system windows --leverage 3
```

# Dataset explanation
| Symbol  |                  Meaning                            																					|
|---------|-----------------------------------------------------------------------------------------------------------------------------------------|
| DFII10  |     10 Year Treasury Inflation-Indexed Security     																					|
|  Gold   |               Gold Prices                           																					|
|SPGSCITR |  SP5 GSCI Total Return Index (commodity and infl.)  																					|
|  FVX    |          Treasury Yield 5 Years                     																					|
|  GSPC   |               SP500 Index                           																					|
|  TYX    |          Treasury Yield 30 Years                    																					|
|  T10Y2Y |          10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity https://fred.stlouisfed.org/series/T10Y2Y           |
|  DFII20 |          20-Year Treasury Inflation-Indexed Security, Constant Maturity https://fred.stlouisfed.org/series/DFII20                       |
|  T10YIE |          10-Year Breakeven Inflation Rate (T10YIE) https://fred.stlouisfed.org/series/T10YIE                                            |

# Todo List
- [X] Create simple vanilla risk parity strategy
- [X] Add the function to automatically calculate the different weights
- [X] Make the format of the data csv files homogeneous among the different sources
- [X] Define the bonds proxy based on yields
- [X] Create a small function to emulate leverage
- [ ] Scan galaxy of assets that are uncorrelated by buckets and save them
- [ ] Create a script to create and execute orders on IBKR (paper trading and live)
- [ ] Bucketing by asset classes and put into place the strategy for the weights of each bucket
- [X] Implement asset rotation strategy
- [ ] Integrate  asset rotation strategy with risk parity (comparison with RP)
- [ ] think about alarms if something is going wrong


