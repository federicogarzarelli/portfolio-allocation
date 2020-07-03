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


# Dataset explanation

| Symbol  |                  Meaning                            |
|---------|-----------------------------------------------------|
| DFII10  |     10 Year Treasury Inflation-Indexed Security     |
|  Gold   |               Gold Prices                           |
|SPGSCITR |  SP5 GSCI Total Return Index (commodity and infl.)  |
|  FVX    |          Treasury Yield 5 Years                     |
|  GSPC   |               SP500 Index                           |
|  TYX    |          Treasury Yield 30 Years                    |


# Todo List

- [X] Create simple vanilla risk parity strategy
- [X] Add the function to automatically calculate the different weights
- [X] Make the format of the data csv files homogeneous among the different sources
- [X] Define the bonds proxy based on yields
- [X] Create a small function to emulate leverage
- [ ] Scan galaxy of assets that are uncorrelated by buckets and save them
- [ ] Create a script to create and execute orders on IBKR

