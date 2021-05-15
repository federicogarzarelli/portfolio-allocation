# -*- coding: utf-8 -*-
"""
Child class of database to obtain filtered tables of the portfolio database.

Created on Sun Mar  8 11:50:49 2020

@author: feder
"""

import utils
import os
import pandas as pd
from datetime import datetime, timedelta, date
import yfinance as yf # https://aroussi.com/post/python-yahoo-finance
# pip install yfinance --upgrade --no-cache-dir
from Database import Database
from GLOBAL_VARS import *
import math
import tkinter as tk
from tkinter import filedialog
import glob
from pathlib import Path

###############
## Constants ##
###############
# DEFAULT_DATE = str(date.today())+ " 00:00:00"
# DEFAULT_STARTDATE = "1975-01-01 00:00:00"

class PortfolioDB(Database):

    # Filter stock information table DIM_STOCKS for a range of tickers
    def getStockInfo(self, ticker):

        sqlQuery = '''
           	SELECT *
        	FROM DIM_STOCKS 
        	WHERE ticker = '{}';     
        ''' \
            .format(ticker)
        data = self.readDatabase(sqlQuery) 
        return data
    
    # Filter historical prices table FACT_HISTPRICES for a range of dates and tickers
    def getPrices(self, ticker):

        sqlQuery = '''
           	SELECT *
        	FROM FACT_HISTPRICES 
        	WHERE ticker = '{}' 
        	ORDER BY ticker, date;     
        ''' \
            .format(ticker)

        data = self.readDatabase(sqlQuery)
        data[self.DATE] = pd.to_datetime(data[self.DATE], format = '%Y-%m-%d %H:%M:%S')
        return data

    # Filter historical dividends table FACT_DIVIDENDS for a range of dates and tickers
    def getDividends(self, ticker, startDate = DEFAULT_STARTDATE, endDate = DEFAULT_DATE):
        
        sqlQuery = '''
           	SELECT *
        	FROM FACT_DIVIDENDS 
        	WHERE ticker = '{}' and datetime(dividend_date) BETWEEN date("{}") AND date("{}")
        	ORDER BY ticker, dividend_date;     
        ''' \
            .format(ticker, startDate, endDate)
        data = self.readDatabase(sqlQuery) 
        return data
    
    # Checks whether stock is in database, if not it stockScrape to get all the data.
    # If it is in data base it checks whether the stock information is up to date and only fetches new data
    # Source can be GOOGLEFINANCE or YAHOOFINANCE

    def updateStockData(self, stockCode, source = "YAHOOFINANCE"):
        # Reads database
        sqlQuery = """SELECT {} FROM {} WHERE {} = '{}'; """ \
        .format(self.TICKER, self.HISTORICAL_TABLE_NAME, self.TICKER, stockCode)

        #print(sqlQuery)
        stockData = self.readDatabase(sqlQuery)

        # Checks whether any previous data has been added for the particular stock code
        # if not then run initialStockScrape to get all past data
        if stockData.empty:
            print('Running stockScrape() on {} using {}. --First run.'.format(stockCode, source))
            self.stockScrape(stockCode, source)
        else:
            #access database to get latestDate
            print('Running stockScrape() on {} using {}. --Updating data.'.format(stockCode, source))
            # Performs SQL query to get the latest stock data date in database
            sqlQuery = """SELECT {}, max({}) AS Date FROM {} WHERE {} = '{}' GROUP BY {}""" \
            .format(self.TICKER, self.DATE, self.HISTORICAL_TABLE_NAME, self.TICKER, stockCode, self.TICKER)

            y = self.readDatabase(sqlQuery)
            minDate = y.Date[0]    # minDate is the earliest data of data that the program needs to download
            # Increment date by 1 day
            minDate = utils.incrementDate(minDate)

            today = datetime.now()
            today = today.replace(hour=0, minute=0, second=0, microsecond=0) # end date

            if utils.convertDate(minDate) < today:
                # Updates stock data
                self.stockScrape(stockCode, source, minDate)
            else:
                # Data are already up to date
                print("Data for {} are already up to date. Module: updateStockData.".format(stockCode))

    def getYahooCode(self, stockCode):
            sqlQuery = ''' SELECT {} FROM {} WHERE {} = '{}' ''' \
                .format(self.YAHOO_SYMBOL, self.STOCKS_TABLE_NAME,
                        self.TICKER, stockCode)
            data = self.readDatabase(sqlQuery)
            # If data is empty return 0
            if data.empty:
                print(('No Yahoo Symbol for {}. Module: getYahooCode.'.format(stockCode)))
                return 0
            return data.at[0, self.YAHOO_SYMBOL]
        
    def getGoogleCode(self, stockCode):
            sqlQuery = ''' SELECT {} FROM {} WHERE {} = '{}' ''' \
                .format(self.GOOGLE_FINANCE_SYMBOL, self.STOCKS_TABLE_NAME,
                        self.TICKER, stockCode)
            data = self.readDatabase(sqlQuery)
            # If data is empty return 0
            if data.empty:
                print(('No Yahoo Symbol for {}. Module: getGoogleCode.'.format(stockCode)))
                return 0
            return data.at[0, self.GOOGLE_FINANCE_SYMBOL]
        
    # function which does the first time initialization of the stock and 
    #downloads all past stock data, returns array of dates, and array of data
    def stockScrape(self, stockCode, source = "YAHOOFINANCE", minDate = DEFAULT_STARTDATE):
        # Initialize pandas dataframe to hold stock data    
        stockDataFrame =  pd.DataFrame({self.DATE: [], self.TICKER: [], self.PRICE: []});

        if source == "YAHOOFINANCE":
            YahooCode = self.getYahooCode(stockCode)
            stock = yf.Ticker(YahooCode)

            sdate = utils.convertDate(minDate)  # start date
            dowloaded_data = stock.history(interval="1d", start = sdate)

            # Manipulate the output
            Dates = dowloaded_data.index.to_frame()

            Dates = Dates.reset_index(drop=True)
            Price = dowloaded_data['Close'].reset_index(drop=True)
            Ticker = pd.DataFrame([stockCode] * len(dowloaded_data['Close']),columns=['Ticker'])

        if source == "GOOGLEFINANCE":
            googleticker = self.getGoogleCode(stockCode)
            data = self.googledatatable.copy()

            sdate = utils.convertDate(minDate)  # start date
            edate = datetime.now()
            edate = edate.replace(hour=0, minute=0, second=0, microsecond=0) # end date

            data.DATE = pd.to_datetime(data.DATE, format = '%d/%m/%Y')
            dates_mask = ((data.DATE >= sdate) & (data.DATE <= edate))
            data = data.loc[dates_mask, [self.DATE, googleticker]]

            data = data.dropna()
            data = data[(data[googleticker] != 'NA')]

            Dates =  pd.DataFrame(data.DATE,columns=['DATE']).reset_index(drop=True)
            Ticker = pd.DataFrame([stockCode] * len(data.DATE),columns=['Ticker'])
            Price =  pd.DataFrame(data[googleticker],columns=[googleticker]).reset_index(drop=True)

        stockDataFrame =  pd.concat([Dates, Ticker, Price], axis = 1)
        stockDataFrame.columns = self.HISTORICAL_COLUMNS
        stockDataFrame.ignore_index=True
               
        # Add to SQL database
        self.addToDatabase(stockDataFrame, self.HISTORICAL_TABLE_NAME)

    def db_update(self):
        # Reads database to get the list of stocks with the min and max price date
        sqlQuery = """SELECT * FROM {}; """ \
            .format(self.DATES_STOCKS_TABLE_NAME)

        stockDataDates = self.readDatabase(sqlQuery)

        sqlQuery = """SELECT * FROM {}; """ \
            .format(self.STOCKS_TABLE_NAME)
        stockData = self.readDatabase(sqlQuery)

        # get the folder with the data to upload in the db
        root = tk.Tk()
        root.withdraw()

        data_path = filedialog.askdirectory()
        source = input("What is the source of the data to upload in the db? (stooq, cleaned): ")

        # Get all the filenames in the data directory. These correspond to the tickers in DIM_STOCK
        fileNames = []
        filePaths = []
        for r, d, f in os.walk(data_path):
            for file in f:
                if file.endswith(".txt") or file.endswith(".csv"):
                    thisPath = os.path.abspath(os.path.join(r, file))
                    filePaths.append(thisPath)
                    fileNames.append(str.upper(os.path.splitext(os.path.basename(thisPath))[0]))

        files = pd.DataFrame({'fileNames': fileNames,
                             'filePaths': filePaths})

        # Loop through the files and check if they are in the DB
        for thisfileName in files['fileNames']:
            thisfilePath = files.loc[files['fileNames'] == thisfileName]['filePaths'].values[0]
            thisdata = pd.read_csv(thisfilePath, skiprows=0, header=0, parse_dates=True)

            if source == 'stooq':
                print('Stock: ' + thisfileName + '. Source: ' + source)
                thisdata = {
                        'date': [datetime.strptime(str(x), "%Y%m%d").strftime("%Y-%m-%d") for x in thisdata['<DATE>']],
                        'ticker': thisfileName,
                        'open': thisdata['<OPEN>'],
                        'high': thisdata['<HIGH>'],
                        'low': thisdata['<LOW>'],
                        'close': thisdata['<CLOSE>'],
                        'volume': thisdata['<VOL>'],
                        }
            elif source == 'cleaned':
                print('Stock: ' + thisfileName + '. Source: ' + source)
                thisdata = {
                        'date':[datetime.strptime(str(x),"%Y-%m-%d").strftime("%Y-%m-%d") for x in thisdata['Date']],
                        'ticker': thisfileName,
                        'open': thisdata['open'],
                        'high': thisdata['high'],
                        'low': thisdata['low'],
                        'close': thisdata['close'],
                        'volume': thisdata['volume'],
                        }
            thisdata = pd.DataFrame(thisdata)

            if not(any(stockData['ticker'] == thisfileName)): # the ticker is not in the db
                print(thisfileName + ' not in the DB. Adding it to DIM_STOCKS.')
                # Table Columns
                thisStockdata = {'name': '',
                                'ticker': thisfileName,
                                'exchange': '',
                                'currency': '',
                                'isin': '',
                                'source': source,
                                'frequency': 'not_categorized',
                                'asset_class': 'not_categorized',
                                'treatment_type':'not_categorized',
                                'maturity': 'not_categorized',
                                }
                self.addToDatabase(thisStockdata, PortfolioDB.STOCKS_TABLE_NAME)
                print(thisfileName + ' prices not in the DB. Adding it to FACT_HISTPRICES.')
                self.addToDatabase(thisdata, PortfolioDB.HISTORICAL_TABLE_NAME)
            else:
                # if so, read the file and check if the max date in the file > max date in the db;
                maxDT_DB = datetime.strptime(stockDataDates.loc[stockDataDates['ticker'] == thisfileName]['max_dt'].values[0], '%Y-%m-%d')
                # if so, filter the file to get the dates which are not in the DB and upload them
                thisdata = thisdata.loc[pd.to_datetime(thisdata['date']) > maxDT_DB]
                if not thisdata.empty:
                    print(thisfileName + ' prices not in the DB for dates after '+ str(maxDT_DB) + '. Adding it to FACT_HISTPRICES.')
                    self.addToDatabase(thisdata, PortfolioDB.HISTORICAL_TABLE_NAME)
        return 1

    # DB Names

    ## Information about the stocks
    # Table Name
    DATES_STOCKS_TABLE_NAME = "DIM_STOCK_DATES"
    STOCKS_TABLE_NAME = "DIM_STOCKS"
    
    # Table Columns
    NAME = "name"
    TICKER = "ticker"
    EXCHANGE = "exchange"
    CURRENCY = "currency"
    ISIN = "isin"
    SOURCE = "source"
    FREQUENCY = "frequency"
    ASSET_CLASS = "asset_class"

    STOCKS_COLUMNS = [NAME, TICKER, EXCHANGE, CURRENCY, ISIN, SOURCE, FREQUENCY, ASSET_CLASS]
    
    STOCKS_COLUMN_LIST = "{} TEXT, {} TEXT, {} TEXT, {} TEXT, {} TEXT, {} TEXT, {} TEXT, {} TEXT".format(NAME, TICKER, EXCHANGE, CURRENCY, ISIN, SOURCE, FREQUENCY, ASSET_CLASS)
    
    ## Historical prices table
    # Table Name
    HISTORICAL_TABLE_NAME = "FACT_HISTPRICES"
    
    DATE = "date"
    TICKER = "ticker"
    PRICE = "price"
    
    HISTORICAL_COLUMNS = [DATE, TICKER, PRICE]
    
    HISTORICAL_COLUMN_LIST = "{} DATE, {} TEXT, {} REAL".format(DATE, TICKER, PRICE)
    
    ## Dividends data table contract
    # Table Name
    DIVIDEND_TABLE_NAME = "FACT_DIVIDENDS"
    
    # Table Columns
    DIVIDEND_DATE = "dividend_date"
    TICKER = "ticker"
    DIVIDEND_AMOUNT = "dividend_amount"
           
    DIVIDEND_TOTAL = "dividend_total"
    
    DIVIDEND_COLUMNS = [DIVIDEND_DATE, TICKER, DIVIDEND_AMOUNT]  
       
    DIVIDEND_COLUMN_LIST =  "{} DATE, {} TEXT, {} REAL".format(DIVIDEND_DATE,  TICKER,  DIVIDEND_AMOUNT)