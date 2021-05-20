'''                   databaseFunctions.py
Created: 28/11/2016
    Python script that holds the functions for creating, reading and loading from 
    sqlite databases.
'''

import sqlite3
import pandas as pd
import utils
import os

# Database class to create a new database at the file location given in databasePath.
# Includes methods for creating tables, clearing and removing tables. Adding to and
# querying the database.
class Database():
    # Class initializer
    def __init__(self, databaseName):
      wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
      self.databasePath = utils.find(databaseName, wd)

      conn = sqlite3.connect(self.databasePath)
      conn.close()
         
    # function which creates the table (tableName) (if it doesn't already exist)
    # with the columns in columnList
    def createTable(self, tableName, columnList):
        conn = sqlite3.connect(self.databasePath)
        cursor = conn.cursor()        
        sql_command = """ CREATE TABLE IF NOT EXISTS {} ({}) """.format(tableName, columnList)
        cursor.execute(sql_command)
        conn.commit()
        conn.close()
        print("Table created") 
    
    # Deletes all rows from the table
    def clearTable(self, tableName):
        conn = sqlite3.connect(self.databasePath)
        cursor = conn.cursor()    
        sql_command = """ DELETE FROM {} """.format(tableName) 
        cursor.execute(sql_command)
        conn.commit()
        conn.close()
        print("Table cleared")

        
    # function which removes the table (tableName) from the database db
    def removeTable(self, tableName):
        conn = sqlite3.connect(self.databasePath)
        cursor = conn.cursor()        
        #Remove database
        sql_command = """ DROP TABLE IF EXISTS {} """.format(tableName) 
        cursor.execute(sql_command)
        conn.commit()
        conn.close()
        print("Table removed")
        
            
    # function which adds data in dataframe to table
    def addToDatabase(self, dataFrame, tableName):
        conn = sqlite3.connect(self.databasePath)
        dataFrame.to_sql(name = tableName, con = conn, if_exists = 'append', index = False)       
        conn.commit()
        conn.close()

    # Uses the query sqlQuery to read the database
    def readDatabase(self, sqlQuery):
        conn = sqlite3.connect(self.databasePath)               
        dataFrame = pd.read_sql(sqlQuery, conn)
        conn.close()
        return dataFrame
        
    # Executes a custom sql command. Can be used for removing rows etc
    def executeCommand(self, sql_command):
        conn = sqlite3.connect(self.databasePath)
        cursor = conn.cursor()    
        cursor.execute(sql_command)
        rowsAffected = cursor.rowcount
        conn.commit()
        conn.close()
        print("Command executed")
        return rowsAffected
    
