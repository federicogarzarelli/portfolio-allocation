import streamlit as st
import os, sys
from pages.home import session_state
from GLOBAL_VARS import *
from PortfolioDB import PortfolioDB

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
import SessionState

def app():
    st.title('Explore Prices DB')

    st.write('Here you can explore the database of prices.')

    st.markdown('## Assets info')

    db = PortfolioDB(databaseName=DB_NAME)
    sqlQry = ""
    data = db.readDatabase(sqlQry)
    st.dataframe(data)

    st.markdown('## Assets metrics')

    with st.form("assets_input_params"):

        col1, col2 = st.beta_columns(2)
        session_state.assets_startdate = col1.date_input('start date', value=session_state.assets_startdate,
                                                        min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                                                        max_value=date.today(), key='assets_startdate',
                                                        help='start date for the asset chart')
        session_state.assets_enddate = col2.date_input('end date', value=session_state.assets_enddate,
                                                      min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                                                      max_value=date.today(), key='assets_enddate',
                                                      help='end date for the asset chart')

        session_state.assets_multiselect = st.multiselect("Select the assets", data['ticker'],
                                                          default=['SP500','ZB.F','ZN.F','BM.F','GC.C'], key=None, help="Select the assets to display")

        sqlQry=""
        data = db.readDatabase(sqlQry)

        # Plot the price

        # Plot the returns
        # Same as in HOME

        # Create table of metrics
        # Total return
        # Annual return
        # max dd
        # Sharpe ratio
        # ...




