import streamlit as st
from app.multiapp import MultiApp
from app.pages import home, settings, exploreDB # import your app modules here
from PIL import Image
import utils
import os, sys

app = MultiApp()

wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logo_path = utils.find('logo.png', wd)
logo = Image.open(logo_path)

st.image(logo)

utils.delete_output_first()

st.markdown("""
# Backtest engine
This app backtests portfolio allocation strategies using historical data.   
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Advanced Settings", settings.app)
app.add_app("Explore Prices DB", exploreDB.app)
# The main app
app.run()

