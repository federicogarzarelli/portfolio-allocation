import streamlit as st
from multiapp import MultiApp
from pages import home, settings # import your app modules here
from PIL import Image

app = MultiApp()

logo = Image.open('src/app/logo.png')

st.image(logo)

st.markdown("""
# Backtest engine
This app backtests portfolio allocation strategies using historica data.   
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Advanced Settings", settings.app)
# The main app
app.run()