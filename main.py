import streamlit as st
from web_functions import load_data

from Tabs import home, predict, visualise

Tabs = {
    "Home" : home, 
    "Prediction" : predict,
    "Visualisation" : visualise
}

st.sidebar.title("Navigasi")

page = st.sidebar.radio("Pages", list(Tabs.keys()))

df, x, y = load_data()

if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, x, y)

else:
    Tabs[page].app()