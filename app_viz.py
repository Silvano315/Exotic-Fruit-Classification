import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

st.set_page_config(layout='wide')
file = st.file_uploader("Drop your csv data:")

if file is not None:
    df = pd.read_csv(file)
    pyg_app = StreamlitRenderer(df)
    pyg_app.explorer()

    #df = pd.read_csv("Data/Fruits_Dataset_Preprocessed_Standard.csv")
    #pyg_app = StreamlitRenderer(df, spec = "Images/pygwalker_config_1.json")
    #pyg_app.explorer()