import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

file = st.file_uploader("Drop your csv data:")

if file is not None:
    df = pd.read_csv(file)
    pyg_app = StreamlitRenderer(df)
    pyg_app.explorer()