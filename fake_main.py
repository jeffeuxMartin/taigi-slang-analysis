import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd

# Ref.: https://towardsdatascience.com/advanced-streamlit-session-state-and-callbacks-for-data-labelling-tool-1e4d9ad32a3f


def renderer(
    classname=st.slider,
    statement="Number input",
    key="slider",
    default_value=5.0,
    **kwargs,
):
    if key in st.session_state:
        print("Knocked \033[01;31mA\033[0m")
        st.session_state[key] = (
            2 if st.session_state[key][0] < 2 else st.session_state[key][0],
            6 if st.session_state[key][1] > 6 else st.session_state[key][1],
        )
        return classname(
            statement,
            **kwargs,
        )
    else:
        print("Knocked \033[01;32mB\033[0m")
        return classname(
            statement,
            value=default_value,
            **kwargs,
        )
    

@st.cache(persist=True)
def load_data():
    DATA = pd.read_csv('taigiData.csv')
    return DATA

def data_split(DATA, boundA, boundB):
    # DATA.columns
    bad = DATA[DATA["臺熟"] <= boundA]
    norm = DATA[(DATA["臺熟"] >= boundA + 1) & (DATA["臺熟"] <= boundB - 1)]
    good = DATA[DATA["臺熟"] >= boundB]
    return bad, norm, good

mi, ma = renderer(
    st.slider,
    "你熟悉臺語嗎？",
    key="slider1",
    default_value=(3, 5),
    min_value=1,
    max_value=7,
    step=1,
)

st.write(list(range(mi, ma)))