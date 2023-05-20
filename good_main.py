import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd

# Ref.: https://towardsdatascience.com/advanced-streamlit-session-state-and-callbacks-for-data-labelling-tool-1e4d9ad32a3f


def renderer(
    classname=st.slider,
    statement="Number input",
    key="slider",
    default_value=(3, 5),
    min_value=1,
    max_value=7,
    step=1,
):
    if key in st.session_state:
        mini, maxi = st.session_state[key]
        # if mini == maxi:
        #     mini = maxi - 1
        if mini < 2:
            mini = 2
        if maxi > 6:
            maxi = 6
        st.session_state[key] = (mini, maxi)
        return classname(
            statement,
            key=key,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )
    else:
        return classname(
            statement,
            key=key,
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=default_value,
        )

@st.cache(persist=True)
def load_data():
    DATA = pd.read_csv('taigiData.csv')
    return DATA

def data_split(
        DATA, 
        column="臺熟", 
        splits=dict(
            生疏=list(range(1, 4)), 
            普通=list(range(4, 6)), 
            精熟=list(range(6, 8)))):
    return {k: DATA[DATA[column].isin(splits[k])] for k in splits}


mi, ma = renderer(
    st.slider,
    "請自行拉動邊界",
    key="slider1",
    default_value=(3, 5),
    min_value=1,
    max_value=7,
    step=1,
)

splits = dict(
    生疏=list(range(1, mi)),
    普通=list(range(mi, ma + 1)),
    精熟=list(range(ma + 1, 7 + 1)),
)

DATA = load_data()

target_word = "<哭枵>"
# options of target_word
target_word = st.selectbox(
    "請選擇欲查詢的詞彙",
    options=([col
        for col in DATA.columns
        if col.startswith("<") and col.endswith(">")]
    )
)
# st.write(splits)
for split_name in splits:
    st.markdown(
        f"""臺語 proficiency: {split_name} 的有 {splits[split_name]}"""
    )
SPLIT_DFs = data_split(DATA, splits=splits)

cols = st.columns(len(SPLIT_DFs))

for k, col in zip(SPLIT_DFs, cols):
    with col:
        df = SPLIT_DFs[k]
        pass
        st.markdown(f"""{k} 的有 {df.shape[0]}""")
        # st.write(df)
        # st.write(df.shape)
        filtered_df = df[target_word].value_counts()
        fig = px.pie(
            filtered_df,
            names=filtered_df.index,
            values=filtered_df.values,
            labels=filtered_df.index,
            title=f"Proficiency {k} 對 {target_word} 的分布",
        )
        st.plotly_chart(fig)

