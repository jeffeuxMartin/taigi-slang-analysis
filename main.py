# import os; os.system("clear")
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
import uuid

# from funcs import *

@st.cache_data
def id_generator(num=1):
    return [uuid.uuid4().hex for _ in range(num)]

@st.cache_data(persist=True)
def load_data():
    DATA = pd.read_csv('taigiData.csv')
    return DATA


st.title("Taigi Slang Analysis")

def SliderTriple(statement_dict, prefix, default_boundaries=(3, 5), min_val=1, max_val=7, step=1):
        
    def slider_kwargs(statement, key):
        return dict(
            label=statement,
            key=key,
            min_value=min_val,
            max_value=max_val,
            step=step,
        )
    
    if any(k not in st.session_state for k in [
            prefix + "slider_real", 
            prefix + "slider_mid", 
            prefix + "slider_low", 
            prefix + "slider_hig"]):
        valL, valH = default_boundaries
        st.session_state[prefix + "slider_real"] = valL, valH
        st.session_state[prefix + "slider_mid"] = valL, valH
        st.session_state[prefix + "slider_low"] = min_val, valL - 1
        st.session_state[prefix + "slider_hig"] = valH + 1, max_val
        
    else:
        valL, valH = st.session_state[prefix + "slider_real"]
        
        midL, midH = st.session_state[prefix + "slider_mid"]
        lowL, lowH = st.session_state[prefix + "slider_low"]
        higL, higH = st.session_state[prefix + "slider_hig"]
        
        if midL != valL or midH != valH:  # move mid
            valL, valH = max(min_val + 1, midL), min(midH, max_val - 1)
        elif lowL != min_val or lowH != valL - 1:  # move low
            lowL = min_val
            if lowH >= valL:  # 跑過頭
                if lowH > max_val - 1 - 1:
                    lowH = max_val - 1 - 1
            valL = lowH + 1
        elif higL != valH + 1 or higH != max_val:  # move hig
            higH = max_val
            if higL < valH:   # 跑過頭
                if higL < min_val + 1 + 1:
                    higL = min_val + 1 + 1
            valH = higL - 1
            
        # 修正
        if valL > valH:
            valH = valL
        elif valH < valL:
            valL = valH

        st.session_state[prefix + "slider_real"] = valL, valH
        st.session_state[prefix + "slider_low"] = min_val, valL - 1
        st.session_state[prefix + "slider_mid"] = valL, valH
        st.session_state[prefix + "slider_hig"] = valH + 1, max_val


    st.slider(**slider_kwargs(statement_dict["low"], prefix + "slider_low"))
    st.slider(**slider_kwargs(statement_dict["mid"], prefix + "slider_mid"))
    st.slider(**slider_kwargs(statement_dict["hig"], prefix + "slider_hig"))
    
    return st.session_state[prefix + "slider_real"]

id_p, id_f = id_generator(2)
with st.expander("按我調整 Proficiency 分組！"):
    st.markdown("## Proficiency")
    poL, poH = SliderTriple(dict(low=":red[生疏]", mid=":blue[普通]", hig=":green[熟悉]"), prefix=id_p)

with st.expander("按我調整 Frequency 分組！ (Not done)"):
    st.markdown("## Frequency")
    foL, foH = SliderTriple(dict(low=":red[少用]", mid=":blue[尚可]", hig=":green[常用]"), prefix=id_f)

def data_split(
        DATA, 
        column="臺熟", 
        splits=dict(
            生疏=list(range(1, 4)), 
            普通=list(range(4, 6)), 
            精熟=list(range(6, 8)))):
    return {k: DATA[DATA[column].isin(splits[k])] for k in splits}

splits = dict(
    生疏=list(range(1, poL)),
    普通=list(range(poL, poH + 1)),
    精熟=list(range(poH + 1, 7 + 1)),
)

DATA = load_data()
# options of target_word
target_words = [col
    for col in DATA.columns
    if col.startswith("<") and col.endswith(">")]
# target_word = st.selectbox(
#     "請選擇欲查詢的詞彙",
#     options=(target_words)
# )
# for split_name in splits:
#     st.markdown(
#         f"""臺語 proficiency: {split_name} 的有 {splits[split_name]}"""
#     )
SPLIT_DFs = data_split(DATA, splits=splits)

cols = st.columns(3)
bardata = {}
for k, col in zip(SPLIT_DFs, cols):
    df = SPLIT_DFs[k]
    portion = f"{df.shape[0] / DATA.shape[0] * 100:.2f} %"
    length = f"{df.shape[0]}"
    col.metric(label=f"{k}\t({length} / {DATA.shape[0]})",
               value=f"{portion}",
               )
    bardata[k] = {"counts": df.shape[0]}
bardata = pd.DataFrame(bardata)
# st.dataframe(bardata)

st.write(
    px.bar(
        bardata,
        orientation="h",
        height=220,
        title=None,
        labels={}
    )
)
tabs = st.tabs(target_words)

for target_word, tab in zip(target_words, tabs):
    with tab:
        # cols = st.columns(len(SPLIT_DFs))
        plot1, plot2, plot3 = st.columns([15, 15, 15])
        figs = []
        for k, tab in zip(SPLIT_DFs, tabs):
            df = SPLIT_DFs[k]
            filtered_df = df[target_word].value_counts()
            fig = px.pie(
                filtered_df,
                names=filtered_df.index,
                values=filtered_df.values,
                labels=list(filtered_df.index),
                title=f"Proficiency {k} \n對 {target_word} 的分布",
            )
            # print(filtered_df.index)
            figs.append(fig)

        plot1.plotly_chart(figs[1 - 1], use_container_width=True)
        plot2.plotly_chart(figs[2 - 1], use_container_width=True)
        plot3.plotly_chart(figs[3 - 1], use_container_width=True)

