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


# st.title("Taigi Slang Analysis (better)")

def SliderTriple(statement_dict, prefix, default_boundaries=(4, 5), min_val=1, max_val=7, step=1):
        
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

# with st.expander("按我調整 Frequency 分組！"):
#     st.markdown("## Frequency")
#     foL, foH = SliderTriple(dict(low=":red[少用]", mid=":blue[尚可]", hig=":green[常用]"), prefix=id_f)

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
target_words = [col
    for col in DATA.columns
    if col.startswith("<") and col.endswith(">")]
target_word = st.selectbox(
    "請選擇欲查詢的詞彙",
    options=(target_words)
)
# for split_name in splits:
#     st.markdown(
#         f"""臺語 proficiency: {split_name} 的有 {splits[split_name]}"""
#     )
SPLIT_DFs = data_split(DATA, splits=splits)

# %%
# st.dataframe(SPLIT_DFs["生疏"].head(10))
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

word = "<啥潲>"
word = target_word
生v = SPLIT_DFs["生疏"][word].value_counts()
普v = SPLIT_DFs["普通"][word].value_counts()
熟v = SPLIT_DFs["精熟"][word].value_counts()
portion = ([生v.sum(), 普v.sum(), 熟v.sum()])
portion = np.array(portion) / sum(portion)
portion /= portion.max()
生v_portion, 普v_portion, 熟v_portion = portion
# %%
fig = make_subplots(
    rows=1, cols=3,
    specs=[[
        {"type": "pie"}, 
        {"type": "pie"}, 
        {"type": "pie"}, 
    ]],
    subplot_titles=(
        "生疏", 
        "普通", 
        "精熟"
    ),
)

cols = st.columns(3)
bardata = {}
for k, col in zip(SPLIT_DFs, cols):
    df = SPLIT_DFs[k]
    portion = f"{df.shape[0] / DATA.shape[0] * 100:.2f} %"
    length = f"{df.shape[0]}"
    # col.metric(label=f"{k}\t({length} / {DATA.shape[0]})",
    #            value=f"{portion}",
    #            )
    bardata[k] = {"counts": df.shape[0]}
split_data = {}
for number in range(1, 7 + 1):
    split_data[number] = {"counts": DATA[DATA["臺熟"] == number].shape[0]}
# st.write(split_data)
# split_data = pd.DataFrame(split_data).T
# st.dataframe(split_data)
split_data = pd.DataFrame(split_data)
fig_p = px.bar(
    split_data,
    orientation="h",
    height=120,
    title=None,
    labels={
        "value": "Proficiency",
    },
    color_discrete_sequence=[
        "#0000ff",
        "#00ff33",
        "#33cc00",
        "#669900",
        "#996600",
        "#cc3300",
        "#ff0000",
    ],
)
fig_p.update_layout(
    showlegend=False,
    yaxis=dict(
        title=None,
        tickmode="array",
        tickvals=[],
        ticktext=[],
        showticklabels=False,
        hoverformat="",
    )
)
st.plotly_chart(
    fig_p,
    use_container_width=True,
)
    
bardata = pd.DataFrame(bardata)
# st.dataframe(bardata)
fig_bar = px.bar(
    bardata,
    orientation="h",
    height=120,
    # width=200,
    title=None,
    labels={
    #     "variable": "Proficiency",
        "value": "生疏 v.s 普通 v.s 精熟",
        # "index": "",
        "counts": "",
    },
    # hide row name
    # hover_data=["counts"],
    # hover_name="counts",
    # text_auto="3d",
    # text="counts",
)
fig_bar.update_layout(
    showlegend=False,
    # height=800,
    # width=1600,
    yaxis=dict(
        title=None,
        tickmode="array",
        tickvals=[],
        ticktext=[],
        showticklabels=False,
        hoverformat="",
    )
)
st.plotly_chart(
    fig_bar,
    use_container_width=True,
)

def pie_generator(df_valcounts, **kwargs):
    return go.Pie(
        labels=df_valcounts.index,
        values=df_valcounts.values,
        # scalegroup="one",
        # name="",
        **kwargs
    )
fig.add_trace(pie_generator(生v, hole=1 - 生v_portion), col=1, row=1)
fig.add_trace(pie_generator(普v, hole=1 - 普v_portion), col=2, row=1)
fig.add_trace(pie_generator(熟v, hole=1 - 熟v_portion), col=3, row=1)
# fig.update_annotations(font_size=30)
fig.update_traces(
    textposition='inside', 
    # textinfo='percent+label', 
    textinfo='percent', 
    # texttemplate="%{label}: %{value} <br />(%{percent})",
    # texttemplate="%{label} (%{percent})",
    textfont_size=15.5,
    # marker=dict(
    #     colors=["#ff0000", "#0000ff", "#00ff00"],
    #     line=dict(color='#000000', width=2),
    # ),
    # hole=.7,
    # hoverinfo="percent+name",
    # hoverinfo="value",
    hoverinfo="label+value",
    # hovertemplate="%{label}: %{value} <br />(%{percent})",
    # hovertemplate="%{value}",
    # hoverlabel=dict(
    #     bgcolor="white",
    #     font_size=16,
    #     font_family="Rockwell"
    # ),
    # disable hover trace
        
        
)
fig.update_layout(
    showlegend=True,
    # height=800,
    # width=1600,
    title_text="Taigi Slang Analysis",
)
st.plotly_chart(fig, use_container_width=True)
