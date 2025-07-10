import streamlit as st
import pandas as pd
import numpy as np 
import altair as alt


# page configuration
st.set_page_config(
    page_title="social media usage and productivity",
    page_icon="🤓",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# heatmap function 
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    return heatmap

# donut 

def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']

    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text


def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'


def make_social_media_usage_over_demograpic(data: pd.DataFrame) -> None:
    st.title("Social-media influence by demographic")

    # ---  Sidebar selectors  ----------------------------------------------
    st.sidebar.header("Filter demographics")
    age_sel   = st.sidebar.selectbox("Age group",   sorted(data["age_group"].unique()))
    gender_sel = st.sidebar.selectbox("Gender",      sorted(data["gender"].unique()))
    work_sel  = st.sidebar.selectbox("Work group",  sorted(data["work_group"].unique()))

    # ---  Filter the dataset  ----------------------------------------------
    mask = (
        (data["age_group"]  == age_sel) &
        (data["gender"]     == gender_sel) &
        (data["work_group"] == work_sel)
    )
    subset = data.loc[mask]

    # ---  Inform if no data  -----------------------------------------------
    if subset.empty:
        st.warning("No records match this demographic slice.")
        st.stop()

    # ---  Draw chart  -------------------------------------------------------
    chart = (
        alt.Chart(subset)
           .mark_bar(size=35)
           .encode(
               x=alt.X("social_media:N", title="Platform"),
               y=alt.Y("affect_score:Q", title="Influence score"),
               tooltip=["social_media", "affect_score"]
           )
           .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)


