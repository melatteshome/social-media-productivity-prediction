import streamlit as st
import pandas as pd
import numpy as np 
import altair as alt
import os 
import plotly.express as px

os.chdir('..')

data = pd.read_csv('streamlit/data/cleaned_data.csv')
df = pd.DataFrame(data=data)
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
    age_sel   = st.sidebar.selectbox("Age group",   sorted(data["age"].unique()))
    gender_sel = st.sidebar.selectbox("Gender",      sorted(data["gender"].unique()))
    work_sel  = st.sidebar.selectbox("Work group",  sorted(data["job_type"].unique()))

    # ---  Filter the dataset  ----------------------------------------------
    mask = (
        (data["age"]  == age_sel) &
        (data["gender"]     == gender_sel) &
        (data["job_type"] == work_sel)
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
               x=alt.X("social_platform_preference", title="Platform"),
               y=alt.Y("actual_productivity_score", title="Influence score"),
               tooltip=["social_platform_preference", "actual_productivity_score"]
           )
           .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)



# make_social_media_usage_over_demograpic(data=df)



# --- Sidebar filters
job_sel  = st.sidebar.multiselect('Job type', df['job_type'].unique())
gender_sel = st.sidebar.multiselect('Gender', df['gender'].unique())
age_bins = [0,25,35,45,55,150]; labels = ['<25','25-34','35-44','45-54','55+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=labels, right=False)

# --- Apply filters
mask = (
    df['job_type'].isin(job_sel) & 
    df['gender'].isin(gender_sel) )
filtered = df.loc[mask] if job_sel or gender_sel else df

# --- KPI cards
col1,col2,col3 = st.columns(3)
col1.metric('Avg SMU time (h)', filtered['daily_social_media_time'].mean().round(1))
col2.metric('Perceived prod.', filtered['perceived_productivity_score'].mean().round(1))
col3.metric('Actual prod.', filtered['actual_productivity_score'].mean().round(1))

# --- Scatter
fig_scatter = px.scatter(filtered, x='daily_social_media_time', y='actual_productivity_score',
                         color='gender', symbol='job_type', facet_row='age_group',
                         trendline='ols', hover_data=['stress_level','sleep_hours'])
st.plotly_chart(fig_scatter, use_container_width=True)
# repeat similarly for heatmap, box/violin and bar …


make_heatmap(input_df=df , input_y='job_type', input_x='social_platform_preference',input_color='blues' ,input_color_theme = 'blues')