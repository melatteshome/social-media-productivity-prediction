import streamlit as st
import pandas as pd
import numpy as np 
import altair as alt
import os 
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

def plot_social_vs_job_heatmap(df: pd.DataFrame):

    #   Build a frequency table: rows = job types, cols = social platforms
    ctab = pd.crosstab(
        df["job_type"],
        df["social_platform_preference"],
        normalize=False  # absolute counts
    )

    #  Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        ctab,               # data
        cmap="Blues",       # colour scheme
        annot=True,         # write counts in each cell
        fmt="d",            # integer annotation format
        linewidths=.5,
        cbar_kws={"label": "Count"},
        ax=ax
    )
    ax.set_xlabel("Preferred Social Platform")
    ax.set_ylabel("Job Type")
    ax.set_title("Social‑Platform Preference vs. Job Type")

    #   Show in Streamlit and hand back the figure
    st.pyplot(fig, clear_figure=False)
    return fig


os.chdir('..')

data = pd.read_csv('streamlit/data/cleaned_data.csv')
df = pd.DataFrame(data=data)
# page configuration
st.set_page_config(
    page_title="social media usage and productivity",
    page_icon="🤓",
    layout="wide",
    initial_sidebar_state="expanded")



# Side bar 

with st.sidebar:
    st.title('Social Media Usage Vs Productivity Dashboard')

    job_sel  = st.sidebar.multiselect('Job type', df['job_type'].unique())
    gender_sel = st.sidebar.multiselect('Gender', df['gender'].unique())

# # --- Apply filters
mask = (
    df['job_type'].isin(job_sel) & 
    df['gender'].isin(gender_sel) )
filtered = df.loc[mask] if job_sel or gender_sel else df

# Visualization functions 

def show_gender_pie(
    data: pd.DataFrame,
    gender_col: str = "gender",
    title: str = "Gender distribution"
):
 
    if gender_col not in data.columns:
        st.error(f"'{gender_col}' not found in the DataFrame.")
        return None

    # Count each label
    counts = data[gender_col].value_counts(dropna=False).reset_index()
    counts.columns = [gender_col, "count"]

    # Build the pie chart
    fig = px.pie(
        counts,
        names=gender_col,
        values="count",
        title=title,
        hole=0.3,   
            
    )
    fig.update_traces(textinfo="percent+label")  # show % and label

    st.plotly_chart(fig, use_container_width=True)

    return fig

# stacked column chart 

def show_focus_vs_wellbeing_bars(
    df: pd.DataFrame,
    focus_col: str = "uses_focus_apps",
    wellbeing_col: str = "has_digital_wellbeing_enabled",
    percent: bool = False
):
    counts = (
        df
        .groupby([focus_col, wellbeing_col], observed=False)
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        counts, x=focus_col, y="count", color=wellbeing_col,
        barmode="stack" if not percent else "group",
        text_auto=True,
        labels={focus_col: "Uses Focus Apps", wellbeing_col: "Digital wellbeing"},
        title="Focus‑app use vs. Digital‑wellbeing status",
        height= 300
    )
    if percent:
        fig.update_layout(barmode="stack")
        fig.update_traces(offsetgroup=0)
        # Convert to 100 %: normalise counts within each bar
        fig.update_traces(
            y=[c / counts[counts[focus_col] == x]["count"].sum()
               for x, c in zip(counts[focus_col], counts["count"])])
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# heat map 


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_social_vs_job_heatmap(df: pd.DataFrame):
    """
    Display (and return) a transparent‑background heat‑map that shows the
    relationship between social_platform_preference and job_type.
    """
    # Build the frequency table
    ctab = pd.crosstab(
        df["job_type"],
        df["social_platform_preference"],
        normalize=False
    )

    # Create a *transparent* figure & axes
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="none")  # fig bg ⬝ transparent
    ax.set_facecolor("none")                                  # axes bg ⬝ transparent

    # Draw the heat‑map
    sns.heatmap(
        ctab,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=.5,
        annot_kws={"color": "white"},   # <-- annotation text
        cbar_kws={"label": "Count"},
        ax=ax
    )

    # -------- make everything else white --------
    ax.tick_params(colors="white")                 # tick labels
    ax.xaxis.label.set_color("white")              # x‑axis title
    ax.yaxis.label.set_color("white")              # y‑axis title
    ax.title.set_color("white")                    # main title

    cbar = ax.collections[0].colorbar             # grab the color‑bar
    cbar.ax.yaxis.label.set_color("white")         # its title
    plt.setp(cbar.ax.get_yticklabels(), color="white")  # its tick labels


    # Optional: hide spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Render in Streamlit
    st.pyplot(fig, clear_figure=False)      # Streamlit respects the transparency
    return fig


# clustered chart


def plot_work_vs_social_by_job(df: pd.DataFrame):

    # --- 1) summarise -------------------------------
    agg = (
        df.groupby("job_type", as_index=False)
          .agg(
              avg_work_hours=("work_hours_per_day", "mean"),
              avg_social_time=("daily_social_media_time", "mean")
          )
          .melt(
              id_vars="job_type",
              var_name="metric",
              value_name="hours"
          )
    )

    # --- 2) build grouped‑bar chart -----------------
    chart = (
        alt.Chart(agg, height=400)
        .mark_bar()
        .encode(
            x=alt.X("job_type:N", title="Job Type"),
            xOffset="metric:N",                       # <-- groups the bars
            y=alt.Y("hours:Q", title="Average hours / day"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["avg_work_hours", "avg_social_time"],
                    range=["#1f77b4", "#ff7f0e"]
                ),
                legend=alt.Legend(title="Metric")
            ),
            tooltip=[
                alt.Tooltip("job_type:N", title="Job type"),
                alt.Tooltip("metric:N",    title="Metric"),
                alt.Tooltip("hours:Q",     title="Hours", format=".2f")
            ]
        )
        .properties(
            width=alt.Step(40),                       # bar width
            title="Work‑hours vs. Social‑media time by Job Type"
        )
    )

    # --- 3) render & return --------------------------
    st.altair_chart(chart, use_container_width=True)   # Streamlit API :contentReference[oaicite:1]{index=1}
    return chart


st.markdown(
    """
    <style>
      .card {border-radius:0.6rem;padding:1rem;text-align:center;color:#fff;}
      .blue  {background:#1f77b4;}
      .green {background:#2ca02c;}
      .purple   {background:#A020F0;}

      .card .stMetric {margin-bottom:0.2rem;}
      .card .stMetric > div > div {font-size:0.9rem;}      /* label */
      .card .stMetric > div span {font-size:1.2rem;}       /* value */
    </style>
    """,
    unsafe_allow_html=True,
)
# App Layout 
with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<div class="card blue">', unsafe_allow_html=True)
            st.metric('Avg SMU time (h)', filtered['daily_social_media_time'].mean().round(1))

        with col2:
            st.markdown('<div class="card green">', unsafe_allow_html=True)
            st.metric('Perceived prod.', filtered['perceived_productivity_score'].mean().round(1))


        with col3:
            st.markdown('<div class="card purple">', unsafe_allow_html=True)
            st.metric('Actual prod.', filtered['actual_productivity_score'].mean().round(1))
            
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
colA, colB, colC = st.columns((2,4,3), gap="medium")

with colA:
    show_gender_pie(df)
    show_focus_vs_wellbeing_bars(df=df)

with colB:
    plot_social_vs_job_heatmap(df=df)
    # plot_work_vs_social_by_job(df=df)
