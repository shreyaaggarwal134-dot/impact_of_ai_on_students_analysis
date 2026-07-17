
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Impact on Students Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ----- AI THEME -----
st.markdown(
    '''
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827);
        color: white;
    }
    h1, h2, h3 {
        color: #7dd3fc;
    }
    .metric-card {
        background-color: rgba(255,255,255,0.08);
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.title("🤖 AI Impact on Students Dashboard")
st.caption("An interactive Streamlit application to explore how Generative AI affects student learning, GPA, burnout, and study habits.")

@st.cache_data
def load_data():
    return pd.read_csv("ai_student_impact_dataset (1).csv")

try:
    df = load_data()
except Exception:
    st.warning("Place 'ai_student_impact_dataset (1).csv' in the same folder as app.py.")
    st.stop()

# Sidebar
st.sidebar.header("Filters")
major = st.sidebar.multiselect(
    "Select Major Category",
    options=sorted(df["Major_Category"].unique()),
    default=sorted(df["Major_Category"].unique())
)

filtered_df = df[df["Major_Category"].isin(major)]

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Students", len(filtered_df))
c2.metric("Avg Weekly GenAI Hours", round(filtered_df["Weekly_GenAI_Hours"].mean(), 2))
c3.metric("Avg GPA Change",
          round((filtered_df["Post_Semester_GPA"] - filtered_df["Pre_Semester_GPA"]).mean(), 2))

st.divider()

# Histogram
st.subheader("1. Weekly GenAI Hours Distribution")
fig = px.histogram(
    filtered_df,
    x="Weekly_GenAI_Hours",
    nbins=20,
    title="Distribution of Weekly GenAI Hours"
)
st.plotly_chart(fig, use_container_width=True)

# Pie Chart
st.subheader("2. Primary Use Cases of AI")
fig = px.pie(filtered_df, names="Primary_Use_Case")
st.plotly_chart(fig, use_container_width=True)

# Prompt Engineering Skills
st.subheader("3. Prompt Engineering Skills")
fig2, ax = plt.subplots()
sns.countplot(
    data=filtered_df,
    x="Prompt_Engineering_Skill",
    hue="Prompt_Engineering_Skill",
    ax=ax
)
plt.xticks(rotation=20)
st.pyplot(fig2)

# GPA Change
st.subheader("4. AI Usage vs GPA Change")
filtered_df["GPA_Change"] = (
    filtered_df["Post_Semester_GPA"] - filtered_df["Pre_Semester_GPA"]
)

fig = px.scatter(
    filtered_df,
    x="Weekly_GenAI_Hours",
    y="GPA_Change",
    color="GPA_Change",
    title="Impact of AI Usage on GPA"
)
st.plotly_chart(fig, use_container_width=True)

# Skill Retention
st.subheader("5. Skill Retention by Prompt Engineering Skill")
skill = filtered_df.groupby(
    "Prompt_Engineering_Skill"
)["Skill_Retention_Score"].mean().reset_index()

fig = px.line(
    skill,
    x="Prompt_Engineering_Skill",
    y="Skill_Retention_Score",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Burnout
st.subheader("6. Burnout Risk Distribution")
fig = px.bar(
    filtered_df["Burnout_Risk_Level"].value_counts().reset_index(),
    x="Burnout_Risk_Level",
    y="count",
    color="Burnout_Risk_Level"
)
st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("7. Correlation Heatmap")
corr_cols = [
    "Weekly_GenAI_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "Pre_Semester_GPA",
    "Post_Semester_GPA",
    "Skill_Retention_Score"
]

fig3, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(filtered_df[corr_cols].corr(), annot=True, ax=ax)
st.pyplot(fig3)

# Data Preview
st.subheader("8. Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

st.markdown("---")
st.markdown(
    "### 🚀 Insights\n"
    "- Track AI adoption among students.\n"
    "- Measure GPA changes before and after AI usage.\n"
    "- Understand burnout and dependency patterns.\n"
    "- Explore prompt engineering and skill retention trends."
)
