
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Impact on Students", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("ai_student_impact_dataset (1).csv")

df = load_data()

st.sidebar.title("Impact of Ai on Students Analysis")
st.sidebar.subheader("Created By : Shreya Aggarwal")
st.sidebar.subheader("Roll Number : 12400229")
st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Dataset Overview", "Data Preprocessing", "Visualizations"]
)

st.title("🤖 AI Impact on Students Dashboard")

if page == "Project Overview":
    st.header("Project Introduction")
    st.write("""
    This project analyzes the impact of Generative AI on students, focusing on academic
    performance, AI dependency, burnout, prompt engineering skills, and learning outcomes.
    """)

    st.subheader("Objectives")
    st.markdown("""
    - Study the adoption of AI among students.
    - Analyze GPA changes before and after AI usage.
    - Evaluate AI dependency and burnout levels.
    - Explore prompt engineering and skill retention.
    - Generate actionable insights using data visualization.
    """)

    st.subheader("Tools & Technologies")
    st.markdown("""
    - Python
    - NumPy
    - Pandas
    - Matplotlib
    - Seaborn
    - Plotly
    - Streamlit
    - Jupyter Notebook
    """)

elif page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Columns Present")
    st.write(['Student_ID', 'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Tool_Diversity', 'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 'Institutional_Policy', 'Anxiety_Level_During_Exams', 'Post_Semester_GPA', 'Skill_Retention_Score', 'Burnout_Risk_Level'])

elif page == "Data Preprocessing":
    st.header("Data Preprocessing")

    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    st.dataframe(missing, use_container_width=True)

    st.subheader("Duplicate Values")
    st.write(f"Total Duplicate Rows: {df.duplicated().sum()}")

    st.subheader("Outlier Detection (IQR Method)")
    numeric_cols = df.select_dtypes(include="number").columns
    outlier_report = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report.append([col, count])

    outlier_df = pd.DataFrame(outlier_report,
                              columns=["Column", "Outlier Count"])
    st.dataframe(outlier_df, use_container_width=True)

    st.subheader("Preprocessing Report")
    st.markdown("""
    - Missing values were detected and reviewed.
    - Duplicate records were identified.
    - Outliers were detected using the IQR method.
    - No outlier treatment was performed.
    - Dataset is ready for exploratory analysis and visualization.
    """)

else:
    st.header("Visualizations")
    st.write("Add all charts from the Jupyter notebook in this section.")

    if "Weekly_GenAI_Hours" in df.columns:
        fig = px.histogram(df, x="Weekly_GenAI_Hours")
        st.plotly_chart(fig, use_container_width=True)

    if "Primary_Use_Case" in df.columns:
        fig = px.pie(df, names="Primary_Use_Case")
        st.plotly_chart(fig, use_container_width=True)

    if "Pre_Semester_GPA" in df.columns and "Post_Semester_GPA" in df.columns:
        temp = df.copy()
        temp["GPA_Change"] = temp["Post_Semester_GPA"] - temp["Pre_Semester_GPA"]
        fig = px.scatter(temp, x="Weekly_GenAI_Hours", y="GPA_Change")
        st.plotly_chart(fig, use_container_width=True)

    if "Burnout_Risk_Level" in df.columns:
        burnout = (df['Burnout_Risk_Level'].value_counts().reset_index())
        fig = px.bar(burnout,x='Burnout_Risk_Level',y='count',color='Burnout_Risk_Level',color_discrete_sequence=px.colors.qualitative.Pastel,title='Distribution of Burnout Risk Levels')
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("Continue adding the remaining visualizations from the notebook in this section.")
