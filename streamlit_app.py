import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from datetime import date

DATA_FILE = "hr_monthly_data.csv"
FEATURES = [
    "overtime", "performance_rating", "job_satisfaction", "environment_satisfaction",
    "years_with_manager", "complaint_from_team", "complaint_about_job",
    "volunteering", "behavioral_change_flag", "department_encoded",
]

st.set_page_config(page_title="Team Retention Tracker", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def train_model(df):
    le = LabelEncoder()
    df = df.copy()
    df["department_encoded"] = le.fit_transform(df["department"])
    X = df[FEATURES]
    y = df["left_company"]
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    return model, le


def risk_color(pct):
    if pct >= 60:
        return "🔴"
    elif pct >= 30:
        return "🟡"
    return "🟢"


df = load_data()
model, le = train_model(df)

page = st.sidebar.radio("Go to", ["Team Overview", "Add Monthly Entry"])

# ------------------------------------------------------------------
# PAGE 1: TEAM OVERVIEW
# ------------------------------------------------------------------
if page == "Team Overview":
    st.title("Team Overview")
    st.caption("This is a model, not a verdict. Every score here is a signal to check in "
               "with someone as a person -- not a reason to act on its own.")

    latest = df.sort_values("month").groupby("employee_id").tail(1).copy()
    latest["department_encoded"] = le.transform(latest["department"])
    latest["risk_pct"] = (model.predict_proba(latest[FEATURES])[:, 1] * 100).round(1)
    latest["flag"] = latest["risk_pct"].apply(risk_color)

    dept_filter = st.selectbox("Filter by department", ["All"] + sorted(df["department"].unique().tolist()))
    view = latest if dept_filter == "All" else latest[latest["department"] == dept_filter]

    view = view.sort_values("risk_pct", ascending=False)
    st.dataframe(
        view[["flag", "employee_id", "department", "month", "risk_pct",
              "overtime", "job_satisfaction", "environment_satisfaction",
              "complaint_from_team", "complaint_about_job", "behavioral_change_flag"]]
        .rename(columns={"risk_pct": "Risk %", "flag": ""}),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------
# PAGE 2: ADD MONTHLY ENTRY
# ------------------------------------------------------------------
else:
    st.title("Add Monthly Entry")
    st.caption("Log one coworker's data for this month. Takes under a minute.")

    with st.form("entry_form"):
        employee_id = st.number_input("Employee ID", min_value=1, step=1)
        department = st.selectbox("Department", sorted(df["department"].unique().tolist()))
        entry_month = st.date_input("Month (use the 1st)", value=date.today().replace(day=1))

        col1, col2 = st.columns(2)
        with col1:
            overtime = st.radio("Overtime this month?", ["No", "Yes"], horizontal=True)
            performance_rating = st.slider("Performance rating (1-5)", 1, 5, 3)
            job_satisfaction = st.slider("Self-reported job satisfaction (1-5)", 1.0, 5.0, 3.0, 0.1)
            environment_satisfaction = st.slider("Environment satisfaction (1-5)", 1.0, 5.0, 3.0, 0.1)
            years_with_manager = st.number_input("Years with current manager", min_value=0.0, step=0.1)
        with col2:
            complaint_from_team = st.radio("Complaint about them from a teammate/manager?", ["No", "Yes"], horizontal=True)
            complaint_about_job = st.radio("Complaint from them about their own situation?", ["No", "Yes"], horizontal=True)
            volunteering = st.radio("Volunteered for anything this month?", ["No", "Yes"], horizontal=True)
            behavioral_change_flag = st.radio("Noticeable change from their own normal behavior?", ["No", "Yes"], horizontal=True)

        submitted = st.form_submit_button("Save entry")

    if submitted:
        st.success(f"Logged for employee {employee_id}, {entry_month.strftime('%B %Y')}.")
        st.info("A concerning score is a reason to check in, never a reason to act alone.")
