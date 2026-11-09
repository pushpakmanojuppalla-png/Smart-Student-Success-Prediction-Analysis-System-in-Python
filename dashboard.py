"""Streamlit dashboard.  Run with:  streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.predict import StudentPredictor  # noqa: E402

st.set_page_config(page_title="Smart Student Success", page_icon="🎓", layout="wide")


@st.cache_data
def load_students():
    return pd.read_csv(config.DATA_FILE) if config.DATA_FILE.exists() else None


@st.cache_resource
def load_predictor():
    try:
        return StudentPredictor()
    except FileNotFoundError:
        return None


def show_figure(name: str, caption: str | None = None):
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


st.title("🎓 Smart Student Success Prediction & Analysis")
page = st.sidebar.radio("Navigate", ["Overview", "Predict a student",
                                      "Batch prediction", "Model performance"])
df = load_students()
predictor = load_predictor()

# --------------------------------------------------------------- Overview
if page == "Overview":
    if df is None:
        st.warning("No dataset found. Run `python main.py all` first.")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(df))
    c2.metric("Average score", f"{df[config.TARGET_SCORE].mean():.1f}")
    c3.metric("At-Risk", f"{(df[config.TARGET_LEVEL] == 'At-Risk').mean() * 100:.1f}%")
    c4.metric("Avg attendance", f"{df['attendance_pct'].mean():.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Students by performance level")
        st.bar_chart(df[config.TARGET_LEVEL].value_counts().reindex(config.LEVELS))
    with right:
        st.subheader("Attendance vs final score")
        st.scatter_chart(df, x="attendance_pct", y=config.TARGET_SCORE,
                         color=config.TARGET_LEVEL)

    st.subheader("Explore the data")
    levels = st.multiselect("Filter by level", config.LEVELS, default=config.LEVELS)
    st.dataframe(df[df[config.TARGET_LEVEL].isin(levels)], width="stretch")
    st.subheader("Analysis figures")
    a, b = st.columns(2)
    with a:
        show_figure("correlation_heatmap.png")
        show_figure("at_risk_by_attendance.png")
    with b:
        show_figure("feature_importance.png")
        show_figure("score_by_family_support.png")
    insights = config.REPORTS_DIR / "insights.md"
    if insights.exists():
        with st.expander("Automated insights"):
            st.markdown(insights.read_text())

# --------------------------------------------------------- Single prediction
elif page == "Predict a student":
    if predictor is None:
        st.warning("Models not trained yet. Run `python main.py all` first.")
        st.stop()
    st.subheader("Enter student details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 16, 40, 20)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        parental_education = st.selectbox("Parental education",
                                          ["High School", "Bachelor", "Master", "PhD"])
        family_support = st.selectbox("Family support", ["Low", "Medium", "High"], index=1)
        internet_access = st.selectbox("Internet access at home", ["Yes", "No"])
    with c2:
        study = st.slider("Study hours / week", 0.0, 40.0, 14.0)
        attendance = st.slider("Attendance %", 0.0, 100.0, 78.0)
        gpa = st.slider("Previous GPA (4.0 scale)", 0.0, 4.0, 3.0, 0.05)
        assignment = st.slider("Assignment completion %", 0.0, 100.0, 75.0)
        internal = st.slider("Internal marks", 0.0, 100.0, 70.0)
    with c3:
        participation = st.slider("Participation (1-10)", 1, 10, 6)
        sleep = st.slider("Sleep hours / night", 3.0, 12.0, 7.0, 0.5)
        stress = st.slider("Stress level (1-10)", 1, 10, 5)
        extra = st.slider("Extracurricular activities", 0, 5, 1)
        absences = st.number_input("Absences", 0, 100, 5)

    if st.button("Predict", type="primary"):
        student = {
            "age": age, "gender": gender, "parental_education": parental_education,
            "family_support": family_support, "internet_access": internet_access,
            "study_hours_per_week": study, "attendance_pct": attendance,
            "previous_gpa": gpa, "assignment_completion_pct": assignment,
            "internal_marks": internal, "participation_score": participation,
            "sleep_hours": sleep, "stress_level": stress,
            "extracurricular_activities": extra, "absences": absences,
        }
        res = predictor.predict(student)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted final score", res["predicted_score"])
        m2.metric("Performance level", res["performance_level"])
        m3.metric("At-Risk probability", f"{res['at_risk_probability'] * 100:.0f}%")
        st.bar_chart(pd.Series(res["probabilities"]))
        if res["risk_flag"]:
            st.error("This student is flagged as At-Risk.")
        else:
            st.success("This student is not flagged as At-Risk.")
        st.subheader("Recommendations")
        for r in res["recommendations"]:
            st.markdown(f"- {r}")

# ---------------------------------------------------------- Batch prediction
elif page == "Batch prediction":
    if predictor is None:
        st.warning("Models not trained yet. Run `python main.py all` first.")
        st.stop()
    st.write("Upload a CSV containing these columns: " + ", ".join(f"`{c}`" for c in config.FEATURES))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            out = predictor.predict_df(pd.read_csv(up))
        except ValueError as e:
            st.error(str(e))
        else:
            st.dataframe(out, width="stretch")
            st.bar_chart(out["predicted_level"].value_counts())
            st.download_button("Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

# --------------------------------------------------------- Model performance
else:
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        st.warning("No metrics found. Run `python main.py train` first.")
        st.stop()
    m = json.loads(path.read_text())
    c, r = m["classification"], m["regression"]
    st.subheader("Classifier: performance level")
    a, b, d = st.columns(3)
    a.metric("Best model", c["best_model"])
    b.metric("Accuracy", f"{c['accuracy']:.3f}")
    d.metric("Macro F1", f"{c['f1_macro']:.3f}")
    st.caption("Cross-validated macro-F1 per candidate")
    st.bar_chart(pd.Series(c["cv_f1_macro"]))
    st.subheader("Regressor: final score")
    a, b, d, e = st.columns(4)
    a.metric("Best model", r["best_model"])
    b.metric("MAE", f"{r['mae']:.2f}")
    d.metric("RMSE", f"{r['rmse']:.2f}")
    e.metric("R²", f"{r['r2']:.3f}")
    x, y = st.columns(2)
    with x:
        show_figure("confusion_matrix.png")
    with y:
        show_figure("regression_actual_vs_pred.png")
