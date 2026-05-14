import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils import (
    load_data, preprocess_for_model, compute_burnout_score,
    classify_burnout, get_recommendations,
    create_attrition_pie, create_department_attrition,
    create_overtime_attrition, create_income_attrition,
    create_satisfaction_attrition, create_worklife_attrition,
    create_age_attrition, create_correlation_heatmap,
    create_burnout_distribution, create_feature_importance
)

st.set_page_config(
    page_title="Workforce Burnout & Attrition Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    df = load_data()
    income_median = df['MonthlyIncome'].median()
    df['BurnoutScore'] = df.apply(lambda row: compute_burnout_score(row, income_median), axis=1)
    df['BurnoutRisk'] = df['BurnoutScore'].apply(classify_burnout)
    return df, income_median


@st.cache_resource
def get_model():
    with open('models/attrition_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']


def show_overview(df, income_median):
    st.markdown('<p class="main-header">📊 Workforce Intelligence Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Key metrics at a glance</p>', unsafe_allow_html=True)

    total = len(df)
    attrition_count = df['Attrition'].value_counts().get('Yes', 0)
    attrition_rate = round((attrition_count / total) * 100, 1)
    avg_burnout = round(df['BurnoutScore'].mean(), 1)
    high_risk = len(df[df['BurnoutRisk'] == 'High'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Total Employees", f"{total:,}")
    with c2:
        st.metric("🚪 Attrition Rate", f"{attrition_rate}%")
    with c3:
        st.metric("🔥 Avg Burnout Score", f"{avg_burnout}")
    with c4:
        st.metric("⚠️ High-Risk Employees", f"{high_risk}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_attrition_pie(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_burnout_distribution(df), use_container_width=True)

    st.divider()
    st.subheader("⚡ Quick Insights")
    dept_attrition = df[df['Attrition'] == 'Yes'].groupby('Department').size()
    top_dept = dept_attrition.idxmax()
    overtime_attr = df[(df['OverTime'] == 'Yes') & (df['Attrition'] == 'Yes')].shape[0]
    overtime_total_attr = df[df['Attrition'] == 'Yes'].shape[0]
    overtime_pct = round((overtime_attr / overtime_total_attr) * 100, 1) if overtime_total_attr > 0 else 0

    i1, i2 = st.columns(2)
    with i1:
        st.info(f"**📉 Attrition Rate:** **{attrition_rate}%** of employees have left the organization.")
        st.warning(f"**🏢 Department Risk:** **{top_dept}** has the highest number of attrition cases.")
    with i2:
        st.error(f"**⏰ Overtime Impact:** **{overtime_pct}%** of employees who left were working overtime.")
        st.error(f"**🔥 Burnout Alert:** **{high_risk}** employees are currently in the high burnout risk zone.")


def show_analytics(df):
    st.markdown('<p class="main-header">📈 Workforce Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore employee data distributions</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Attrition Analysis", "Employee Factors", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_department_attrition(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_overtime_attrition(df), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_income_attrition(df), use_container_width=True)
        with col4:
            st.plotly_chart(create_age_attrition(df), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_satisfaction_attrition(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_worklife_attrition(df), use_container_width=True)

        try:
            model, feature_names = get_model()
            st.plotly_chart(create_feature_importance(model, feature_names), use_container_width=True)
        except FileNotFoundError:
            st.warning("Model file not found. Run the notebook first to train and save the model.")

    with tab3:
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)


def show_prediction(df, income_median):
    st.markdown('<p class="main-header">🔮 Attrition Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter employee details to predict attrition risk and burnout level</p>', unsafe_allow_html=True)

    try:
        model, feature_names = get_model()
    except FileNotFoundError:
        st.error("Model file not found. Please run the Jupyter notebook first to train and save the model.")
        return

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Personal Details**")
            age = st.slider("Age", 18, 60, 30)
            monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            distance = st.slider("Distance From Home (km)", 1, 29, 10)

        with col2:
            st.markdown("**🏢 Job Satisfaction**")
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)

        with col3:
            st.markdown("**📊 Employee History**")
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            num_companies = st.slider("Companies Worked", 0, 9, 2)
            years_since_promotion = st.slider("Years Since Promotion", 0, 15, 1)
            training_times = st.slider("Training Times Last Year", 0, 6, 3)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        employee_data = {
            'OverTime': overtime,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'MonthlyIncome': monthly_income,
            'YearsSinceLastPromotion': years_since_promotion,
            'EnvironmentSatisfaction': environment_satisfaction,
            'NumCompaniesWorked': num_companies,
            'YearsAtCompany': years_at_company,
            'TrainingTimesLastYear': training_times,
        }

        burnout_score = compute_burnout_score(employee_data, income_median)
        burnout_risk = classify_burnout(burnout_score)

        input_row = pd.DataFrame([{
            'Age': age,
            'BusinessTravel': 1,
            'DailyRate': 800,
            'Department': 1,
            'DistanceFromHome': distance,
            'Education': 3,
            'EducationField': 2,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': 1,
            'HourlyRate': 65,
            'JobInvolvement': job_involvement,
            'JobLevel': 2,
            'JobRole': 4,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': 1,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': 14000,
            'NumCompaniesWorked': num_companies,
            'OverTime': 1 if overtime == 'Yes' else 0,
            'PercentSalaryHike': 14,
            'PerformanceRating': 3,
            'RelationshipSatisfaction': 3,
            'StockOptionLevel': 1,
            'TotalWorkingYears': years_at_company + 2,
            'TrainingTimesLastYear': training_times,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': max(0, years_at_company - 2),
            'YearsSinceLastPromotion': years_since_promotion,
            'YearsWithCurrManager': max(0, years_at_company - 1),
        }])

        for col in feature_names:
            if col not in input_row.columns:
                input_row[col] = 0
        input_row = input_row[feature_names]

        prediction = model.predict(input_row)[0]
        probability = model.predict_proba(input_row)[0]

        st.divider()
        st.subheader("Prediction Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            if prediction == 1:
                st.error("### ⚠️ HIGH ATTRITION RISK")
                st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
            else:
                st.success("### ✅ LOW ATTRITION RISK")
                st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")

        with r2:
            if burnout_risk == 'High':
                st.error(f"### 🔥 Burnout: {burnout_risk}")
            elif burnout_risk == 'Medium':
                st.warning(f"### ⚡ Burnout: {burnout_risk}")
            else:
                st.success(f"### 💚 Burnout: {burnout_risk}")
            st.markdown(f"**Score:** {burnout_score} / 100")

        with r3:
            st.subheader("💡 Recommendations")
            recs = get_recommendations(employee_data, income_median)
            for rec in recs:
                st.info(f"**→** {rec}")


def show_recommendations(df, income_median):
    st.markdown('<p class="main-header">💡 HR Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-driven workforce recommendations</p>', unsafe_allow_html=True)

    high_risk = df[df['BurnoutRisk'] == 'High'].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🔴 High Risk", len(df[df['BurnoutRisk'] == 'High']))
    with c2:
        st.metric("🟡 Medium Risk", len(df[df['BurnoutRisk'] == 'Medium']))
    with c3:
        st.metric("🟢 Low Risk", len(df[df['BurnoutRisk'] == 'Low']))

    st.divider()

    st.subheader("High Burnout Risk Employees")
    if not high_risk.empty:
        display_cols = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime',
                        'JobSatisfaction', 'WorkLifeBalance', 'BurnoutScore', 'Attrition']
        available_cols = [c for c in display_cols if c in high_risk.columns]
        st.dataframe(
            high_risk[available_cols].sort_values('BurnoutScore', ascending=False).head(20),
            use_container_width=True, height=400
        )
    else:
        st.info("No high-risk employees found.")

    st.divider()

    st.subheader("General Workforce Recommendations")

    overtime_pct = round((df[df['OverTime'] == 'Yes'].shape[0] / len(df)) * 100, 1)
    low_sat = round((df[df['JobSatisfaction'] <= 2].shape[0] / len(df)) * 100, 1)
    poor_wlb = round((df[df['WorkLifeBalance'] <= 2].shape[0] / len(df)) * 100, 1)

    if overtime_pct > 25:
        st.warning(f"**⏰ Overtime Management:** {overtime_pct}% of employees work overtime. Consider reviewing workload distribution and hiring additional staff for high-demand roles.")

    if low_sat > 20:
        st.warning(f"**😟 Job Satisfaction:** {low_sat}% of employees report low satisfaction (rating 1-2). Implement regular feedback sessions and address core dissatisfaction drivers.")

    if poor_wlb > 15:
        st.warning(f"**⚖️ Work-Life Balance:** {poor_wlb}% of employees have poor work-life balance. Introduce flexible schedules, remote work options, and wellness programs.")

    avg_income_leavers = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
    avg_income_stayers = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()
    if avg_income_leavers < avg_income_stayers:
        gap = round(avg_income_stayers - avg_income_leavers)
        st.warning(f"**💰 Compensation Gap:** Employees who left earned ${gap:,} less on average. Review salary benchmarks for roles with high attrition.")

    no_promo = df[df['YearsSinceLastPromotion'] >= 5].shape[0]
    no_promo_pct = round((no_promo / len(df)) * 100, 1)
    if no_promo_pct > 10:
        st.warning(f"**📈 Career Growth:** {no_promo_pct}% of employees haven't been promoted in 5+ years. Develop clear career progression paths and internal mobility programs.")


def main():
    df, income_median = get_data()

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/analytics.png", width=80)
        st.title("Navigation")
        page = st.radio(
            "Go to",
            ["📊 Overview", "📈 Analytics", "🔮 Prediction", "💡 Recommendations"],
            index=0
        )

    if page == "📊 Overview":
        show_overview(df, income_median)
    elif page == "📈 Analytics":
        show_analytics(df)
    elif page == "🔮 Prediction":
        show_prediction(df, income_median)
    elif page == "💡 Recommendations":
        show_recommendations(df, income_median)


if __name__ == "__main__":
    main()
