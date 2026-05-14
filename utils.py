import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder


BURNOUT_WEIGHTS = {
    'overtime': 30,
    'satisfaction': 20,
    'work_life_balance': 20,
    'promotion_years': 15,
    'income': 15,
}

DROP_COLUMNS = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']

CATEGORICAL_COLUMNS = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus'
]


def load_data(path='data/employee_attrition.csv'):
    df = pd.read_csv(path)
    return df


def preprocess_for_model(df):
    data = df.copy()

    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns], errors='ignore')

    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    y = data['Attrition']
    X = data.drop(columns=['Attrition'])
    feature_names = list(X.columns)

    return X, y, feature_names, label_encoders


def compute_burnout_score(row, income_median):
    score = 0

    overtime_val = row.get('OverTime', 'No')
    if overtime_val == 'Yes' or overtime_val == 1:
        score += BURNOUT_WEIGHTS['overtime']

    satisfaction = row.get('JobSatisfaction', 3)
    score += ((4 - satisfaction) / 3) * BURNOUT_WEIGHTS['satisfaction']

    wlb = row.get('WorkLifeBalance', 3)
    score += ((4 - wlb) / 3) * BURNOUT_WEIGHTS['work_life_balance']

    promo_years = min(row.get('YearsSinceLastPromotion', 0), 10)
    score += (promo_years / 10) * BURNOUT_WEIGHTS['promotion_years']

    monthly_income = row.get('MonthlyIncome', income_median)
    if monthly_income < income_median:
        income_ratio = monthly_income / income_median
        score += (1 - income_ratio) * BURNOUT_WEIGHTS['income']

    return round(score, 1)


def classify_burnout(score):
    if score <= 35:
        return 'Low'
    elif score <= 65:
        return 'Medium'
    else:
        return 'High'


def get_recommendations(emp, income_median):
    recs = []

    overtime_val = emp.get('OverTime', 'No')
    is_overtime = overtime_val == 'Yes' or overtime_val == 1

    if is_overtime and emp.get('JobSatisfaction', 3) <= 2:
        recs.append("Reduce overtime hours and introduce workload balancing measures.")

    if emp.get('MonthlyIncome', income_median) < income_median:
        burnout = compute_burnout_score(emp, income_median)
        if burnout >= 66:
            recs.append("Review compensation package — employee is underpaid with high burnout risk.")

    if emp.get('WorkLifeBalance', 3) <= 2:
        recs.append("Consider flexible work arrangements or remote work options.")

    if emp.get('YearsSinceLastPromotion', 0) >= 5 and emp.get('JobSatisfaction', 3) <= 2:
        recs.append("Employee may feel stagnant — evaluate for promotion or role change.")

    if emp.get('EnvironmentSatisfaction', 3) <= 2:
        recs.append("Improve workplace environment and team dynamics.")

    if emp.get('TrainingTimesLastYear', 3) <= 1:
        recs.append("Increase training and development opportunities.")

    if emp.get('NumCompaniesWorked', 0) >= 7 and emp.get('YearsAtCompany', 5) <= 2:
        recs.append("High flight risk — prioritize engagement and retention efforts.")

    if not recs:
        recs.append("No immediate concerns — continue regular check-ins.")

    return recs


def create_attrition_pie(df):
    counts = df['Attrition'].value_counts().reset_index()
    counts.columns = ['Attrition', 'Count']
    fig = px.pie(
        counts, values='Count', names='Attrition',
        title='Attrition Distribution',
        color_discrete_sequence=['#2ecc71', '#e74c3c'],
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    return fig


def create_department_attrition(df):
    dept = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
    fig = px.bar(
        dept, x='Department', y='Count', color='Attrition',
        barmode='group', title='Department-wise Attrition',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
    )
    fig.update_layout(height=400)
    return fig


def create_overtime_attrition(df):
    ot = df.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
    fig = px.bar(
        ot, x='OverTime', y='Count', color='Attrition',
        barmode='group', title='Overtime vs Attrition',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
    )
    fig.update_layout(height=400)
    return fig


def create_income_attrition(df):
    fig = px.box(
        df, x='Attrition', y='MonthlyIncome',
        title='Monthly Income vs Attrition',
        color='Attrition',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
    )
    fig.update_layout(height=400)
    return fig


def create_satisfaction_attrition(df):
    sat = df.groupby(['JobSatisfaction', 'Attrition']).size().reset_index(name='Count')
    sat['JobSatisfaction'] = sat['JobSatisfaction'].astype(str)
    fig = px.bar(
        sat, x='JobSatisfaction', y='Count', color='Attrition',
        barmode='group', title='Job Satisfaction vs Attrition',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
        labels={'JobSatisfaction': 'Satisfaction Level (1=Low, 4=High)'}
    )
    fig.update_layout(height=400)
    return fig


def create_worklife_attrition(df):
    wlb = df.groupby(['WorkLifeBalance', 'Attrition']).size().reset_index(name='Count')
    wlb['WorkLifeBalance'] = wlb['WorkLifeBalance'].astype(str)
    fig = px.bar(
        wlb, x='WorkLifeBalance', y='Count', color='Attrition',
        barmode='group', title='Work-Life Balance vs Attrition',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
        labels={'WorkLifeBalance': 'Balance Rating (1=Bad, 4=Best)'}
    )
    fig.update_layout(height=400)
    return fig


def create_age_attrition(df):
    fig = px.histogram(
        df, x='Age', color='Attrition',
        barmode='overlay', nbins=20,
        title='Age Distribution by Attrition',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
        opacity=0.7
    )
    fig.update_layout(height=400)
    return fig


def create_correlation_heatmap(df):
    data = df.copy()
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0}) if data['Attrition'].dtype == 'object' else data['Attrition']
    data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0}) if data['OverTime'].dtype == 'object' else data['OverTime']

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr = data[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
    ))
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=700, width=800,
        xaxis_tickangle=-45
    )
    return fig


def create_burnout_distribution(df):
    fig = px.histogram(
        df, x='BurnoutScore', color='BurnoutRisk',
        nbins=25, title='Burnout Score Distribution',
        color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
        category_orders={'BurnoutRisk': ['Low', 'Medium', 'High']}
    )
    fig.update_layout(height=400)
    return fig


def create_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)

    fig = px.bar(
        feat_df, x='Importance', y='Feature',
        orientation='h', title='Top 10 Feature Importances (Random Forest)',
        color='Importance', color_continuous_scale='Viridis'
    )
    fig.update_layout(height=450, showlegend=False)
    return fig
