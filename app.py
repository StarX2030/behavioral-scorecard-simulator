import sys
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import random

# Check and install missing packages
required_packages = {
    'streamlit': '1.29.0',
    'pandas': '2.1.0',
    'numpy': '1.24.3',
    'plotly': '5.15.0',
    'scikit-learn': '1.3.0'
}

def install(package, version=None):
    if version:
        package += f'>={version}'
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    for package, version in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            install(package, version)
except subprocess.CalledProcessError:
    st.error("Failed to install required packages. Please check your internet connection.")
    st.stop()

# Custom blue color theme
BLUE_THEME = {
    "primaryColor": "#1E3F66",
    "backgroundColor": "#F0F5FF",
    "secondaryBackgroundColor": "#E1EBFA",
    "textColor": "#1E3F66",
    "font": "sans serif"
}

# Page configuration
st.set_page_config(
    page_title="ASTERIQX - Behavioral Scorecard Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
st.markdown(f"""
    <style>
        .css-1aumxhk {{
            background-color: {BLUE_THEME["backgroundColor"]};
        }}
        .css-1v3fvcr {{
            color: {BLUE_THEME["textColor"]};
        }}
        .st-b7 {{
            color: {BLUE_THEME["textColor"]};
        }}
        .css-1cpxqw2 {{
            background-color: {BLUE_THEME["primaryColor"]};
        }}
        .css-1q8dd3e {{
            background-color: {BLUE_THEME["primaryColor"]};
        }}
        .css-1q8dd3e:hover {{
            background-color: #2A5D8A;
        }}
        .css-1v0mbdj {{
            border: 1px solid {BLUE_THEME["primaryColor"]};
        }}
    </style>
""", unsafe_allow_html=True)

# ASTERIQX branding with blue theme
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://placehold.co/100x40/1E3F66/FFFFFF?text=ASTERIQX&font=roboto);
                background-repeat: no-repeat;
                background-position: 20px 20px;
                background-size: 100px 40px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

# Generate synthetic data
@st.cache_data
def generate_data(num_samples=10000):
    np.random.seed(42)
    
    # Generate scores with some correlation to default probability
    scores = np.clip(np.random.normal(600, 150, num_samples), 300, 900).astype(int)
    
    # Age groups with different distributions
    age_groups = ['<25', '25-35', '36-50', '>50']
    age = np.random.choice(age_groups, num_samples, p=[0.2, 0.35, 0.3, 0.15])
    
    # Income - higher for older groups
    income = np.where(age == '<25', np.random.normal(2000, 500, num_samples),
                     np.where(age == '25-35', np.random.normal(3500, 800, num_samples),
                             np.where(age == '36-50', np.random.normal(4500, 1000, num_samples),
                                     np.random.normal(3800, 1200, num_samples))))
    income = np.clip(income, 500, 10000).astype(int)
    
    # Behavioral flags
    missed_payment = np.random.binomial(1, 0.15 + (900 - scores) * 0.0005, num_samples)
    account_dormancy = np.random.binomial(1, 0.1 + (900 - scores) * 0.0003, num_samples)
    high_utilization = np.random.binomial(1, 0.2 + (900 - scores) * 0.0006, num_samples)
    
    # Default probability based on score and behavioral flags
    default_prob = 1 / (1 + np.exp(-(-5 + 0.01 * (300 - scores) + 
                                   0.5 * missed_payment + 
                                   0.3 * account_dormancy + 
                                   0.4 * high_utilization)))
    default = np.random.binomial(1, default_prob)
    
    data = pd.DataFrame({
        'score': scores,
        'age': age,
        'income': income,
        'missed_payment': missed_payment,
        'account_dormancy': account_dormancy,
        'high_utilization': high_utilization,
        'default': default
    })
    
    return data

# Load or generate data
df = generate_data()

# Main content with usage instructions
st.title("📊 ASTERIQX Behavioral Scorecard Simulator")

with st.expander("ℹ️ How to Use This Tool", expanded=True):
    st.markdown("""
    **Welcome to the ASTERIQX Behavioral Scorecard Simulator**  
    This interactive tool helps you analyze the impact of credit scoring decisions on your portfolio.  
    
    ### Usage Instructions:
    1. **Adjust Parameters** in the sidebar to simulate different scenarios:
       - Filter by age group and income range
       - Select behavioral flags to analyze specific risk segments
       - Set your desired cutoff score
       - Define risk tier thresholds
    
    2. **View Results** in real-time:
       - Key metrics (approval rate, bad rate) update automatically
       - Interactive charts show score distributions and risk composition
       - Automated insights provide expert analysis of your scenario
    
    3. **Optimize Your Strategy**:
       - Experiment with different cutoff scores to balance risk/reward
       - Identify sensitive thresholds where small changes have big impacts
       - Use the insights to inform your credit policies
    
    *Pro Tip: Start with conservative settings, then gradually relax criteria while monitoring the bad rate.*
    """)

st.markdown("---")

# Sidebar for input parameters
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    
    # Age filter
    age_options = ['All'] + sorted(df['age'].unique().tolist())
    selected_age = st.selectbox("Age Group", age_options)
    
    # Income filter
    min_income, max_income = int(df['income'].min()), int(df['income'].max())
    income_range = st.slider(
        "Monthly Income Range (USD)",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income)
    )
    
    # Behavioral flags
    st.markdown("**🚩 Behavioral Flags**")
    missed_payment_flag = st.checkbox("Missed Payment", value=False)
    account_dormancy_flag = st.checkbox("Account Dormancy", value=False)
    high_utilization_flag = st.checkbox("High Credit Utilization", value=False)
    
    # Score cutoff
    cutoff_score = st.slider(
        "Cutoff Score",
        min_value=300,
        max_value=900,
        value=600,
        step=10
    )
    
    # Risk tier thresholds
    st.markdown("**📊 Risk Tier Thresholds**")
    col1, col2 = st.columns(2)
    with col1:
        low_med = st.number_input("Low-Medium", value=600, min_value=300, max_value=900)
    with col2:
        med_high = st.number_input("Medium-High", value=750, min_value=300, max_value=900)

# Filter data based on user inputs
filtered_df = df.copy()

if selected_age != 'All':
    filtered_df = filtered_df[filtered_df['age'] == selected_age]

filtered_df = filtered_df[
    (filtered_df['income'] >= income_range[0]) & 
    (filtered_df['income'] <= income_range[1])
]

flag_filters = []
if missed_payment_flag:
    flag_filters.append(filtered_df['missed_payment'] == 1)
if account_dormancy_flag:
    flag_filters.append(filtered_df['account_dormancy'] == 1)
if high_utilization_flag:
    flag_filters.append(filtered_df['high_utilization'] == 1)

if flag_filters:
    combined_filter = flag_filters[0]
    for f in flag_filters[1:]:
        combined_filter = combined_filter & f
    filtered_df = filtered_df[combined_filter]

# Apply scorecard logic
filtered_df['approved'] = filtered_df['score'] >= cutoff_score
filtered_df['risk_tier'] = pd.cut(
    filtered_df['score'],
    bins=[300, low_med, med_high, 900],
    labels=['High Risk', 'Medium Risk', 'Low Risk']
)

# Calculate metrics
approval_rate = filtered_df['approved'].mean()
bad_rate = filtered_df[filtered_df['approved']]['default'].mean()

# Risk tier distribution among approved
approved_df = filtered_df[filtered_df['approved']]
risk_tier_dist = approved_df['risk_tier'].value_counts(normalize=True).sort_index()

# Key metrics with blue styling
st.header("📈 Portfolio Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Approval Rate", f"{approval_rate:.1%}", 
           help="Percentage of applicants who would be approved")
col2.metric("Bad Rate", f"{bad_rate:.1%}", 
           delta_color="inverse",
           help="Percentage of approved applicants who would default")
col3.metric("Approved Applicants", f"{len(approved_df):,}", 
           help="Total number of applicants who would be approved")

st.markdown("---")

# Charts section
st.header("📊 Simulation Results")

# Approval/Rejection pie chart
fig1 = px.pie(
    filtered_df,
    names=filtered_df['approved'].map({True: 'Approved', False: 'Rejected'}),
    title='Approval/Rejection Distribution',
    hole=0.4,
    color_discrete_sequence=['#1E3F66', '#4A90E2']
)
fig1.update_traces(textposition='inside', textinfo='percent+label')

# Bad rate by risk tier
bad_rate_by_tier = approved_df.groupby('risk_tier')['default'].mean().reset_index()
fig2 = px.bar(
    bad_rate_by_tier,
    x='risk_tier',
    y='default',
    title='Bad Rate by Risk Tier',
    labels={'default': 'Bad Rate', 'risk_tier': 'Risk Tier'},
    text_auto='.1%',
    color_discrete_sequence=['#1E3F66']
)
fig2.update_layout(yaxis_tickformat=".0%")

# Score distribution with cutoff
fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=filtered_df['score'],
    name='All Applicants',
    marker_color='#E1EBFA'
))
fig3.add_trace(go.Histogram(
    x=approved_df['score'],
    name='Approved Applicants',
    marker_color='#1E3F66'
))
fig3.add_vline(
    x=cutoff_score,
    line_dash="dash",
    line_color="#FF4B4B",
    annotation_text=f"Cutoff: {cutoff_score}",
    annotation_position="top"
)
fig3.update_layout(
    title='Score Distribution with Cutoff',
    xaxis_title='Score',
    yaxis_title='Count',
    barmode='overlay'
)
fig3.update_traces(opacity=0.75)

# Show charts
col1, col2 = st.columns(2)
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# Risk tier distribution
st.header("🔍 Portfolio Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Tier Distribution")
    fig4 = px.pie(
        approved_df,
        names='risk_tier',
        title='Approved Applicants by Risk Tier',
        hole=0.4,
        color_discrete_sequence=['#1E3F66', '#4A90E2', '#7FB3FF']
    )
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Total Applicants', 'Approved', 'Rejected', 
                   'Approval Rate', 'Bad Rate', 
                   'Avg Score (Approved)', 'Avg Score (Rejected)'],
        'Value': [
            len(filtered_df),
            len(approved_df),
            len(filtered_df) - len(approved_df),
            f"{approval_rate:.1%}",
            f"{bad_rate:.1%}",
            f"{approved_df['score'].mean():.1f}",
            f"{filtered_df[~filtered_df['approved']]['score'].mean():.1f}"
        ]
    })
    st.dataframe(metrics_df.style
                .set_properties(**{'background-color': '#F0F5FF', 'color': '#1E3F66'})
                .hide(axis="index"),
                use_container_width=True)

# Generate automated commentary
st.header("💡 Simulation Insights")

# Commentary on approval rate
if approval_rate > 0.7:
    approval_comment = "High approval rate suggests a lenient credit policy, which may increase portfolio risk."
elif approval_rate > 0.4:
    approval_comment = "Moderate approval rate balances risk and opportunity appropriately."
else:
    approval_comment = "Conservative approval rate indicates tight credit standards, potentially missing opportunities."

# Commentary on bad rate
if bad_rate > 0.15:
    bad_rate_comment = "⚠️ High bad rate indicates significant credit risk in approved applications. Consider tightening standards or reviewing scorecard weights."
elif bad_rate > 0.08:
    bad_rate_comment = "Moderate bad rate within typical expectations for this portfolio."
else:
    bad_rate_comment = "Low bad rate suggests effective risk management or potentially overly conservative approvals."

# Commentary on risk distribution
if len(approved_df) > 0:
    high_risk_pct = risk_tier_dist.get('High Risk', 0)
    if high_risk_pct > 0.3:
        risk_dist_comment = f"High proportion ({high_risk_pct:.1%}) of approved applicants in High Risk tier warrants caution."
    elif high_risk_pct > 0.15:
        risk_dist_comment = f"Moderate High Risk exposure ({high_risk_pct:.1%}) - monitor performance closely."
    else:
        risk_dist_comment = f"Low High Risk exposure ({high_risk_pct:.1%}) indicates conservative risk positioning."
else:
    risk_dist_comment = "No approved applications to analyze risk distribution."

# Commentary on score cutoff position
score_mean = filtered_df['score'].mean()
if cutoff_score > score_mean + 50:
    cutoff_comment = f"Cutoff score ({cutoff_score}) is significantly above average ({score_mean:.1f}) - highly selective policy."
elif cutoff_score > score_mean - 50:
    cutoff_comment = f"Cutoff score ({cutoff_score}) aligns with average scores ({score_mean:.1f}) - balanced approach."
else:
    cutoff_comment = f"Cutoff score ({cutoff_score}) is below average ({score_mean:.1f}) - inclusive policy with higher risk."

# Compile all commentary
commentary = f"""
### Approval Strategy Analysis
- **Approval Rate**: {approval_comment}
- **Bad Rate**: {bad_rate_comment}
- **Cutoff Positioning**: {cutoff_comment}

### Risk Composition
- **Risk Distribution**: {risk_dist_comment}
"""

# Add behavioral flag impact if any flags are selected
if missed_payment_flag or account_dormancy_flag or high_utilization_flag:
    flag_count = sum([missed_payment_flag, account_dormancy_flag, high_utilization_flag])
    commentary += f"""
### Behavioral Flag Impact
- Analyzing {flag_count} behavioral flag{'s' if flag_count > 1 else ''} shows specialized segment performance:
  - Approval rate: {approval_rate:.1%} vs overall {df['score'].ge(cutoff_score).mean():.1%}
  - Bad rate: {bad_rate:.1%} vs overall {df[df['score'] >= cutoff_score]['default'].mean():.1%}
"""

st.markdown(commentary)

# Add recommendation based on analysis
st.subheader("📌 Recommendation")
if bad_rate > 0.15 and approval_rate > 0.5:
    rec = "🚨 Strongly consider increasing cutoff score or adjusting risk tier thresholds to reduce bad rate."
elif bad_rate > 0.1 and approval_rate > 0.6:
    rec = "Consider moderate tightening of credit standards or more aggressive pricing for higher risk tiers."
elif bad_rate < 0.05 and approval_rate < 0.3:
    rec = "Potential opportunity to safely expand approvals by slightly lowering cutoff score."
else:
    rec = "Current parameters appear balanced. Monitor performance for any shifts."

st.info(rec)

# Confusion matrix (hidden by default)
with st.expander("🔎 Show Confusion Matrix"):
    y_true = filtered_df['default']
    y_pred = filtered_df['approved']
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Rejected', 'Approved'],
        y=['Non-Default', 'Default'],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
    )
    fig_cm.update_layout(title="Confusion Matrix (Default Prediction)")
    st.plotly_chart(fig_cm, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #1E3F66;'>
        <p>ASTERIQX Behavioral Scorecard Simulator | Confidential</p>
    </div>
""", unsafe_allow_html=True)
