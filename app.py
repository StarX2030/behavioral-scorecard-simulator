import os
import streamlit as st
import sys
import subprocess
import pkg_resources
from modules.data_loader import load_data, process_data
from modules.simulations import run_simulation
from modules.analytics import (
    display_score_distribution,
    plot_tradeoff_curves,
    generate_report,
    generate_auto_commentary  # New import
)
from modules.ai_recommender import generate_recommendations
from modules.optimizer import optimize_cutoffs
from utils.helpers import save_session, load_session
import pandas as pd
import numpy as np

# Enhanced package management
REQUIRED_PACKAGES = {
    'streamlit': '1.32.2',
    'pandas': '2.1.4',
    'numpy': '1.26.2',
    'plotly': '5.18.0',
    'scikit-learn': '1.3.2',
    'openpyxl': '3.1.2',  # For Excel support
    'fpdf': '1.7.2'       # For PDF reports
}

def install_missing_packages():
    """Install missing packages with version control"""
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = REQUIRED_PACKAGES.keys() - installed
    
    if missing:
        python = sys.executable
        for package in missing:
            try:
                st.warning(f"Installing {package}...")
                subprocess.check_call(
                    [python, '-m', 'pip', 'install', 
                     f"{package}=={REQUIRED_PACKAGES[package]}"],
                    stdout=subprocess.DEVNULL
                )
                st.success(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install {package}: {str(e)}")
                st.stop()

install_missing_packages()

# App configuration
st.set_page_config(
    page_title="Behavioral Scorecard Simulator PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced sidebar with version info
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.markdown("""
**Behavioral Scorecard Simulator**  
*Version 2.1 - Enterprise Edition*
""")

app_mode = st.sidebar.radio(
    "Navigation",
    ["📤 Data Upload", "🔄 Simulation", "🤖 Optimization", "📊 Reports"],
    index=0
)

# Session state initialization with additional metrics
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cutoffs' not in st.session_state:
    st.session_state.cutoffs = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'commentary' not in st.session_state:
    st.session_state.commentary = ""

def main():
    """Main application controller"""
    if app_mode == "📤 Data Upload":
        render_data_upload()
    elif app_mode == "🔄 Simulation":
        render_simulation()
    elif app_mode == "🤖 Optimization":
        render_optimization()
    elif app_mode == "📊 Reports":
        render_reports()

def render_data_upload():
    """Enhanced data upload section with auto-detection"""
    st.title("📂 Data Upload and Configuration")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload customer data (CSV, Excel, or JSON)",
            type=["csv", "xlsx", "xls", "json"],
            help="Upload your dataset with score, segment, and behavioral features"
        )
    
    with col2:
        st.markdown("### Sample Data")
        if st.button("Load Sample Dataset"):
            # Create sample data if no file uploaded
            sample_data = pd.DataFrame({
                'customer_id': range(1000),
                'score': np.random.normal(650, 100, 1000).astype(int),
                'segment': np.random.choice(['Retail', 'SME', 'Corporate'], 1000),
                'payment_missed': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
                'account_dormancy': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
            })
            st.session_state.data = sample_data
            st.success("Sample data loaded!")
    
    if st.session_state.data is not None:
        with st.expander("🔍 Data Preview", expanded=True):
            st.dataframe(st.session_state.data.head(10))
            
            # Auto-detect segments and initialize cutoffs
            if 'segment' in st.session_state.data.columns:
                segments = st.session_state.data['segment'].unique()
                st.session_state.cutoffs = {
                    seg: int(st.session_state.data['score'].quantile(0.7))
                    for seg in segments
                }
                st.info(f"Detected segments: {', '.join(segments)}")

def render_simulation():
    """Enhanced simulation with auto-commentary"""
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
        
    st.title("🔁 Advanced Simulation Engine")
    
    # Segment selection with metrics
    segments = st.session_state.data['segment'].unique()
    selected_segment = st.selectbox("Select Segment", segments)
    
    # Dynamic cutoff configuration
    min_score = int(st.session_state.data['score'].min())
    max_score = int(st.session_state.data['score'].max())
    current_cutoff = st.session_state.cutoffs.get(selected_segment, 600)
    
    col1, col2 = st.columns(2)
    with col1:
        cutoff = st.slider(
            f"Score Cutoff for {selected_segment}",
            min_score, max_score,
            value=current_cutoff,
            key=f"cutoff_{selected_segment}"
        )
        st.session_state.cutoffs[selected_segment] = cutoff
        
    with col2:
        st.metric("Current Approval Rate", 
                 f"{len(st.session_state.data[st.session_state.data['score'] >= cutoff])/len(st.session_state.data)*100:.1f}%")
    
    # Enhanced simulation button with progress
    if st.button("🚀 Run Advanced Simulation", type="primary"):
        with st.spinner("Running comprehensive analysis..."):
            results = run_simulation(
                st.session_state.data,
                st.session_state.cutoffs
            )
            st.session_state.results = results
            
            # Generate auto-commentary
            st.session_state.commentary = generate_auto_commentary(
                results,
                st.session_state.cutoffs
            )
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Results", "Visualizations", "Commentary"])
            
            with tab1:
                st.subheader("📈 Simulation Metrics")
                st.json(results)
                
            with tab2:
                st.subheader("📊 Data Visualizations")
                display_score_distribution(st.session_state.data, cutoff)
                plot_tradeoff_curves(results)
                
            with tab3:
                st.subheader("🤖 AI-Powered Commentary")
                st.markdown(st.session_state.commentary)
                if st.button("📋 Copy Commentary"):
                    st.session_state.commentary = generate_auto_commentary(
                        results,
                        st.session_state.cutoffs,
                        style="executive"  # Can be "technical" or "detailed"
                    )

def render_optimization():
    """Enhanced optimization with multi-objective support"""
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
        
    st.title("🧠 Intelligent Optimization")
    
    # Multi-objective optimization
    objectives = st.multiselect(
        "Optimization Objectives",
        ["Maximize Approval", "Minimize Risk", "Maximize Profit", "Balance Portfolio"],
        default=["Balance Portfolio"]
    )
    
    constraints = st.expander("⚙️ Advanced Constraints")
    with constraints:
        col1, col2 = st.columns(2)
        with col1:
            min_approval = st.slider("Minimum Approval Rate", 0, 100, 30)
        with col2:
            max_risk = st.slider("Maximum Bad Rate", 0, 30, 10)
    
    if st.button("✨ Run Smart Optimization", type="primary"):
        with st.spinner("Finding optimal cutoffs..."):
            optimized_cutoffs = optimize_cutoffs(
                st.session_state.data,
                objectives=objectives,
                constraints={
                    'min_approval': min_approval/100,
                    'max_risk': max_risk/100
                }
            )
            st.session_state.cutoffs = optimized_cutoffs
            
            # Display results
            st.subheader("Optimized Cutoffs")
            st.write(optimized_cutoffs)
            
            # Generate and display recommendations
            with st.expander("📝 AI Recommendations", expanded=True):
                recommendations = generate_recommendations(
                    st.session_state.data,
                    optimized_cutoffs,
                    style="actionable"  # Can be "technical" or "strategic"
                )
                st.markdown(recommendations)

def render_reports():
    """Enhanced reporting with auto-commentary"""
    if st.session_state.results is None:
        st.warning("Please run a simulation first")
        return
        
    st.title("📄 Intelligent Reporting")
    
    # Report configuration
    report_type = st.selectbox(
        "Report Style",
        ["Executive Summary", "Technical Analysis", "Board Presentation"],
        index=0
    )
    
    commentary_style = st.radio(
        "Commentary Tone",
        ["Concise", "Detailed", "Data-Driven"],
        horizontal=True
    )
    
    if st.button("🖨️ Generate Smart Report"):
        with st.spinner("Compiling comprehensive report..."):
            report = generate_report(
                st.session_state.data,
                st.session_state.cutoffs,
                st.session_state.results,
                report_type,
                commentary={
                    'style': commentary_style.lower(),
                    'content': st.session_state.commentary
                }
            )
            
            # Display report
            st.subheader("Generated Report Preview")
            st.markdown(report[:2000] + "...")  # Show first part
            
            # Enhanced download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "💾 Download PDF Report",
                    data=report,
                    file_name=f"scorecard_report_{report_type.lower().replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
            with col2:
                st.download_button(
                    "📊 Download Excel Summary",
                    data=st.session_state.data.to_csv(index=False),
                    file_name="scorecard_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
