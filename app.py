import streamlit as st
import sys
import subprocess
import pkg_resources
from modules.data_loader import load_data, process_data
from modules.simulations import run_simulation
from modules.analytics import (
    display_score_distribution,
    plot_tradeoff_curves,
    generate_report
)
from modules.ai_recommender import generate_recommendations
from modules.optimizer import optimize_cutoffs
from utils.helpers import save_session, load_session
import os
# Check and install missing packages
required = {
    'streamlit': '1.29.0',
    'pandas': '1.5.3',
    'numpy': '1.23.5',
    'plotly': '5.13.0',
    'scikit-learn': '1.2.2'
}

def install_missing():
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required.keys() - installed
    
    if missing:
        python = sys.executable
        for package in missing:
            try:
                st.warning(f"Installing {package}...")
                subprocess.check_call([python, '-m', 'pip', 'install', 
                                     f"{package}=={required[package]}"], 
                                    stdout=subprocess.DEVNULL)
                st.success(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                st.error(f"Failed to install {package}")
                st.stop()

install_missing()

# App configuration
st.set_page_config(
    page_title="Behavioral Scorecard Simulator",
    page_icon="📊",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.image("assets/logo.png", width=200)
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Data Upload", "Simulation", "Optimization", "Reports"]
)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cutoffs' not in st.session_state:
    st.session_state.cutoffs = {}
if 'results' not in st.session_state:
    st.session_state.results = None

# Main app logic
def main():
    if app_mode == "Data Upload":
        render_data_upload()
    elif app_mode == "Simulation":
        render_simulation()
    elif app_mode == "Optimization":
        render_optimization()
    elif app_mode == "Reports":
        render_reports()

def render_data_upload():
    st.title("📂 Data Upload and Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload customer data (CSV or Excel)",
        type=["csv", "xlsx"]
    )
    
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.session_state.data = process_data(df)
            
            # Display sample data
            st.success("Data loaded successfully!")
            st.dataframe(st.session_state.data.head())
            
            # Initialize default cutoffs
            segments = st.session_state.data['segment'].unique()
            st.session_state.cutoffs = {seg: 600 for seg in segments}
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

def render_simulation():
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
        
    st.title("🔁 Cutoff Simulation Engine")
    
    # Segment selection
    segments = st.session_state.data['segment'].unique()
    selected_segment = st.selectbox("Select Segment", segments)
    
    # Cutoff slider
    min_score = int(st.session_state.data['score'].min())
    max_score = int(st.session_state.data['score'].max())
    cutoff = st.slider(
        f"Score Cutoff for {selected_segment}",
        min_score, max_score,
        value=st.session_state.cutoffs.get(selected_segment, 600)
    )
    st.session_state.cutoffs[selected_segment] = cutoff
    
    # Run simulation when button clicked
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            results = run_simulation(
                st.session_state.data,
                st.session_state.cutoffs
            )
            st.session_state.results = results
            
            # Display results
            st.subheader("Simulation Results")
            st.json(results)
            
            # Show visualizations
            display_score_distribution(st.session_state.data, cutoff)
            plot_tradeoff_curves(results)

def render_optimization():
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
        
    st.title("🧠 AI-Powered Optimization")
    
    # Optimization objective selection
    objective = st.selectbox(
        "Optimization Objective",
        ["Maximize Approval", "Minimize Risk", "Maximize Profit"]
    )
    
    if st.button("Run Optimization"):
        with st.spinner("Optimizing cutoffs..."):
            optimized_cutoffs = optimize_cutoffs(
                st.session_state.data,
                objective=objective
            )
            st.session_state.cutoffs = optimized_cutoffs
            
            # Display optimized cutoffs
            st.subheader("Optimized Cutoffs")
            st.write(optimized_cutoffs)
            
            # Generate AI recommendations
            st.subheader("AI Recommendations")
            recommendations = generate_recommendations(
                st.session_state.data,
                optimized_cutoffs
            )
            st.write(recommendations)

def render_reports():
    if st.session_state.results is None:
        st.warning("Please run a simulation first")
        return
        
    st.title("📄 Report Generation")
    
    # Report configuration
    report_type = st.selectbox(
        "Report Type",
        ["Executive Summary", "Technical Analysis", "Full Report"]
    )
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            report = generate_report(
                st.session_state.data,
                st.session_state.cutoffs,
                st.session_state.results,
                report_type
            )
            
            # Display report
            st.subheader("Generated Report")
            st.markdown(report)
            
            # Download options
            st.download_button(
                "Download PDF Report",
                data=report,
                file_name="scorecard_report.pdf"
            )

if __name__ == "__main__":
    main()
