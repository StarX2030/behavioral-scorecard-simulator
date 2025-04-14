import json
import datetime
import streamlit as st

def format_percentage(value):
    """Format decimal as percentage"""
    return f"{value:.1%}"

def save_session(filename):
    """Save current session to file"""
    session_data = {
        'data': st.session_state.data.to_dict() if st.session_state.data is not None else None,
        'cutoffs': st.session_state.cutoffs,
        'results': st.session_state.results,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(filename, 'w') as f:
        json.dump(session_data, f)

def load_session(filename):
    """Load session from file"""
    with open(filename, 'r') as f:
        session_data = json.load(f)
    
    if session_data['data'] is not None:
        st.session_state.data = pd.DataFrame.from_dict(session_data['data'])
    st.session_state.cutoffs = session_data['cutoffs']
    st.session_state.results = session_data['results']
