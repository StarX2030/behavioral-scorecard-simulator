import pandas as pd
from utils.constants import SEGMENT_MAPPING

def load_data(uploaded_file):
    """Load and validate uploaded data"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Validate required columns
    required_cols = {'customer_id', 'score', 'default_flag', 'segment'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    return df

def process_data(df):
    """Clean and preprocess data"""
    # Convert segment names
    df['segment'] = df['segment'].map(SEGMENT_MAPPING).fillna('Other')
    
    # Clean score values
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score'])
    
    # Ensure default flag is binary
    df['default_flag'] = df['default_flag'].astype(int)
    
    return df
