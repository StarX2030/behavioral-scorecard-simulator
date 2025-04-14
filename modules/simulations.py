import pandas as pd
import numpy as np
from utils.constants import LGD_VALUES

def run_simulation(data, cutoffs):
    """Run cutoff simulation across segments"""
    results = {}
    
    for segment, cutoff in cutoffs.items():
        segment_data = data[data['segment'] == segment]
        approved = segment_data[segment_data['score'] >= cutoff]
        rejected = segment_data[segment_data['score'] < cutoff]
        
        # Basic metrics
        approval_rate = len(approved) / len(segment_data) if len(segment_data) > 0 else 0
        bad_rate = approved['default_flag'].mean() if len(approved) > 0 else 0
        
        # Advanced risk metrics
        expected_loss = calculate_expected_loss(approved, segment)
        risk_adjusted_return = calculate_rar(approved, segment)
        
        results[segment] = {
            'approval_rate': approval_rate,
            'bad_rate': bad_rate,
            'expected_loss': expected_loss,
            'risk_adjusted_return': risk_adjusted_return,
            'customers_approved': len(approved),
            'customers_rejected': len(rejected)
        }
    
    return results

def calculate_expected_loss(approved_data, segment):
    """Calculate expected loss using LGD and EAD"""
    lgd = LGD_VALUES.get(segment, 0.45)
    ead = approved_data.get('exposure', 10000).mean()  # Default EAD if not provided
    pd = approved_data['default_flag'].mean()
    
    return pd * lgd * ead

def calculate_rar(approved_data, segment):
    """Calculate risk-adjusted return"""
    # Simplified calculation - would use business rules in practice
    expected_loss = calculate_expected_loss(approved_data, segment)
    revenue = approved_data.get('revenue', 2000).sum()  # Default if not provided
    
    return (revenue - expected_loss) / len(approved_data) if len(approved_data) > 0 else 0
