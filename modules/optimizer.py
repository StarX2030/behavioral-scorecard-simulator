from scipy.optimize import minimize
import numpy as np

def optimize_cutoffs(data, objective="Maximize Approval"):
    """Optimize cutoffs based on selected objective"""
    segments = data['segment'].unique()
    optimized_cutoffs = {}
    
    for segment in segments:
        segment_data = data[data['segment'] == segment]
        scores = segment_data['score'].values
        defaults = segment_data['default_flag'].values
        
        # Initial guess (median score)
        x0 = np.median(scores)
        
        # Define objective function
        if objective == "Maximize Approval":
            def objective_func(cutoff):
                return -np.sum(scores >= cutoff)  # Negative for maximization
        elif objective == "Minimize Risk":
            def objective_func(cutoff):
                approved = scores >= cutoff
                if np.sum(approved) == 0:
                    return 0
                return np.sum(defaults[approved]) / np.sum(approved)
        else:  # Maximize Profit
            def objective_func(cutoff):
                approved = scores >= cutoff
                if np.sum(approved) == 0:
                    return 0
                revenue = 2000 * np.sum(approved)  # Simplified
                losses = 10000 * np.sum(defaults[approved])  # Simplified
                return -(revenue - losses)  # Negative for maximization
        
        # Constraint: cutoff between min and max scores
        bounds = [(np.min(scores), np.max(scores))]
        
        # Run optimization
        result = minimize(
            objective_func,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimized_cutoffs[segment] = int(result.x[0])
    
    return optimized_cutoffs
