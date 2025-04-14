from utils.constants import SEGMENT_THRESHOLDS

def generate_recommendations(data, cutoffs):
    """Generate AI-powered recommendations"""
    segments = data['segment'].unique()
    recommendations = []
    
    for segment in segments:
        segment_data = data[data['segment'] == segment]
        approved = segment_data[segment_data['score'] >= cutoffs.get(segment, 600)]
        
        if len(approved) == 0:
            continue
            
        bad_rate = approved['default_flag'].mean()
        threshold = SEGMENT_THRESHOLDS.get(segment, {}).get('max_bad_rate', 0.05)
        
        if bad_rate > threshold:
            recommendations.append({
                'segment': segment,
                'action': 'Increase cutoff',
                'current_cutoff': cutoffs.get(segment, 600),
                'suggested_adjustment': '+20 points',
                'rationale': f"Bad rate {bad_rate:.1%} exceeds threshold of {threshold:.1%}"
            })
        
        # Additional business rules would go here...
    
    return recommendations
