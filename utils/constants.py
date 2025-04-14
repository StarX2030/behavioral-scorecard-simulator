# Segment mapping for standardization
SEGMENT_MAPPING = {
    'retail': 'Retail',
    'sme': 'SME',
    'corp': 'Corporate',
    'micro': 'Microfinance'
}

# Loss Given Default by segment
LGD_VALUES = {
    'Retail': 0.45,
    'SME': 0.40,
    'Corporate': 0.35,
    'Microfinance': 0.50
}

# Risk thresholds by segment
SEGMENT_THRESHOLDS = {
    'Retail': {
        'max_bad_rate': 0.05,
        'min_approval': 0.60
    },
    'SME': {
        'max_bad_rate': 0.07,
        'min_approval': 0.50
    },
    # Other segments...
}
