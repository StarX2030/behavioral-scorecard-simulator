import plotly.express as px
import pandas as pd
from utils.helpers import format_percentage

def display_score_distribution(data, cutoff=None):
    """Show interactive score distribution plot"""
    fig = px.histogram(
        data,
        x='score',
        color='segment',
        nbins=50,
        title='Score Distribution by Segment'
    )
    
    if cutoff:
        fig.add_vline(
            x=cutoff,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Cutoff: {cutoff}"
        )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_tradeoff_curves(results):
    """Show approval vs bad rate tradeoff"""
    df = pd.DataFrame.from_dict(results, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'segment'}, inplace=True)
    
    fig = px.scatter(
        df,
        x='approval_rate',
        y='bad_rate',
        color='segment',
        size='customers_approved',
        hover_name='segment',
        title='Approval Rate vs. Bad Rate Tradeoff'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_report(data, cutoffs, results, report_type):
    """Generate markdown report"""
    if report_type == "Executive Summary":
        return _generate_exec_summary(data, cutoffs, results)
    elif report_type == "Technical Analysis":
        return _generate_tech_analysis(data, cutoffs, results)
    else:
        return _generate_full_report(data, cutoffs, results)

def _generate_exec_summary(data, cutoffs, results):
    """Generate executive summary"""
    total_approval = sum(res['customers_approved'] for res in results.values())
    total_customers = len(data)
    overall_approval = total_approval / total_customers
    
    report = f"""
# Behavioral Scorecard Simulation Report - Executive Summary

## Key Metrics
- **Overall Approval Rate**: {format_percentage(overall_approval)}
- **Portfolio Expected Loss**: ${sum(res['expected_loss'] for res in results.values()):,.2f}
- **Risk-Adjusted Return**: ${sum(res['risk_adjusted_return'] for res in results.values()):,.2f}

## Segment Performance
"""
    for segment, res in results.items():
        report += f"""
### {segment}
- Approval Rate: {format_percentage(res['approval_rate'])}
- Bad Rate: {format_percentage(res['bad_rate'])}
- Customers Approved: {res['customers_approved']:,}
"""
    
    return report

# Additional reporting functions would go here...
