"""Real-world retail workflow example."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.production_workflows import RetailReviewWorkflow
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

def analyze_retail_reviews():
    """Complete retail review analysis workflow."""
    
    print("=== Retail Review Analysis System ===\n")
    
    # Initialize workflow
    workflow = RetailReviewWorkflow()
    
    # Simulate loading reviews from different sources
    reviews_sources = {
        'website': [
            "Excellent product quality! Fast shipping and great packaging.",
            "The item broke after just one week. Very disappointed.",
            "Good value for money, but customer service could be better.",
            "URGENT: Wrong item delivered! Need immediate assistance!",
            "Perfect! Exactly what I was looking for."
        ],
        'mobile_app': [
            "App crashed during checkout. Lost my cart!",
            "Love the new features in the app update.",
            "Shipping took too long, but product is good.",
            "5 stars! Great experience from start to finish.",
            "Product damaged in shipping. Need refund ASAP!"
        ],
        'email': [
            "Thank you for the quick resolution to my issue.",
            "Still waiting for my refund after 2 weeks...",
            "The product quality has really declined lately.",
            "Best online shopping experience ever!",
            "Package never arrived. Tracking shows delivered."
        ]
    }
    
    # Process reviews by source
    all_results = []
    source_insights = {}
    
    for source, reviews in reviews_sources.items():
        print(f"Processing {len(reviews)} reviews from {source}...")
        results = workflow.process_batch(reviews)
        
        # Add source information
        for result in results:
            result['source'] = source
        
        all_results.extend(results)
        
        # Generate source-specific insights
        insights = workflow.generate_insights(results)
        source_insights[source] = insights
    
    # Overall analysis
    print(f"\nTotal reviews processed: {len(all_results)}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Sentiment by source
    print("\n=== Sentiment Analysis by Source ===")
    sentiment_by_source = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    print(sentiment_by_source)
    
    # Priority distribution
    print("\n=== Priority Distribution ===")
    priority_dist = df['priority'].value_counts()
    print(priority_dist)
    
    # Category analysis
    print("\n=== Top Categories by Source ===")
    for source in reviews_sources.keys():
        source_df = df[df['source'] == source]
        categories = []
        for cats in source_df['categories']:
            categories.extend(cats)
        
        if categories:
            cat_counts = pd.Series(categories).value_counts()
            print(f"\n{source}:")
            print(cat_counts.head(3))
    
    # Generate visualizations
    create_visualizations(df, source_insights)
    
    # Generate action items
    print("\n=== Action Items ===")
    generate_action_items(df)
    
    # Save detailed report
    save_report(all_results, source_insights)

def create_visualizations(df, source_insights):
    """Create visualization plots."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Sentiment distribution pie chart
    ax1 = axes[0, 0]
    sentiment_counts = df['sentiment'].value_counts()
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    ax1.set_title('Overall Sentiment Distribution')
    
    # 2. Priority levels bar chart
    ax2 = axes[0, 1]
    priority_counts = df['priority'].value_counts()
    ax2.bar(priority_counts.index, priority_counts.values)
    ax2.set_title('Review Priority Levels')
    ax2.set_xlabel('Priority')
    ax2.set_ylabel('Count')
    
    # 3. Sentiment by source
    ax3 = axes[1, 0]
    sentiment_by_source = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_source.plot(kind='bar', ax=ax3)
    ax3.set_title('Sentiment by Source')
    ax3.set_xlabel('Source')
    ax3.set_ylabel('Count')
    ax3.legend(title='Sentiment')
    
    # 4. Category frequency
    ax4 = axes[1, 1]
    all_categories = []
    for cats in df['categories']:
        all_categories.extend(cats)
    cat_series = pd.Series(all_categories)
    cat_counts = cat_series.value_counts().head(5)
    ax4.barh(cat_counts.index, cat_counts.values)
    ax4.set_title('Top 5 Categories')
    ax4.set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('retail_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'retail_analysis.png'")

def generate_action_items(df):
    """Generate actionable insights from the analysis."""
    
    # Urgent reviews
    urgent_reviews = df[df['priority'].isin(['urgent', 'high'])]
    
    if not urgent_reviews.empty:
        print(f"\n1. URGENT: {len(urgent_reviews)} reviews require immediate attention")
        for _, review in urgent_reviews.iterrows():
            print(f"   - {review['source']}: \"{review['original_text'][:60]}...\"")
    
    # Negative sentiment analysis
    negative_reviews = df[df['sentiment'] == 'NEGATIVE']
    if not negative_reviews.empty:
        neg_categories = []
        for cats in negative_reviews['categories']:
            neg_categories.extend(cats)
        
        if neg_categories:
            top_neg_category = pd.Series(neg_categories).value_counts().index[0]
            print(f"\n2. Most negative feedback is about: {top_neg_category}")
            print(f"   Recommendation: Review and improve {top_neg_category} processes")
    
    # Source-specific insights
    source_sentiments = df.groupby('source')['sentiment'].apply(
        lambda x: (x == 'NEGATIVE').sum() / len(x)
    )
    worst_source = source_sentiments.idxmax()
    
    print(f"\n3. Highest negative rate from: {worst_source} ({source_sentiments[worst_source]:.1%})")
    print(f"   Recommendation: Investigate {worst_source} user experience")

def save_report(results, insights):
    """Save detailed analysis report."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_reviews': len(results),
            'sources': list(insights.keys()),
            'urgent_count': sum(1 for r in results if r['priority'] in ['urgent', 'high'])
        },
        'source_insights': insights,
        'detailed_results': results
    }
    
    report_path = Path('retail_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    analyze_retail_reviews()
