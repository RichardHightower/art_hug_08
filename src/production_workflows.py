"""End-to-end production workflow example."""

import json
import time
from pathlib import Path

from transformers import pipeline

from src.config import BATCH_SIZE, DEVICE


class RetailReviewWorkflow:
    """Production workflow for retail review analysis."""

    def __init__(self):
        # Initialize pipelines
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if DEVICE == "cuda" else -1,
        )

        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if DEVICE == "cuda" else -1,
        )

        self.categories = [
            "product quality",
            "shipping",
            "customer service",
            "pricing",
            "packaging",
        ]

        self.priority_keywords = {
            "urgent": ["broken", "damaged", "fraud", "stolen", "urgent"],
            "high": ["terrible", "awful", "worst", "refund", "complaint"],
            "medium": ["disappointed", "issue", "problem", "concern"],
            "low": ["suggestion", "feedback", "minor"],
        }

    def preprocess(self, text):
        """Clean and normalize review text."""
        import re

        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate very long reviews
        words = text.split()
        if len(words) > 200:
            text = " ".join(words[:200]) + "..."

        return text

    def analyze_priority(self, text):
        """Determine review priority based on keywords."""
        text_lower = text.lower()

        for priority, keywords in self.priority_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return priority

        return "normal"

    def process_batch(self, reviews):
        """Process a batch of reviews."""
        # Preprocess
        cleaned_reviews = [self.preprocess(review) for review in reviews]

        # Sentiment analysis
        sentiments = self.sentiment_pipeline(cleaned_reviews, batch_size=BATCH_SIZE)

        # Category classification
        categories = []
        for review in cleaned_reviews:
            result = self.classification_pipeline(
                review, self.categories, multi_label=True
            )
            categories.append(
                {
                    "labels": result["labels"][:2],  # Top 2 categories
                    "scores": result["scores"][:2],
                }
            )

        # Combine results
        results = []
        for i, review in enumerate(reviews):
            results.append(
                {
                    "original_text": review,
                    "cleaned_text": cleaned_reviews[i],
                    "sentiment": sentiments[i]["label"],
                    "sentiment_score": sentiments[i]["score"],
                    "categories": categories[i]["labels"],
                    "category_scores": categories[i]["scores"],
                    "priority": self.analyze_priority(review),
                    "timestamp": time.time(),
                }
            )

        return results

    def generate_insights(self, results):
        """Generate business insights from processed reviews."""
        total = len(results)

        # Sentiment distribution
        sentiment_counts = {}
        for r in results:
            sentiment = r["sentiment"]
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        # Category distribution
        category_counts = {}
        for r in results:
            for category in r["categories"]:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Priority distribution
        priority_counts = {}
        for r in results:
            priority = r["priority"]
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        insights = {
            "total_reviews": total,
            "sentiment_distribution": {
                k: {"count": v, "percentage": v / total * 100}
                for k, v in sentiment_counts.items()
            },
            "top_categories": sorted(
                category_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "priority_distribution": priority_counts,
            "urgent_reviews": [
                r for r in results if r["priority"] in ["urgent", "high"]
            ][:5],
        }

        return insights


def demonstrate_production_workflow():
    """Demonstrate a complete production workflow."""

    print("=== Retail Review Analysis Workflow ===\n")

    # Initialize workflow
    workflow = RetailReviewWorkflow()

    # Sample reviews (in production, these would come from a database/stream)
    sample_reviews = [
        "This product is absolutely amazing! Fast shipping and great quality.",
        "Terrible experience. The item arrived broken and customer service "
        "was unhelpful.",
        "Good value for money, but packaging could be better.",
        "URGENT: Received wrong item. Need immediate refund!",
        "The product works as described. Delivery was on time.",
        "Worst purchase ever! Complete waste of money. Want my refund NOW!",
        "Nice product but a bit overpriced compared to competitors.",
        "Package was damaged during shipping but product was fine.",
        "Excellent customer service! They resolved my issue quickly.",
        "Product quality is okay but not worth the premium price.",
        "Love it! Exceeded my expectations in every way.",
        "Shipping took forever and no tracking updates provided.",
    ]

    print(f"Processing {len(sample_reviews)} reviews...\n")

    # Process reviews
    start_time = time.time()
    results = workflow.process_batch(sample_reviews)
    process_time = time.time() - start_time

    print(
        f"Processed in {process_time:.2f}s "
        f"({len(sample_reviews)/process_time:.1f} reviews/sec)\n"
    )

    # Generate insights
    insights = workflow.generate_insights(results)

    print("=== Business Insights ===\n")

    print("1. Sentiment Distribution:")
    for sentiment, data in insights["sentiment_distribution"].items():
        print(f"   {sentiment}: {data['count']} ({data['percentage']:.1f}%)")

    print("\n2. Top Categories:")
    for category, count in insights["top_categories"]:
        print(f"   {category}: {count} mentions")

    print("\n3. Priority Distribution:")
    for priority, count in insights["priority_distribution"].items():
        print(f"   {priority}: {count} reviews")

    print("\n4. Urgent Reviews Requiring Attention:")
    for review in insights["urgent_reviews"]:
        print(f"\n   Priority: {review['priority'].upper()}")
        print(f"   Text: \"{review['original_text'][:100]}...\"")
        print(f"   Sentiment: {review['sentiment']} ({review['sentiment_score']:.2f})")
        print(f"   Categories: {', '.join(review['categories'])}")

    print("\n=== Workflow Performance ===")
    print(f"Total processing time: {process_time:.2f}s")
    print(f"Average time per review: {process_time/len(sample_reviews)*1000:.1f}ms")
    print(f"Throughput: {len(sample_reviews)/process_time:.1f} reviews/second")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save detailed results
    with open(output_dir / "review_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save insights
    with open(output_dir / "business_insights.json", "w") as f:
        json.dump(insights, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    print("\n=== Next Steps ===")
    print("1. Connect to real-time data stream (Kafka, SQS)")
    print("2. Set up automated alerts for urgent reviews")
    print("3. Create dashboard for insights visualization")
    print("4. Implement model monitoring and retraining")
    print("5. Add A/B testing for model improvements")


if __name__ == "__main__":
    demonstrate_production_workflow()
