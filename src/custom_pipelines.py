"""Custom pipeline creation and composition examples."""

import time

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

# Note: register_pipeline API has changed in newer versions
# For demo purposes, we'll show the concept without actual registration

from src.config import BATCH_SIZE, DEFAULT_NER_MODEL, DEFAULT_SENTIMENT_MODEL, DEVICE


class CustomSentimentPipeline(Pipeline):
    """Custom sentiment pipeline with preprocessing and business logic."""
    
    def _sanitize_parameters(self, **kwargs):
        """Separate preprocessing, forward, and postprocessing parameters."""
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        # Extract custom parameters
        if "normalize_text" in kwargs:
            preprocess_kwargs["normalize_text"] = kwargs.pop("normalize_text")
        if "confidence_threshold" in kwargs:
            postprocess_kwargs["confidence_threshold"] = kwargs.pop("confidence_threshold")
            
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, inputs, normalize_text=True):
        """Clean and normalize text before processing."""
        # Handle both single string and list inputs
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, dict):
            # Handle dict input from pipeline
            inputs = inputs.get("text", inputs.get("inputs", []))
            if isinstance(inputs, str):
                inputs = [inputs]

        cleaned = []
        for text in inputs:
            if normalize_text:
                # Remove HTML tags
                import re

                text = re.sub(r"<.*?>", "", text)

                # Normalize
                text = text.lower().strip()

                # Remove excessive punctuation
                text = re.sub(r"[!?]{2,}", "!", text)

            cleaned.append(text)

        # Return in format expected by parent class
        return super().preprocess(cleaned)

    def postprocess(self, outputs):
        """Add business logic to outputs."""
        results = super().postprocess(outputs)

        for result in results:
            # Add confidence level
            score = result["score"]
            if score > 0.95:
                result["confidence"] = "high"
            elif score > 0.8:
                result["confidence"] = "medium"
            else:
                result["confidence"] = "low"

            # Add action recommendation
            if result["label"] == "NEGATIVE" and score > 0.9:
                result["action"] = "urgent_review"
            elif result["label"] == "NEGATIVE":
                result["action"] = "review"
            else:
                result["action"] = "none"

        return results


class SentimentNERPipeline(Pipeline):
    """Composite pipeline for sentiment + entity recognition."""

    def __init__(self, sentiment_pipeline, ner_pipeline, **kwargs):
        self.sentiment_pipeline = sentiment_pipeline
        self.ner_pipeline = ner_pipeline
        super().__init__(
            model=sentiment_pipeline.model,
            tokenizer=sentiment_pipeline.tokenizer,
            **kwargs,
        )

    def _forward(self, model_inputs):
        """Run both pipelines and combine results."""
        # Get the original text
        texts = model_inputs.get("texts", [])

        # Run sentiment analysis
        sentiment_results = self.sentiment_pipeline(texts)

        # Run NER
        ner_results = self.ner_pipeline(texts)

        # Combine results
        combined = []
        for i, text in enumerate(texts):
            combined.append(
                {
                    "text": text,
                    "sentiment": (
                        sentiment_results[i] if i < len(sentiment_results) else None
                    ),
                    "entities": ner_results[i] if i < len(ner_results) else [],
                }
            )

        return combined

    def preprocess(self, inputs):
        """Store original text for later use."""
        if isinstance(inputs, str):
            inputs = [inputs]
        return {"texts": inputs}

    def postprocess(self, outputs):
        """Return combined results."""
        return outputs


def demonstrate_custom_pipelines():
    """Demonstrate custom pipeline creation and usage."""

    print("1. Standard Pipeline Baseline")
    print("-" * 40)

    try:
        # Standard pipeline
        standard_pipe = pipeline(
            "sentiment-analysis",
            model=DEFAULT_SENTIMENT_MODEL,
            device=0 if DEVICE == "cuda" else -1,
        )
    except Exception as e:
        print(f"Error loading model {DEFAULT_SENTIMENT_MODEL}: {e}")
        print("Falling back to default model...")
        standard_pipe = pipeline(
            "sentiment-analysis",
            device=0 if DEVICE == "cuda" else -1,
        )

    test_texts = [
        "This product is absolutely amazing! Best purchase ever!!!",
        "Terrible quality... Broke after 2 days :(",
        "<p>Good product overall.</p> Would recommend.",
        "The service was okay, nothing special.",
    ]

    start = time.time()
    standard_results = standard_pipe(test_texts, batch_size=BATCH_SIZE)
    standard_time = time.time() - start

    print(f"Standard pipeline results ({standard_time:.3f}s):")
    for text, result in zip(test_texts, standard_results, strict=False):
        print(f"  '{text[:50]}...' -> {result}")

    print("\n2. Custom Sentiment Pipeline")
    print("-" * 40)

    # Create custom pipeline
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_SENTIMENT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SENTIMENT_MODEL)

    custom_pipe = CustomSentimentPipeline(
        model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1
    )

    start = time.time()
    custom_results = custom_pipe(test_texts, batch_size=BATCH_SIZE)
    custom_time = time.time() - start

    print(f"Custom pipeline results ({custom_time:.3f}s):")
    for text, result in zip(test_texts, custom_results, strict=False):
        print(f"  '{text[:30]}...' ->")
        print(f"    Label: {result['label']}, Score: {result['score']:.3f}")
        print(f"    Confidence: {result['confidence']}, Action: {result['action']}")

    print("\n3. Composite Pipeline (Sentiment + NER)")
    print("-" * 40)

    # Note: In newer versions of transformers, the registration API has changed
    # For demonstration, we'll create the pipeline directly
    # register_pipeline(
    #     task="sentiment-ner", pipeline_class=SentimentNERPipeline, pt_model=True
    # )

    # Create component pipelines
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=DEFAULT_SENTIMENT_MODEL,
        device=0 if DEVICE == "cuda" else -1,
    )

    ner_pipe = pipeline(
        "ner", model=DEFAULT_NER_MODEL, device=0 if DEVICE == "cuda" else -1
    )

    # Create composite pipeline
    composite_pipe = SentimentNERPipeline(
        sentiment_pipeline=sentiment_pipe, ner_pipeline=ner_pipe
    )

    business_texts = [
        "Apple Inc. makes amazing products! Tim Cook is a visionary.",
        "Amazon's delivery was terrible. Jeff Bezos should fix this.",
        "Microsoft Teams worked well for our meeting in Seattle.",
    ]

    start = time.time()
    composite_results = composite_pipe(business_texts)
    composite_time = time.time() - start

    print(f"Composite pipeline results ({composite_time:.3f}s):")
    for result in composite_results:
        print(f"\nText: '{result['text'][:60]}...'")
        print(f"Sentiment: {result['sentiment']}")
        if result["entities"]:
            print("Entities found:")
            for entity in result["entities"]:
                print(f"  - {entity['word']} ({entity['entity']})")

    print("\n4. Performance Comparison")
    print("-" * 40)
    print(f"Standard pipeline: {standard_time:.3f}s")
    print(f"Custom pipeline: {custom_time:.3f}s")
    print(f"Composite pipeline: {composite_time:.3f}s")
    print(
        f"\nCustom preprocessing overhead: {((custom_time/standard_time)-1)*100:.1f}%"
    )


if __name__ == "__main__":
    demonstrate_custom_pipelines()
