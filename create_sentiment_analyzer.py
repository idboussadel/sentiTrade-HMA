content = '''\"\"\"
Sentiment Analysis with FinBERT for financial text.
\"\"\"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinBERTSentimentAnalyzer:
    \"\"\"
    Sentiment analyzer using FinBERT model.
    Supports 4-bit quantization for efficient inference.
    \"\"\"
    
    def __init__(
        self,
        model_name: str = \"ProsusAI/finbert\",
        use_quantization: bool = True,
        device: str = None
    ):
        \"\"\"
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
            use_quantization: Use 4-bit quantization for efficiency
            device: Device to use (cuda/cpu). Auto-detect if None.
        \"\"\"
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        # Detect device
        if device is None:
            self.device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
        else:
            self.device = device
            
        logger.info(f\"Initializing FinBERT on device: {self.device}\")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f\" Tokenizer loaded: {model_name}\")
        
        # Load model with quantization if supported
        if use_quantization and self.device == \"cuda\" and torch.cuda.is_available():
            logger.info(\"Loading model with 4-bit quantization...\")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type=\"nf4\"
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=\"auto\"
                )
                logger.info(\" Model loaded with 4-bit quantization\")
            except Exception as e:
                logger.warning(f\"Quantization failed: {e}. Loading standard model...\")
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
        else:
            logger.info(\"Loading standard model...\")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(\" Model loaded\")
        
        self.model.eval()
        
        # Label mapping
        self.id2label = {0: \"negative\", 1: \"neutral\", 2: \"positive\"}
        self.label2score = {\"negative\": -1.0, \"neutral\": 0.0, \"positive\": 1.0}
        
    def analyze_text(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Dict[str, Union[str, float, Dict]]:
        \"\"\"
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            return_all_scores: Return scores for all labels
            
        Returns:
            Dictionary with sentiment label, score, and optionally all scores
        \"\"\"
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors=\"pt\",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get results
        scores = predictions[0].cpu().numpy()
        label_id = np.argmax(scores)
        label = self.id2label[label_id]
        confidence = float(scores[label_id])
        sentiment_score = self.label2score[label]
        
        result = {
            \"label\": label,
            \"score\": sentiment_score,
            \"confidence\": confidence
        }
        
        if return_all_scores:
            result[\"all_scores\"] = {
                self.id2label[i]: float(scores[i]) for i in range(len(scores))
            }
        
        return result
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        \"\"\"
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            List of sentiment results
        \"\"\"
        results = []
        
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors=\"pt\",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            scores = predictions.cpu().numpy()
            
            for j in range(len(batch_texts)):
                label_id = np.argmax(scores[j])
                label = self.id2label[label_id]
                confidence = float(scores[j][label_id])
                sentiment_score = self.label2score[label]
                
                results.append({
                    \"label\": label,
                    \"score\": sentiment_score,
                    \"confidence\": confidence
                })
            
            if show_progress:
                batch_num = i // batch_size + 1
                logger.info(f\"  Processed batch {batch_num}/{num_batches}\")
        
        return results
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = \"text\",
        batch_size: int = 8
    ) -> pd.DataFrame:
        \"\"\"
        Analyze sentiment for texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with added sentiment columns
        \"\"\"
        logger.info(f\"Analyzing sentiment for {len(df)} texts...\")
        
        # Get texts
        texts = df[text_column].fillna(\"\").tolist()
        
        # Analyze
        results = self.analyze_batch(texts, batch_size=batch_size)
        
        # Add to DataFrame
        df = df.copy()
        df[\"sentiment_label\"] = [r[\"label\"] for r in results]
        df[\"sentiment_score\"] = [r[\"score\"] for r in results]
        df[\"sentiment_confidence\"] = [r[\"confidence\"] for r in results]
        
        logger.info(f\" Sentiment analysis complete\")
        logger.info(f\"  Distribution: {df['sentiment_label'].value_counts().to_dict()}\")
        
        return df


class SentimentAggregator:
    \"\"\"Aggregate sentiment scores by ticker and date.\"\"\"
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def aggregate_daily(
        self,
        sentiment_df: pd.DataFrame,
        method: str = \"weighted_mean\"
    ) -> pd.DataFrame:
        \"\"\"
        Aggregate sentiment scores by ticker and date.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            method: Aggregation method (mean, median, weighted_mean)
            
        Returns:
            Aggregated DataFrame
        \"\"\"
        sentiment_df = sentiment_df.copy()
        sentiment_df[\"date\"] = pd.to_datetime(sentiment_df[\"date\"])
        
        if method == \"mean\":
            agg_df = sentiment_df.groupby([\"ticker\", \"date\"]).agg({
                \"sentiment_score\": \"mean\",
                \"sentiment_confidence\": \"mean\"
            }).reset_index()
            
        elif method == \"median\":
            agg_df = sentiment_df.groupby([\"ticker\", \"date\"]).agg({
                \"sentiment_score\": \"median\",
                \"sentiment_confidence\": \"mean\"
            }).reset_index()
            
        elif method == \"weighted_mean\":
            def weighted_avg(group):
                weights = group[\"sentiment_confidence\"]
                return np.average(group[\"sentiment_score\"], weights=weights)
            
            agg_df = sentiment_df.groupby([\"ticker\", \"date\"]).apply(
                lambda x: pd.Series({
                    \"sentiment_score\": weighted_avg(x),
                    \"sentiment_confidence\": x[\"sentiment_confidence\"].mean()
                })
            ).reset_index()
        
        return agg_df
    
    def merge_with_financial(
        self,
        financial_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        \"\"\"
        Merge sentiment with financial data.
        
        Args:
            financial_df: Financial data DataFrame
            sentiment_df: Sentiment data DataFrame
            
        Returns:
            Merged DataFrame
        \"\"\"
        # Prepare DataFrames
        financial_df = financial_df.copy()
        sentiment_df = sentiment_df.copy()
        
        financial_df[\"Date\"] = pd.to_datetime(financial_df[\"Date\"])
        sentiment_df[\"date\"] = pd.to_datetime(sentiment_df[\"date\"])
        
        # Merge
        merged = financial_df.merge(
            sentiment_df,
            left_on=[\"Ticker\", \"Date\"],
            right_on=[\"ticker\", \"date\"],
            how=\"left\"
        )
        
        # Fill missing sentiment with neutral
        merged[\"sentiment_score\"] = merged[\"sentiment_score\"].fillna(0.0)
        merged[\"sentiment_confidence\"] = merged[\"sentiment_confidence\"].fillna(0.5)
        
        # Drop duplicate columns
        merged = merged.drop(columns=[\"ticker\", \"date\"], errors=\"ignore\")
        
        self.logger.info(f\" Merged financial and sentiment data\")
        self.logger.info(f\"  Total rows: {len(merged)}\")
        self.logger.info(f\"  Rows with sentiment: {merged['sentiment_score'].notna().sum()}\")
        
        return merged


if __name__ == \"__main__\":
    # Test the sentiment analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Test single text
    test_text = \"Apple stock surges on strong earnings report and positive outlook.\"
    result = analyzer.analyze_text(test_text, return_all_scores=True)
    
    print(f\"Text: {test_text}\")
    print(f\"Sentiment: {result['label']}\")
    print(f\"Score: {result['score']}\")
    print(f\"Confidence: {result['confidence']:.2%}\")
    print(f\"All scores: {result['all_scores']}\")
'''

with open('src/models/sentiment_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(' sentiment_analyzer.py créé')
