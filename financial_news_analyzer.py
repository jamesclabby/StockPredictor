#!/usr/bin/env python3
"""
Financial News Analysis Agent

A simple Python script that fetches financial news for specified stock tickers
and performs sentiment analysis on the news headlines.

Requirements:
- Alpha Vantage API key (stored in .env file)
- Python 3.9+
- Required packages: requests, python-dotenv, transformers, torch
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from transformers import pipeline
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ErrorType(Enum):
    """Enumeration of different error types for better error handling."""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_ERROR = "auth_error"
    PARSING_ERROR = "parsing_error"
    MODEL_ERROR = "model_error"
    CONFIGURATION_ERROR = "config_error"
    UNKNOWN_ERROR = "unknown_error"


class APIError(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, message: str, error_type: ErrorType, retry_after: Optional[int] = None):
        self.message = message
        self.error_type = error_type
        self.retry_after = retry_after
        super().__init__(self.message)


def retry_with_exponential_backoff(
    func, 
    max_retries: int = 3, 
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except APIError as e:
            if e.error_type == ErrorType.RATE_LIMIT:
                # For rate limits, use the retry_after value if provided
                wait_time = e.retry_after if e.retry_after else min(
                    base_delay * (backoff_factor ** attempt), 
                    max_delay
                )
                print(f"⚠️  Rate limit hit. Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            elif e.error_type in [ErrorType.NETWORK_ERROR, ErrorType.API_ERROR]:
                if attempt < max_retries:
                    wait_time = min(base_delay * (backoff_factor ** attempt), max_delay)
                    print(f"⚠️  API error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                # Don't retry for auth errors, config errors, etc.
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = min(base_delay * (backoff_factor ** attempt), max_delay)
                print(f"⚠️  Unexpected error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise


class CircuitBreaker:
    """Circuit breaker pattern implementation for API calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


def load_api_key():
    """Load the Alpha Vantage API key from .env file with enhanced validation."""
    load_dotenv()
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        raise APIError(
            "Please set your ALPHA_VANTAGE_API_KEY in the .env file. "
            "Get a free API key from: https://www.alphavantage.co/support/#api-key",
            ErrorType.CONFIGURATION_ERROR
        )
    
    # Basic API key format validation
    if len(api_key) < 10:
        raise APIError(
            "API key appears to be invalid (too short). Please check your .env file.",
            ErrorType.CONFIGURATION_ERROR
        )
    
    return api_key


def fetch_news_with_retry(ticker: str, api_key: str, circuit_breaker: CircuitBreaker) -> List[Dict]:
    """
    Fetch news articles with retry logic and circuit breaker.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key
        circuit_breaker: Circuit breaker instance
        
    Returns:
        List of news articles
    """
    def _fetch_news():
        if not circuit_breaker.can_execute():
            raise APIError(
                f"Circuit breaker is OPEN for {ticker}. API appears to be down.",
                ErrorType.API_ERROR
            )
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': api_key
        }
        
        try:
            print(f"🔍 DEBUG: Making API call to Alpha Vantage for {ticker}")
            print(f"🔍 DEBUG: URL: {url}")
            print(f"🔍 DEBUG: Params: {params}")
            response = requests.get(url, params=params, timeout=30)
            print(f"🔍 DEBUG: Response status: {response.status_code}")
            
            # Handle HTTP errors
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise APIError(
                    f"Rate limit exceeded for {ticker}",
                    ErrorType.RATE_LIMIT,
                    retry_after=retry_after
                )
            elif response.status_code == 401:
                raise APIError(
                    f"Authentication failed for {ticker}. Check your API key.",
                    ErrorType.AUTHENTICATION_ERROR
                )
            elif response.status_code >= 500:
                raise APIError(
                    f"Server error {response.status_code} for {ticker}",
                    ErrorType.API_ERROR
                )
            
            response.raise_for_status()
            
            try:
                data = response.json()
                print(f"🔍 DEBUG: Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                if 'feed' in data:
                    print(f"🔍 DEBUG: Found {len(data['feed'])} articles in feed")
                else:
                    print(f"🔍 DEBUG: No 'feed' key in response")
            except json.JSONDecodeError as e:
                print(f"🔍 DEBUG: JSON decode error: {e}")
                raise APIError(
                    f"Invalid JSON response for {ticker}: {e}",
                    ErrorType.PARSING_ERROR
                )
            
            # Check for API-specific errors
            if 'Error Message' in data:
                error_msg = data['Error Message']
                print(f"🔍 DEBUG: API Error Message for {ticker}: {error_msg}")
                if 'Invalid API call' in error_msg:
                    raise APIError(
                        f"Invalid API call for {ticker}: {error_msg}",
                        ErrorType.API_ERROR
                    )
                else:
                    raise APIError(
                        f"API error for {ticker}: {error_msg}",
                        ErrorType.API_ERROR
                    )
            
            # Check for rate limit messages
            if 'Note' in data:
                note_msg = data['Note']
                print(f"🔍 DEBUG: API Note for {ticker}: {note_msg}")
                if 'API call frequency' in note_msg or 'rate limit' in note_msg.lower():
                    raise APIError(
                        f"Rate limit exceeded for {ticker}: {note_msg}",
                        ErrorType.RATE_LIMIT
                    )
            
            # Check for information messages about rate limits
            if 'Information' in data:
                info_msg = data['Information']
                print(f"🔍 DEBUG: API Information for {ticker}: {info_msg}")
                if 'rate limit' in info_msg.lower() or 'requests per day' in info_msg.lower():
                    raise APIError(
                        f"Daily rate limit exceeded for {ticker}: {info_msg}",
                        ErrorType.RATE_LIMIT
                    )
                # Extract retry time from note if possible
                    retry_after = 60  # Default to 1 minute
                    if 'per minute' in note:
                        retry_after = 60
                    elif 'per day' in note:
                        retry_after = 86400  # 24 hours
                    
                    raise APIError(
                        f"Rate limit reached for {ticker}: {note}",
                        ErrorType.RATE_LIMIT,
                        retry_after=retry_after
                    )
                else:
                    raise APIError(
                        f"API note for {ticker}: {note}",
                        ErrorType.API_ERROR
                    )
            
            # Success - record it in circuit breaker
            circuit_breaker.record_success()
            
            articles = data.get('feed', [])
            print(f"✓ Successfully fetched {len(articles)} articles for {ticker}")
            if not articles:
                print(f"🔍 DEBUG: No articles in feed for {ticker}. Full response: {json.dumps(data, indent=2)[:500]}...")
            return articles
            
        except requests.exceptions.Timeout:
            raise APIError(
                f"Request timeout for {ticker}",
                ErrorType.NETWORK_ERROR
            )
        except requests.exceptions.ConnectionError:
            raise APIError(
                f"Connection error for {ticker}",
                ErrorType.NETWORK_ERROR
            )
        except requests.exceptions.RequestException as e:
            raise APIError(
                f"Request error for {ticker}: {e}",
                ErrorType.NETWORK_ERROR
            )
        except APIError:
            # Re-raise API errors
            raise
        except Exception as e:
            raise APIError(
                f"Unexpected error fetching news for {ticker}: {e}",
                ErrorType.UNKNOWN_ERROR
            )
    
    try:
        return retry_with_exponential_backoff(_fetch_news)
    except APIError as e:
        circuit_breaker.record_failure()
        print(f"❌ Failed to fetch news for {ticker} after retries: {e.message}")
        raise


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('financial_news_analyzer.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def analyze_sentiment_robust(text: str, sentiment_pipeline) -> Dict:
    """
    Analyze sentiment with robust error handling.
    
    Args:
        text: Text to analyze
        sentiment_pipeline: Pre-loaded sentiment analysis pipeline
        
    Returns:
        Sentiment analysis result
    """
    logger = logging.getLogger(__name__)
    
    if not text or not text.strip():
        logger.warning("Empty text provided for sentiment analysis")
        return {'label': 'UNKNOWN', 'score': 0.0}
    
    try:
        # Truncate very long text to avoid model issues
        if len(text) > 512:
            text = text[:512]
            logger.info("Text truncated to 512 characters for sentiment analysis")
        
        result = sentiment_pipeline(text)
        return result[0]  # Return the first (and only) result
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {'label': 'UNKNOWN', 'score': 0.0}


def handle_api_error(error: APIError, ticker: str) -> None:
    """
    Handle API errors with appropriate user messaging.
    
    Args:
        error: The API error that occurred
        ticker: The ticker that caused the error
    """
    logger = logging.getLogger(__name__)
    
    if error.error_type == ErrorType.RATE_LIMIT:
        if error.retry_after:
            if error.retry_after > 3600:  # More than 1 hour
                print(f"⚠️  Rate limit reached for {ticker}. Please try again tomorrow.")
            else:
                print(f"⚠️  Rate limit reached for {ticker}. Please wait {error.retry_after} seconds.")
        else:
            print(f"⚠️  Rate limit reached for {ticker}. Please try again later.")
    elif error.error_type == ErrorType.AUTHENTICATION_ERROR:
        print(f"❌ Authentication failed for {ticker}. Please check your API key.")
    elif error.error_type == ErrorType.NETWORK_ERROR:
        print(f"🌐 Network error for {ticker}. Please check your internet connection.")
    elif error.error_type == ErrorType.API_ERROR:
        print(f"🔧 API error for {ticker}: {error.message}")
    else:
        print(f"❓ Unexpected error for {ticker}: {error.message}")
    
    logger.error(f"API Error for {ticker}: {error.message} (Type: {error.error_type})")


def is_within_24_hours(time_published):
    """
    Check if the given time is within the last 24 hours.
    
    Args:
        time_published (str): Time in format 'YYYYMMDDTHHMMSS'
        
    Returns:
        bool: True if within last 24 hours, False otherwise
    """
    try:
        # Parse the time string
        article_time = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
        
        # Calculate 24 hours ago
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        
        return article_time >= twenty_four_hours_ago
        
    except ValueError:
        print(f"Error parsing time: {time_published}")
        return False


def format_sentiment_output(sentiment_result):
    """
    Format sentiment result for display.
    
    Args:
        sentiment_result (dict): Sentiment analysis result
        
    Returns:
        str: Formatted sentiment label
    """
    label = sentiment_result['label']
    score = sentiment_result['score']
    
    if label == 'POSITIVE':
        return f"[POSITIVE] (confidence: {score:.2f})"
    elif label == 'NEGATIVE':
        return f"[NEGATIVE] (confidence: {score:.2f})"
    else:
        return f"[{label}] (confidence: {score:.2f})"


def main():
    """Main execution function with robust error handling."""
    # Set up logging
    logger = setup_logging()
    
    print("=== Financial News Analysis Agent (Robust Version) ===\n")
    
    # Initialize circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
    
    # Load API key
    try:
        api_key = load_api_key()
        logger.info("API key loaded successfully")
    except APIError as e:
        print(f"❌ Configuration Error: {e.message}")
        logger.error(f"Configuration error: {e.message}")
        return
    
    # Hardcoded list of stock tickers to analyze
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    
    # Initialize sentiment analysis pipeline once
    print("Initializing sentiment analysis model...")
    try:
        sentiment_pipeline = pipeline(
            'sentiment-analysis', 
            model='distilbert-base-uncased-finetuned-sst-2-english'
        )
        print("✓ Sentiment analysis model loaded successfully\n")
        logger.info("Sentiment analysis model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading sentiment analysis model: {e}")
        logger.error(f"Error loading sentiment analysis model: {e}")
        return
    
    # Store results for each ticker
    results = {}
    successful_tickers = 0
    failed_tickers = 0
    
    # Process each ticker
    for ticker in tickers:
        try:
            # Fetch news articles with retry logic
            articles = fetch_news_with_retry(ticker, api_key, circuit_breaker)
            
            if not articles:
                print(f"ℹ️  No articles found for {ticker}\n")
                results[ticker] = []
                continue
            
            print(f"✓ Found {len(articles)} articles for {ticker}")
            
            # Filter articles from last 24 hours
            recent_articles = []
            for article in articles:
                time_published = article.get('time_published', '')
                if time_published and is_within_24_hours(time_published):
                    recent_articles.append(article)
            
            print(f"✓ Found {len(recent_articles)} articles from last 24 hours")
            
            # Analyze sentiment for each recent article
            ticker_results = []
            for article in recent_articles:
                title = article.get('title', '')
                if title:
                    # Analyze sentiment
                    sentiment_result = analyze_sentiment_robust(title, sentiment_pipeline)
                    
                    # Store results for positive and negative sentiments
                    if sentiment_result['label'] in ['POSITIVE', 'NEGATIVE']:
                        ticker_results.append({
                            'title': title,
                            'sentiment': sentiment_result,
                            'time_published': article.get('time_published', ''),
                            'source': article.get('source', 'Unknown')
                        })
            
            results[ticker] = ticker_results
            successful_tickers += 1
            print(f"✓ Analyzed {len(ticker_results)} articles with clear sentiment\n")
            
        except APIError as e:
            handle_api_error(e, ticker)
            results[ticker] = []
            failed_tickers += 1
            print()  # Empty line for readability
        except Exception as e:
            print(f"❓ Unexpected error processing {ticker}: {e}")
            logger.error(f"Unexpected error processing {ticker}: {e}")
            results[ticker] = []
            failed_tickers += 1
            print()  # Empty line for readability
    
    # Print summary statistics
    print(f"📊 Processing Summary: {successful_tickers} successful, {failed_tickers} failed\n")
    
    # Print formatted summary
    print("--- Daily Financial News Summary ---\n")
    
    for ticker, articles in results.items():
        if not articles:
            print(f"Ticker: {ticker}")
            print("No recent articles with clear sentiment found.\n")
            continue
        
        print(f"Ticker: {ticker}")
        for article in articles:
            sentiment_label = format_sentiment_output(article['sentiment'])
            print(f"{sentiment_label} {article['title']}")
        print()  # Empty line for readability


if __name__ == "__main__":
    main()
