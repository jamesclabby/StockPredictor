#!/usr/bin/env python3
"""
Google Cloud Function for Daily Financial News Analysis
Refactored from Streamlit app to send email summaries using Resend
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Third-party imports
import requests
import resend
from google.cloud import secretmanager

# Local imports
from financial_news_analyzer import (
    load_api_key, fetch_news_with_retry, analyze_sentiment_robust,
    setup_logging, CircuitBreaker, APIError, ErrorType, is_within_24_hours
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded tickers for analysis
TICKERS = ['AAPL', 'GOOGL', 'TSLA']

# Email configuration (hardcoded for now)
SENDER_EMAIL = "onboarding@resend.dev"  # Sender
RECIPIENT_EMAIL = "jamesclabby12@gmail.com"  # Recipient
EMAIL_SUBJECT = "Daily Financial News Analysis"

def get_secret(secret_name: str) -> str:
    """
    Retrieve secrets from Google Cloud Secret Manager or environment variables.
    
    Args:
        secret_name: Name of the secret to retrieve
        
    Returns:
        Secret value as string
        
    Raises:
        ValueError: If secret is not found
    """
    try:
        # Try to get from Google Cloud Secret Manager first
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        
        if project_id:
            secret_name_full = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": secret_name_full})
            secret_value = response.payload.data.decode("UTF-8")
            logger.info(f"Retrieved secret {secret_name} from Secret Manager")
            return secret_value
    except Exception as e:
        logger.warning(f"Failed to retrieve {secret_name} from Secret Manager: {e}")
    
    # Fallback to environment variables for local testing
    secret_value = os.getenv(secret_name)
    if secret_value:
        logger.info(f"Retrieved secret {secret_name} from environment variables")
        return secret_value
    
    raise ValueError(f"Secret {secret_name} not found in Secret Manager or environment variables")

def send_summary_email(summary_html: str) -> bool:
    """
    Send the analysis summary via email using Resend.
    
    Args:
        summary_html: HTML formatted summary to send
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Get Resend API key
        resend_api_key = get_secret('RESEND_API_KEY')
        
        # Set the API key
        resend.api_key = resend_api_key
        
        # Create email message
        params = {
            "from": SENDER_EMAIL,
            "to": [RECIPIENT_EMAIL],
            "subject": EMAIL_SUBJECT,
            "html": summary_html
        }
        
        # Send email
        response = resend.Emails.send(params)
        
        if response and 'id' in response:
            logger.info(f"Email sent successfully. ID: {response['id']}")
            return True
        else:
            logger.error(f"Failed to send email. Response: {response}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

def format_analysis_summary(results: Dict) -> str:
    """
    Format the analysis results as HTML for email.
    
    Args:
        results: Dictionary containing analysis results for each ticker
        
    Returns:
        HTML formatted string
    """
    html_parts = [
        "<html><body>",
        "<h1>📈 Daily Financial News Analysis</h1>",
        f"<p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<hr>"
    ]
    
    for ticker, data in results.items():
        if 'error' in data:
            html_parts.append(f"<h2>❌ {ticker} - Error</h2>")
            html_parts.append(f"<p style='color: red;'>{data['error']}</p>")
            continue
            
        articles = data.get('articles', [])
        summary = data.get('summary', {})
        
        if not articles:
            html_parts.append(f"<h2>⚠️ {ticker} - No Recent Articles</h2>")
            html_parts.append("<p>No articles found in the last 24 hours.</p>")
            continue
        
        # Ticker summary
        positive_count = summary.get('positive', 0)
        negative_count = summary.get('negative', 0)
        avg_confidence = summary.get('avg_confidence', 0)
        
        html_parts.append(f"<h2>📊 {ticker} Analysis Summary</h2>")
        html_parts.append(f"<p><strong>Total Articles:</strong> {len(articles)}</p>")
        html_parts.append(f"<p><strong>Positive:</strong> {positive_count} | <strong>Negative:</strong> {negative_count}</p>")
        html_parts.append(f"<p><strong>Average Confidence:</strong> {avg_confidence:.1f}%</p>")
        
        # Individual articles
        html_parts.append("<h3>📰 Recent Articles</h3>")
        html_parts.append("<ul>")
        
        for article in articles:
            title = article.get('title', 'No title')
            sentiment = article.get('sentiment', {})
            label = sentiment.get('label', 'UNKNOWN')
            score = sentiment.get('score', 0) * 100
            
            # Color code sentiment
            color = "green" if label == "POSITIVE" else "red" if label == "NEGATIVE" else "gray"
            
            html_parts.append(
                f"<li style='margin-bottom: 10px;'>"
                f"<strong style='color: {color};'>{label}</strong> "
                f"({score:.1f}%) - {title}"
                f"</li>"
            )
        
        html_parts.append("</ul>")
        html_parts.append("<hr>")
    
    html_parts.extend([
        "<p><em>This analysis was generated automatically by the Financial News Analysis system.</em></p>",
        "</body></html>"
    ])
    
    return "\n".join(html_parts)

def analyze_tickers_cloud(tickers: List[str], api_key: str) -> Dict:
    """
    Analyze sentiment for multiple tickers (Cloud Functions version).
    
    Args:
        tickers: List of ticker symbols to analyze
        api_key: Alpha Vantage API key
        
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    circuit_breaker = CircuitBreaker()
    
    # Initialize sentiment model
    try:
        from transformers import pipeline
        sentiment_pipeline = pipeline('sentiment-analysis', 
                                    model='distilbert-base-uncased-finetuned-sst-2-english')
        logger.info("Sentiment analysis model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        return results
    
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        
        try:
            # Fetch news articles
            articles = fetch_news_with_retry(ticker, api_key, circuit_breaker)
            
            if not articles:
                logger.warning(f"No articles found for {ticker}")
                results[ticker] = {'articles': [], 'summary': {}}
                continue
            
            # Filter recent articles (last 24 hours)
            recent_articles = []
            for article in articles:
                time_published = article.get('time_published', '')
                if time_published and is_within_24_hours(time_published):
                    recent_articles.append(article)
            
            if not recent_articles:
                logger.warning(f"No recent articles found for {ticker}")
                results[ticker] = {'articles': [], 'summary': {}}
                continue
            
            # Analyze sentiment
            analyzed_articles = []
            sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0}
            total_confidence = 0
            
            for article in recent_articles:
                title = article.get('title', '')
                if title:
                    try:
                        sentiment_result = analyze_sentiment_robust(title, sentiment_pipeline)
                        
                        if sentiment_result['label'] in ['POSITIVE', 'NEGATIVE']:
                            analyzed_articles.append({
                                'title': title,
                                'sentiment': sentiment_result,
                                'time_published': article.get('time_published', ''),
                                'url': article.get('url', '')
                            })
                            
                            sentiment_counts[sentiment_result['label']] += 1
                            total_confidence += sentiment_result['score'] * 100
                    except Exception as e:
                        logger.warning(f"Error analyzing sentiment for article '{title[:50]}...': {e}")
                        continue
            
            # Calculate summary
            avg_confidence = total_confidence / len(analyzed_articles) if analyzed_articles else 0
            summary = {
                'total_articles': len(analyzed_articles),
                'positive': sentiment_counts['POSITIVE'],
                'negative': sentiment_counts['NEGATIVE'],
                'avg_confidence': avg_confidence,
                'sentiment_counts': sentiment_counts
            }
            
            results[ticker] = {
                'articles': analyzed_articles,
                'summary': summary
            }
            
            logger.info(f"Completed analysis for {ticker}: {len(analyzed_articles)} articles")
            
        except APIError as e:
            if e.error_type == ErrorType.RATE_LIMIT:
                logger.error(f"Rate limit exceeded for {ticker}")
                results[ticker] = {'articles': [], 'summary': {}, 'error': 'Rate limit exceeded'}
            else:
                logger.error(f"API error for {ticker}: {e.message}")
                results[ticker] = {'articles': [], 'summary': {}, 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error analyzing {ticker}: {e}")
            results[ticker] = {'articles': [], 'summary': {}, 'error': str(e)}
    
    return results

def run_daily_analysis(event, context):
    """
    Main entry point for Google Cloud Functions.
    
    Args:
        event: Event data from the trigger
        context: Context object with metadata about the event
        
    Returns:
        Success message
    """
    try:
        logger.info("Starting daily financial news analysis")
        
        # Get API key
        api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        
        # Run analysis
        results = analyze_tickers_cloud(TICKERS, api_key)
        
        # Format results as HTML
        summary_html = format_analysis_summary(results)
        
        # Send email
        email_sent = send_summary_email(summary_html)
        
        if email_sent:
            logger.info("Daily analysis completed successfully")
            return "Summary email sent successfully."
        else:
            logger.error("Failed to send summary email")
            return "Analysis completed but failed to send email."
            
    except Exception as e:
        logger.error(f"Error in daily analysis: {e}")
        return f"Error: {str(e)}"

# For local testing
if __name__ == "__main__":
    # Test the function locally
    result = run_daily_analysis(None, None)
    print(result)
