#!/usr/bin/env python3
"""
Google Cloud Function for Daily Financial News Analysis
Refactored from Cloud Run to Cloud Functions
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import resend
from google.cloud import secretmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded tickers for analysis
TICKERS = ['AAPL', 'GOOGL', 'TSLA']

# Email configuration
SENDER_EMAIL = "stocks@featureforge.dev"  # Your verified domain
RECIPIENT_EMAIL = "jamesclabby12@gmail.com"  # Your email
EMAIL_SUBJECT = "Daily Financial News Analysis"

# Custom exceptions
class ErrorType:
    RATE_LIMIT = "RATE_LIMIT"
    API_ERROR = "API_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

class APIError(Exception):
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)

def get_secret(secret_name: str) -> str:
    """
    Retrieve secrets from Google Cloud Secret Manager or environment variables.
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
            return secret_value.strip()
    except Exception as e:
        logger.warning(f"Failed to retrieve {secret_name} from Secret Manager: {e}")
    
    # Fallback to environment variables for local testing
    secret_value = os.getenv(secret_name)
    if secret_value:
        logger.info(f"Retrieved secret {secret_name} from environment variables")
        return secret_value.strip()
    
    raise ValueError(f"Secret {secret_name} not found in Secret Manager or environment variables")

def is_within_24_hours(time_str: str) -> bool:
    """
    Check if a time string is within the last 24 hours.
    """
    try:
        # Parse the time string (format: YYYYMMDDTHHMMSS)
        if len(time_str) >= 15:
            time_obj = datetime.strptime(time_str[:15], '%Y%m%dT%H%M%S')
            now = datetime.now()
            time_diff = now - time_obj
            return time_diff.total_seconds() < 24 * 3600  # 24 hours in seconds
        return False
    except Exception as e:
        logger.warning(f"Error parsing time string '{time_str}': {e}")
        return False

def fetch_news_simple(ticker: str, api_key: str) -> List[Dict]:
    """
    Fetch news articles for a ticker using Alpha Vantage API.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Information' in data:
            if 'rate limit' in data['Information'].lower():
                raise APIError(ErrorType.RATE_LIMIT, "API rate limit exceeded")
            else:
                raise APIError(ErrorType.API_ERROR, data['Information'])
        
        if 'Note' in data:
            if 'rate limit' in data['Note'].lower():
                raise APIError(ErrorType.RATE_LIMIT, "API rate limit exceeded")
            else:
                raise APIError(ErrorType.API_ERROR, data['Note'])
        
        # Extract articles
        articles = data.get('feed', [])
        logger.info(f"Fetched {len(articles)} articles for {ticker}")
        
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {ticker}: {e}")
        raise APIError(ErrorType.API_ERROR, f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {ticker}: {e}")
        raise APIError(ErrorType.API_ERROR, f"Invalid JSON response: {str(e)}")

def simple_sentiment_analysis(text: str) -> Dict[str, any]:
    """
    Simple keyword-based sentiment analysis to avoid heavy ML dependencies.
    """
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Positive keywords
    positive_keywords = [
        'positive', 'good', 'great', 'excellent', 'strong', 'growth', 'profit',
        'gain', 'rise', 'up', 'increase', 'success', 'win', 'beat', 'outperform',
        'bullish', 'optimistic', 'surge', 'rally', 'breakthrough', 'record',
        'milestone', 'achievement', 'expansion', 'acquisition', 'partnership'
    ]
    
    # Negative keywords
    negative_keywords = [
        'negative', 'bad', 'poor', 'weak', 'decline', 'loss', 'fall', 'down',
        'decrease', 'failure', 'lose', 'miss', 'underperform', 'bearish',
        'pessimistic', 'drop', 'crash', 'crisis', 'concern', 'risk', 'warning',
        'challenge', 'struggle', 'cut', 'layoff', 'bankruptcy', 'default'
    ]
    
    # Count positive and negative keywords
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        sentiment = 'POSITIVE'
        confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = 'NEGATIVE'
        confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
    else:
        sentiment = 'NEUTRAL'
        confidence = 0.5
    
    return {
        'label': sentiment,
        'score': confidence
    }

def send_summary_email(summary_html: str) -> bool:
    """
    Send the analysis summary via email using Resend.
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

def analyze_tickers_simple(tickers: List[str], api_key: str) -> Dict:
    """
    Analyze sentiment for multiple tickers using simple keyword analysis.
    """
    results = {}
    
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        
        try:
            # Fetch news articles
            articles = fetch_news_simple(ticker, api_key)
            
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
            
            # Analyze sentiment using simple keyword analysis
            analyzed_articles = []
            sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0}
            total_confidence = 0
            
            for article in recent_articles:
                title = article.get('title', '')
                if title:
                    try:
                        sentiment_result = simple_sentiment_analysis(title)
                        
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

def run_daily_analysis(event=None, context=None):
    """
    Main entry point for Google Cloud Functions.
    This function is triggered by Cloud Scheduler or HTTP.
    """
    try:
        logger.info("Starting daily financial news analysis")
        
        # Get API key
        api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        
        # Run analysis
        results = analyze_tickers_simple(TICKERS, api_key)
        
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