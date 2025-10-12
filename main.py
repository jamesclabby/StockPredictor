#!/usr/bin/env python3
"""
Google Cloud Function for Daily Financial News Analysis
Enhanced with LangChain ReAct Agent and custom tools
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import resend
from google.cloud import secretmanager

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_openai import ChatOpenAI
from langchain import hub

# Transformers for sentiment analysis
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded tickers for analysis while testing
TICKERS = ['AAPL', 'GOOGL', 'TSLA']

# Email configuration
SENDER_EMAIL = "stocks@featureforge.dev"  # Your verified domain
RECIPIENT_EMAIL = "jamesclabby12@gmail.com"  # Your email
EMAIL_SUBJECT = "Daily Financial News Analysis - AI Agent Report"

# Global variables for AI components (initialized when needed)
llm = None
sentiment_pipeline = None

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

def initialize_ai_components():
    """
    Initialize AI components (LLM and sentiment pipeline) when needed.
    """
    global llm, sentiment_pipeline
    
    if llm is None or sentiment_pipeline is None:
        logger.info("Initializing AI components...")
        try:
            # Get OpenAI API key
            openai_api_key = get_secret('OPENAI_API_KEY')
            
            # Initialize LLM
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=openai_api_key)
            
            # Initialize sentiment pipeline
            sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            
            logger.info("AI components initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            raise

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

# Custom LangChain Tools

@tool
def fetch_stock_news(ticker: str) -> str:
    """
    Fetches raw news headlines from Alpha Vantage. 
    IMPORTANT: This tool must ONLY extract and return the headline text, ignoring any pre-packaged sentiment scores.
    """
    try:
        api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Information' in data:
            if 'rate limit' in data['Information'].lower():
                return f"API rate limit exceeded for {ticker}"
            else:
                return f"API error: {data['Information']}"
        
        if 'Note' in data:
            if 'rate limit' in data['Note'].lower():
                return f"API rate limit exceeded for {ticker}"
            else:
                return f"API error: {data['Note']}"
        
        # Extract only the raw headlines, ignoring sentiment scores
        articles = data.get('feed', [])
        headlines = []
        
        for article in articles:
            title = article.get('title', '')
            if title:
                headlines.append(title)
        
        if not headlines:
            return f"No recent news headlines found for {ticker}"
        
        # Return headlines as a formatted string
        headlines_text = "\n".join([f"- {headline}" for headline in headlines[:3]])  # Limit to 3 most recent to reduce API calls
        return f"Recent news headlines for {ticker}:\n{headlines_text}"
        
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return f"Error fetching news for {ticker}: {str(e)}"

@tool
def analyze_headline_sentiment(headline: str) -> str:
    """
    Analyzes a headline's sentiment using a local Hugging Face transformers model. 
    Returns 'Positive' or 'Negative'.
    """
    try:
        results = sentiment_pipeline(headline)
        label = results[0]['label'].title()
        score = results[0]['score']
        return f"Sentiment: {label} (Confidence: {score:.2f})"
    except Exception as e:
        logger.error(f"Error analyzing sentiment for headline: {e}")
        return f"Error analyzing sentiment: {str(e)}"

@tool
def analyze_multiple_headlines(headlines_text: str) -> str:
    """
    Analyzes multiple headlines' sentiment in one batch to reduce API calls.
    Takes a newline-separated string of headlines and returns sentiment for each.
    """
    try:
        headlines = [h.strip() for h in headlines_text.split('\n') if h.strip()]
        if not headlines:
            return "No headlines to analyze"
        
        # Analyze all headlines in one batch
        results = sentiment_pipeline(headlines)
        
        # Format results
        sentiment_analysis = []
        for i, result in enumerate(results):
            label = result['label'].title()
            score = result['score']
            sentiment_analysis.append(f"Headline {i+1}: {label} (Confidence: {score:.2f})")
        
        return "Batch sentiment analysis:\n" + "\n".join(sentiment_analysis)
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        return f"Error in batch sentiment analysis: {str(e)}"

@tool
def get_fallback_tickers() -> str:
    """
    Provides a list of reliable fallback tickers when the market scanner fails or returns unsupported tickers.
    Returns major stocks that are known to work well with Alpha Vantage APIs.
    """
    fallback_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    return ', '.join(fallback_tickers)

@tool
def get_stock_performance(ticker: str) -> str:
    """
    Gets the latest stock price and daily change from Alpha Vantage's GLOBAL_QUOTE endpoint.
    """
    try:
        api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Information' in data:
            return f"API error: {data['Information']}"
        
        if 'Note' in data:
            return f"API error: {data['Note']}"
        
        quote = data.get('Global Quote', {})
        if not quote:
            return f"No price data available for {ticker}"
        
        price = quote.get('05. price', 'N/A')
        change = quote.get('09. change', 'N/A')
        change_percent = quote.get('10. change percent', 'N/A')
        
        return f"{ticker} - Current Price: ${price}, Day's Change: {change} ({change_percent})"
        
    except Exception as e:
        logger.error(f"Error fetching stock performance for {ticker}: {e}")
        return f"Error fetching stock performance for {ticker}: {str(e)}"

@tool
def scan_market_for_trending_tickers(dummy: str = "") -> str:
    """
    Scans the market using Financial Modeling Prep API to find the most actively traded stocks.
    Validates each ticker with Alpha Vantage news API and returns 6 working tickers.
    
    Args:
        dummy: Unused parameter (for LangChain compatibility)
    """
    try:
        # Get FMP data
        fmp_api_key = get_secret('FMP_API_KEY')
        fmp_url = f"https://financialmodelingprep.com/stable/most-actives?apikey={fmp_api_key}"
        
        response = requests.get(fmp_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Filter for NYSE and NASDAQ exchanges only
        filtered_stocks = [
            stock for stock in data 
            if stock.get('exchange') in ['NYSE', 'NASDAQ']
        ]
        
        if not filtered_stocks:
            logger.warning("No stocks found after filtering")
            return "No trending tickers available"
        
        # Get Alpha Vantage API key for validation
        av_api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        working_tickers = []
        
        # Try each ticker in order until we get 4 working ones (reduced to limit API calls)
        for stock in filtered_stocks:
            if len(working_tickers) >= 4:
                break
                
            ticker = stock['symbol']
            try:
                # Test if ticker works with Alpha Vantage news API
                test_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={av_api_key}"
                test_response = requests.get(test_url, timeout=10)
                test_data = test_response.json()
                
                # Debug: Log the response for the first few tickers
                if len(working_tickers) < 3:
                    logger.info(f"Debug - {ticker} response: {str(test_data)[:200]}...")
                
                # Check if ticker is valid
                if 'Information' not in test_data and 'Error Message' not in test_data:
                    # Check if it's a rate limit note (which is OK) vs invalid ticker
                    if 'Note' in test_data:
                        if 'rate limit' in test_data['Note'].lower():
                            # Rate limit is OK - ticker is valid but we hit rate limit
                            working_tickers.append(ticker)
                            logger.info(f"Validated ticker: {ticker} (rate limit hit)")
                        else:
                            logger.info(f"Skipping invalid ticker: {ticker} - {test_data['Note']}")
                    else:
                        # No errors - ticker is valid
                        working_tickers.append(ticker)
                        logger.info(f"Validated ticker: {ticker}")
                else:
                    logger.info(f"Skipping invalid ticker: {ticker}")
                    
            except Exception as e:
                logger.info(f"Skipping ticker {ticker} due to error: {e}")
                continue
        
        if working_tickers:
            result = ', '.join(working_tickers)
            logger.info(f"Found {len(working_tickers)} working tickers: {result}")
            return result
        else:
            logger.warning("No working tickers found from FMP data")
            return "No trending tickers available"
        
    except Exception as e:
        logger.error(f"Error in market scanner: {e}")
        logger.info("No trending tickers found, returning default message")
        return "No trending tickers available due to technical issues"


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

def format_agent_analysis_summary(agent_results: Dict) -> str:
    """
    Format the AI agent analysis results as HTML for email.
    """
    html_parts = [
        "<html><body>",
        "<h1>🤖 AI-Powered Market Trends Analysis</h1>",
        f"<p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<p><em>Powered by LangChain ReAct Agent with Market Scanner and custom financial analysis tools</em></p>",
        "<hr>"
    ]
    
    # Handle the new single market analysis format
    if 'market_analysis' in agent_results:
        analysis = agent_results['market_analysis']
        
        if 'error' in analysis:
            html_parts.append("<h2>❌ Market Analysis Error</h2>")
            html_parts.append(f"<p style='color: red;'>{analysis['error']}</p>")
        else:
            summary = analysis.get('summary', 'No analysis available')
            
            html_parts.append("<h2>📊 Daily Market Trends Analysis</h2>")
            html_parts.append("<div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;'>")
            html_parts.append("<p><strong>AI Agent Market Report:</strong></p>")
            # Convert newlines to HTML breaks for better formatting
            formatted_summary = summary.replace('\n', '<br>')
            html_parts.append(f"<p>{formatted_summary}</p>")
            html_parts.append("</div>")
    else:
        # Fallback for old format (if needed)
        for ticker, analysis in agent_results.items():
            if 'error' in analysis:
                html_parts.append(f"<h2>❌ {ticker} - Error</h2>")
                html_parts.append(f"<p style='color: red;'>{analysis['error']}</p>")
                continue
            
            summary = analysis.get('summary', 'No analysis available')
            
            html_parts.append(f"<h2>📊 {ticker} AI Analysis</h2>")
            html_parts.append(f"<div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;'>")
            html_parts.append(f"<p><strong>AI Agent Summary:</strong></p>")
            html_parts.append(f"<p>{summary}</p>")
            html_parts.append("</div>")
    
    html_parts.extend([
        "<p><em>This analysis was generated by an AI agent using LangChain ReAct framework with Market Scanner and custom financial analysis tools.</em></p>",
        "</body></html>"
    ])
    
    return "\n".join(html_parts)

def run_ai_agent_analysis(tickers: List[str]) -> Dict:
    """
    Run AI agent analysis using LangChain ReAct agent with market scanner.
    Enhanced with comprehensive error handling.
    """
    try:
        # Initialize AI components
        initialize_ai_components()
        
        # Define tools
        tools = [scan_market_for_trending_tickers, get_fallback_tickers, fetch_stock_news, analyze_headline_sentiment, analyze_multiple_headlines, get_stock_performance]
        
        # Get the ReAct prompt template
        prompt = hub.pull("hwchase17/react")
        
        # Create the agent and executor with error handling
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,  # Handle LLM output parsing errors
            max_iterations=30,  # Allow enough iterations for 4 tickers with batch sentiment analysis
            # early_stopping_method="generate"  # Not supported in this version  # Stop early if agent gets stuck
        )
        
        # Create the new, high-level prompt for the agent
        master_prompt = """
        Your mission is to create a daily market trends summary email.

        First, you MUST use the 'scan_market_for_trending_tickers' tool to discover which stocks are the most significant market movers today. This tool will return 4 validated tickers that work with the Alpha Vantage news API.

        For each ticker returned by the market scanner, perform a full analysis by:
        1. Fetching its news using 'fetch_stock_news'
        2. Analyzing the sentiment of that news using 'analyze_multiple_headlines' (preferred) or 'analyze_headline_sentiment' for individual headlines
        3. Getting its recent stock performance using 'get_stock_performance'

        IMPORTANT: Use 'analyze_multiple_headlines' when possible to reduce API calls. This tool can analyze multiple headlines at once.

        Finally, compile all the individual analyses into a single, comprehensive report formatted as a string, ready to be sent as an email. Structure the report with clear headings for each ticker.
        """
        
        # Invoke the agent with comprehensive error handling
        logger.info("Running AI agent with market scanner for dynamic ticker discovery")
        
        try:
            final_report_dict = agent_executor.invoke({"input": master_prompt})
            final_report_string = final_report_dict['output']
            
            logger.info(f"Agent execution completed successfully. Output length: {len(final_report_string)}")
            
        except StopIteration as e:
            logger.error(f"Agent execution stopped with StopIteration: {e}")
            final_report_string = "Market analysis could not be completed due to technical issues with the AI agent."
            
        except Exception as e:
            logger.error(f"Agent execution failed with error: {e}")
            final_report_string = f"Market analysis could not be completed due to technical error: {str(e)}"
        
        # Return the report as a single entry for the email formatter
        return {
            'market_analysis': {
                'summary': final_report_string,
                'status': 'success'
            }
        }
        
    except Exception as e:
        logger.error(f"Error running AI agent with market scanner: {e}")
        return {
            'market_analysis': {
                'error': f"AI agent analysis failed: {str(e)}",
                'status': 'error'
            }
        }

def run_daily_analysis(event=None, context=None):
    """
    Main entry point for Google Cloud Functions.
    This function is triggered by Cloud Scheduler or HTTP.
    Enhanced with comprehensive logging.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting AI-powered financial news analysis")
        logger.info(f"Event: {event}")
        logger.info(f"Context: {context}")
        logger.info("=" * 60)
        
        # Run AI agent analysis with market scanner
        logger.info("Calling run_ai_agent_analysis with empty ticker list")
        results = run_ai_agent_analysis([])  # Empty list since agent will discover tickers dynamically
        
        logger.info(f"Agent analysis results: {results}")
        
        # Format results as HTML
        logger.info("Formatting results as HTML")
        summary_html = format_agent_analysis_summary(results)
        
        logger.info(f"HTML summary length: {len(summary_html)} characters")
        
        # Send email
        logger.info("Sending summary email")
        email_sent = send_summary_email(summary_html)
        
        if email_sent:
            logger.info("✅ AI agent analysis completed successfully")
            return "AI-powered analysis completed and email sent successfully."
        else:
            logger.error("❌ Failed to send summary email")
            return "AI analysis completed but failed to send email."
            
    except Exception as e:
        logger.error(f"❌ Error in daily analysis: {e}")
        logger.exception("Full traceback:")
        return f"Error: {str(e)}"

# For local testing
if __name__ == "__main__":
    # Test the function locally
    result = run_daily_analysis(None, None)
    print(result)