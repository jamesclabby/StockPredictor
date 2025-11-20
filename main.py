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

# Markdown to HTML conversion
import markdown

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
    Initialize AI components (LLM) when needed.
    """
    global llm
    
    if llm is None:
        logger.info("Initializing AI components...")
        try:
            # Get OpenAI API key
            openai_api_key = get_secret('OPENAI_API_KEY')
            
            # Initialize LLM
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=openai_api_key)
            
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
def analyze_headline_sentiment(headline: str) -> str:
    """
    Analyzes sentiment of a financial news headline and optional summary using GPT-4 with financial context.
    Returns 'Positive', 'Negative', or 'Neutral'.
    
    Args:
        headline: Headline text, optionally with summary on new line starting with "Summary:"
    """
    try:
        # Ensure LLM is initialized
        initialize_ai_components()
        
        # Check if summary is included
        if "Summary:" in headline:
            parts = headline.split("Summary:", 1)
            headline_text = parts[0].strip()
            summary_text = parts[1].strip()
            content = f"Headline: {headline_text}\n\nSummary: {summary_text}"
        else:
            content = f"Headline: {headline}"
        
        prompt = f"""Analyze the sentiment of this financial news from the perspective of a stock investor.

{content}

Consider:
- Positive: News that would likely increase stock price (gains, beats expectations, partnerships, approvals, strong earnings)
- Negative: News that would likely decrease stock price (losses, misses expectations, lawsuits, failures, weak earnings)
- Neutral: Mixed news, routine updates, unclear impact, or balanced information

Respond with ONLY one word: Positive, Negative, or Neutral"""

        response = llm.invoke(prompt)
        sentiment = response.content.strip()
        
        return f"Sentiment: {sentiment}"
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return f"Error analyzing sentiment: {str(e)}"

@tool
def analyze_multiple_headlines(headlines_text: str) -> str:
    """
    Analyzes multiple headlines' sentiment in one batch using GPT-4.
    Takes a newline-separated string of headlines (with optional summaries) and returns sentiment for each.
    Handles format from fetch_stock_news_with_fallback which includes "Ticker: SYMBOL" and "Headlines:" headers.
    """
    try:
        # Ensure LLM is initialized
        initialize_ai_components()
        
        # Parse headlines (may include summaries and ticker headers)
        lines = [h.strip() for h in headlines_text.split('\n') if h.strip()]
        if not lines:
            return "No headlines to analyze"
        
        # Skip "Ticker:" and "Headlines:" header lines
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Ticker:') or stripped.startswith('Headlines:'):
                continue  # Skip header lines
            filtered_lines.append(line)  # Keep original line with spacing
        
        if not filtered_lines:
            return "No headlines to analyze"
        
        # Group lines into articles (headline + optional summary)
        articles = []
        current_article = []
        
        for line in filtered_lines:
            stripped = line.strip()
            if stripped.startswith('- '):
                # New headline
                if current_article:
                    articles.append('\n'.join(current_article))
                current_article = [stripped[2:]]  # Remove '- ' prefix
            elif 'Summary:' in stripped:
                # Summary line (may have leading spaces, handle them)
                # Extract summary text after "Summary:"
                if 'Summary:' in stripped:
                    summary_text = stripped.split('Summary:', 1)[1].strip()
                    if current_article:
                        current_article.append(f"Summary: {summary_text}")
            else:
                # Continuation of current article (could be part of summary)
                if current_article:
                    current_article.append(stripped)
        
        if current_article:
            articles.append('\n'.join(current_article))
        
        if not articles:
            return "No valid headlines to analyze"
        
        # Analyze each article
        sentiment_analysis = []
        for i, article in enumerate(articles):
            try:
                # Use the single headline analysis logic
                if "Summary:" in article:
                    parts = article.split("Summary:", 1)
                    headline_text = parts[0].strip()
                    summary_text = parts[1].strip()
                    content = f"Headline: {headline_text}\n\nSummary: {summary_text}"
                else:
                    content = f"Headline: {article}"
                
                prompt = f"""Analyze the sentiment of this financial news from the perspective of a stock investor.

{content}

Consider:
- Positive: News that would likely increase stock price (gains, beats expectations, partnerships, approvals, strong earnings)
- Negative: News that would likely decrease stock price (losses, misses expectations, lawsuits, failures, weak earnings)
- Neutral: Mixed news, routine updates, unclear impact, or balanced information

Respond with ONLY one word: Positive, Negative, or Neutral"""

                response = llm.invoke(prompt)
                sentiment = response.content.strip()
                sentiment_analysis.append(f"Headline {i+1}: {sentiment}")
            except Exception as e:
                logger.error(f"Error analyzing article {i+1}: {e}")
                sentiment_analysis.append(f"Headline {i+1}: Error - {str(e)}")
        
        return "Batch sentiment analysis:\n" + "\n".join(sentiment_analysis)
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        return f"Error in batch sentiment analysis: {str(e)}"

@tool
def fetch_stock_news_with_fallback(tickers_list: str) -> str:
    """
    Fetches news for multiple tickers, trying each in order until 3 valid tickers with news are found.
    Returns a formatted string mapping each successful ticker to its headlines and summaries.
    
    This tool automatically handles errors and empty responses by trying the next ticker.
    Stops when 3 tickers with valid, non-empty news are found.
    
    Args:
        tickers_list: Comma-separated list of tickers (e.g., "AAPL, GOOGL, TSLA, MSFT")
    
    Returns:
        Formatted string with ticker-to-headlines mapping for up to 3 successful tickers.
        Format: "Ticker: SYMBOL\nHeadlines:\n- headline1\n  Summary: summary1\n..."
    """
    try:
        # Parse comma-separated tickers
        tickers = [t.strip().upper() for t in tickers_list.split(',') if t.strip()]
        
        if not tickers:
            return "Error: No tickers provided in the list"
        
        api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        successful_tickers = []
        results = []
        
        logger.info(f"Starting fallback news fetch for {len(tickers)} tickers, targeting 3 successful")
        
        for ticker in tickers:
            # Stop if we have 3 successful tickers
            if len(successful_tickers) >= 3:
                logger.info(f"Found 3 valid tickers, stopping early. Tickers: {', '.join(successful_tickers)}")
                break
            
            try:
                # Fetch news for this ticker
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Check for API errors
                if 'Information' in data:
                    error_msg = data['Information']
                    if 'rate limit' in error_msg.lower():
                        logger.info(f"Rate limit hit for {ticker}, trying next ticker")
                        continue
                    else:
                        logger.info(f"API error for {ticker}: {error_msg}, trying next ticker")
                        continue
                
                if 'Note' in data:
                    note_msg = data['Note']
                    if 'rate limit' in note_msg.lower():
                        logger.info(f"Rate limit note for {ticker}, trying next ticker")
                        continue
                    else:
                        logger.info(f"API note for {ticker}: {note_msg}, trying next ticker")
                        continue
                
                if 'Error Message' in data:
                    logger.info(f"Error message for {ticker}: {data['Error Message']}, trying next ticker")
                    continue
                
                # Extract articles with both title and summary, deduplicating by title
                articles = data.get('feed', [])
                article_data = []
                seen_titles = set()  # Track titles to avoid duplicates
                
                for article in articles:
                    title = article.get('title', '').strip()
                    summary = article.get('summary', '').strip()
                    
                    if title:  # At minimum, we need a title
                        # Normalize title for comparison (lowercase, remove extra spaces)
                        title_normalized = ' '.join(title.lower().split())
                        
                        # Skip if we've already seen this title
                        if title_normalized not in seen_titles:
                            seen_titles.add(title_normalized)
                            article_data.append({
                                'title': title,
                                'summary': summary  # May be empty, that's OK
                            })
                
                # Check if we have valid articles
                if not article_data:
                    logger.info(f"No articles found for {ticker}, trying next ticker")
                    continue
                
                # Success! Add this ticker to results
                successful_tickers.append(ticker)
                
                # Format articles with both headline and summary (limit to 3 most recent unique articles)
                formatted_articles = []
                for article in article_data[:3]:  # Limit to 3 most recent unique articles
                    article_text = f"- {article['title']}"
                    if article['summary']:
                        article_text += f"\n  Summary: {article['summary']}"
                    formatted_articles.append(article_text)
                
                headlines_text = "\n".join(formatted_articles)
                results.append(f"Ticker: {ticker}\nHeadlines:\n{headlines_text}")
                logger.info(f"Successfully fetched {len(article_data)} articles for {ticker}")
                
            except requests.exceptions.RequestException as e:
                logger.info(f"Request error for {ticker}: {e}, trying next ticker")
                continue
            except Exception as e:
                logger.info(f"Unexpected error for {ticker}: {e}, trying next ticker")
                continue
        
        # Format final results
        if not results:
            return "Error: Could not fetch news for any tickers. All tickers failed or returned no headlines."
        
        final_result = "\n\n".join(results)
        logger.info(f"Successfully fetched news for {len(successful_tickers)} tickers: {', '.join(successful_tickers)}")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in fetch_stock_news_with_fallback: {e}")
        return f"Error in fetch_stock_news_with_fallback: {str(e)}"

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
        # Strip quotes and whitespace that the agent might include
        ticker = ticker.strip().strip("'\"")
        
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
    Returns top 6-8 tickers from NYSE/NASDAQ without validation (validation happens during news fetch).
    
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
        
        # Return top 6-8 tickers without validation (saves API calls)
        # Validation will happen naturally when fetch_stock_news_with_fallback is called
        tickers = [stock['symbol'] for stock in filtered_stocks[:8]]
        
        result = ', '.join(tickers)
        logger.info(f"Found {len(tickers)} trending tickers from FMP (no validation): {result}")
        return result
        
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
            # Convert markdown to HTML
            formatted_summary = markdown.markdown(summary, extensions=['nl2br', 'fenced_code'])
            html_parts.append(formatted_summary)
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
            # Convert markdown to HTML
            formatted_summary = markdown.markdown(summary, extensions=['nl2br', 'fenced_code'])
            html_parts.append(formatted_summary)
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
        tools = [scan_market_for_trending_tickers, get_fallback_tickers, fetch_stock_news_with_fallback, analyze_headline_sentiment, analyze_multiple_headlines, get_stock_performance]
        
        # Get the ReAct prompt template
        prompt = hub.pull("hwchase17/react")
        
        # Create the agent and executor with error handling
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,  # Handle LLM output parsing errors
            max_iterations=30,  # Allow enough iterations for 3 tickers with batch sentiment analysis
            # early_stopping_method="generate"  # Not supported in this version  # Stop early if agent gets stuck
        )
        
        # Create the new, high-level prompt for the agent
        master_prompt = """
        Your mission is to create a daily market trends summary email.

        First, you MUST use the 'scan_market_for_trending_tickers' tool to discover which stocks are the most significant market movers today. This tool will return a list of trending tickers (typically 6-8 tickers).

        Next, you MUST use the 'fetch_stock_news_with_fallback' tool with the full list of tickers from the market scanner. This tool will automatically try tickers in order until it finds 3 with valid news data. It handles errors and empty responses automatically by trying the next ticker.

        For each of the tickers returned by 'fetch_stock_news_with_fallback' (which will be up to 3 tickers), perform analysis by:
        1. Analyzing the sentiment of the news headlines using 'analyze_multiple_headlines' (preferred) or 'analyze_headline_sentiment' for individual headlines. Do not use both for any one unique headline.
        2. Getting its recent stock performance using 'get_stock_performance'

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