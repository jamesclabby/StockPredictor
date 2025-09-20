#!/usr/bin/env python3
"""
Streamlit Web Application for Stock Sentiment Analysis

A web interface for the financial news analysis agent that provides
interactive sentiment analysis for multiple stock tickers.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional

# Import our existing financial news analyzer
from financial_news_analyzer import (
    load_api_key, fetch_news_with_retry, analyze_sentiment_robust,
    is_within_24_hours, format_sentiment_output, APIError, ErrorType,
    CircuitBreaker, setup_logging
)

# Import ticker data manager
from ticker_data_manager import TickerDataManager

# Configure page
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'circuit_breaker' not in st.session_state:
    st.session_state.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
if 'sentiment_pipeline' not in st.session_state:
    st.session_state.sentiment_pipeline = None
if 'ticker_manager' not in st.session_state:
    st.session_state.ticker_manager = TickerDataManager()
if 'all_tickers' not in st.session_state:
    st.session_state.all_tickers = []
if 'popular_tickers' not in st.session_state:
    st.session_state.popular_tickers = []

def load_ticker_data():
    """Load ticker data if not already loaded."""
    if not st.session_state.all_tickers:
        with st.spinner("Loading ticker data..."):
            try:
                st.session_state.all_tickers = st.session_state.ticker_manager.get_tickers()
                st.session_state.popular_tickers = st.session_state.ticker_manager.get_popular_tickers()
                st.success(f"✅ Loaded {len(st.session_state.all_tickers)} tickers")
            except Exception as e:
                st.error(f"❌ Error loading ticker data: {e}")
                # Fallback to basic list
                st.session_state.popular_tickers = [
                    'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'NFLX',
                    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE'
                ]
                st.session_state.all_tickers = []

def initialize_sentiment_model():
    """Initialize the sentiment analysis model if not already loaded."""
    if st.session_state.sentiment_pipeline is None:
        with st.spinner("Loading sentiment analysis model..."):
            try:
                from transformers import pipeline
                st.session_state.sentiment_pipeline = pipeline(
                    'sentiment-analysis', 
                    model='distilbert-base-uncased-finetuned-sst-2-english'
                )
                st.success("✅ Sentiment analysis model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading sentiment analysis model: {e}")
                return False
    return True

def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment display."""
    if sentiment == 'POSITIVE':
        return '#28a745'
    elif sentiment == 'NEGATIVE':
        return '#dc3545'
    else:
        return '#6c757d'

def create_sentiment_distribution_chart(sentiment_data: Dict[str, int]) -> go.Figure:
    """Create a pie chart for sentiment distribution."""
    labels = list(sentiment_data.keys())
    values = list(sentiment_data.values())
    colors = [get_sentiment_color(label) for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    return fig

def create_confidence_chart(articles_data: List[Dict]) -> go.Figure:
    """Create a bar chart showing confidence scores."""
    if not articles_data:
        return go.Figure()
    
    titles = [article['title'][:50] + '...' if len(article['title']) > 50 else article['title'] 
              for article in articles_data]
    confidences = [article['sentiment']['score'] * 100 for article in articles_data]
    sentiments = [article['sentiment']['label'] for article in articles_data]
    colors = [get_sentiment_color(sentiment) for sentiment in sentiments]
    
    fig = go.Figure(data=[go.Bar(
        x=titles,
        y=confidences,
        marker_color=colors,
        text=[f"{conf:.1f}%" for conf in confidences],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Confidence Scores by Article",
        xaxis_title="Articles",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def display_ticker_results(ticker: str, results: Dict):
    """Display results for a specific ticker."""
    st.subheader(f"📊 {ticker} Analysis Results")
    
    if not results or not results.get('articles'):
        st.warning(f"No recent articles found for {ticker}")
        return
    
    articles = results['articles']
    summary = results['summary']
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(articles))
    
    with col2:
        st.metric("Positive Articles", summary.get('positive', 0))
    
    with col3:
        st.metric("Negative Articles", summary.get('negative', 0))
    
    with col4:
        avg_confidence = summary.get('avg_confidence', 0)
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Sentiment distribution chart
    sentiment_counts = summary.get('sentiment_counts', {})
    if sentiment_counts:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_pie = create_sentiment_distribution_chart(sentiment_counts)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = create_confidence_chart(articles)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Individual articles
    st.subheader("📰 Individual Articles")
    
    for i, article in enumerate(articles, 1):
        sentiment = article['sentiment']['label']
        confidence = article['sentiment']['score'] * 100
        color = get_sentiment_color(sentiment)
        
        with st.expander(f"Article {i}: {article['title'][:80]}{'...' if len(article['title']) > 80 else ''}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                st.write(f"**Published:** {article.get('time_published', 'Unknown')}")
            
            with col2:
                st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown(f"<span class='{sentiment.lower()}-sentiment'>{sentiment}</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: {color}; font-size: 1.2em; font-weight: bold;'>{confidence:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Confidence bar
                progress_color = color
                st.markdown(f"""
                <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; margin: 10px 0;">
                    <div style="background-color: {progress_color}; width: {confidence}%; height: 100%; border-radius: 10px; transition: width 0.3s ease;"></div>
                </div>
                """, unsafe_allow_html=True)

def analyze_tickers(tickers: List[str]) -> Dict:
    """Analyze sentiment for multiple tickers."""
    results = {}
    
    # Initialize API key and model
    try:
        api_key = load_api_key()
    except APIError as e:
        st.error(f"❌ Configuration Error: {e.message}")
        return results
    
    if not initialize_sentiment_model():
        return results
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Analyzing {ticker}...")
        
        try:
            # Fetch news articles
            articles = fetch_news_with_retry(ticker, api_key, st.session_state.circuit_breaker)
            
            if not articles:
                results[ticker] = {'articles': [], 'summary': {}}
                continue
            
            # Filter recent articles
            recent_articles = []
            for article in articles:
                time_published = article.get('time_published', '')
                if time_published and is_within_24_hours(time_published):
                    recent_articles.append(article)
            
            # Analyze sentiment
            analyzed_articles = []
            sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0}
            total_confidence = 0
            
            for article in recent_articles:
                title = article.get('title', '')
                if title:
                    sentiment_result = analyze_sentiment_robust(title, st.session_state.sentiment_pipeline)
                    
                    if sentiment_result['label'] in ['POSITIVE', 'NEGATIVE']:
                        analyzed_articles.append({
                            'title': title,
                            'sentiment': sentiment_result,
                            'time_published': article.get('time_published', ''),
                            'source': article.get('source', 'Unknown')
                        })
                        
                        sentiment_counts[sentiment_result['label']] += 1
                        total_confidence += sentiment_result['score']
            
            # Calculate summary
            total_articles = len(analyzed_articles)
            avg_confidence = (total_confidence / total_articles * 100) if total_articles > 0 else 0
            
            summary = {
                'total_articles': total_articles,
                'positive': sentiment_counts['POSITIVE'],
                'negative': sentiment_counts['NEGATIVE'],
                'avg_confidence': avg_confidence,
                'sentiment_counts': sentiment_counts,
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            results[ticker] = {
                'articles': analyzed_articles,
                'summary': summary
            }
            
        except APIError as e:
            st.error(f"❌ Error analyzing {ticker}: {e.message}")
            results[ticker] = {'articles': [], 'summary': {}, 'error': str(e)}
        except Exception as e:
            st.error(f"❌ Unexpected error analyzing {ticker}: {e}")
            results[ticker] = {'articles': [], 'summary': {}, 'error': str(e)}
        
        # Update progress
        progress_bar.progress((i + 1) / len(tickers))
    
    status_text.text("Analysis complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load ticker data
    load_ticker_data()
    
    # Sidebar for input
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Stock ticker selection
        st.subheader("Select Stock Tickers")
        
        # Ticker selection method
        selection_method = st.radio(
            "Selection Method",
            ["Popular Tickers", "Search All Tickers", "Custom Input"],
            help="Choose how to select tickers"
        )
        
        all_tickers = []
        
        if selection_method == "Popular Tickers":
            # Multi-select for popular tickers
            selected_tickers = st.multiselect(
                "Popular Tickers",
                st.session_state.popular_tickers,
                default=['AAPL', 'GOOGL', 'TSLA'],
                help="Select from popular stock tickers"
            )
            all_tickers = selected_tickers
            
        elif selection_method == "Search All Tickers":
            # Search functionality
            search_query = st.text_input(
                "Search Tickers",
                placeholder="Type to search (e.g., AAPL, Apple, Tech)",
                help="Search by ticker symbol or company name"
            )
            
            if search_query:
                search_results = st.session_state.ticker_manager.search_tickers(
                    search_query, st.session_state.all_tickers, limit=100
                )
                
                if search_results:
                    # Create options for multiselect
                    ticker_options = [f"{ticker['symbol']} - {ticker['name']}" for ticker in search_results]
                    selected_indices = st.multiselect(
                        "Search Results",
                        ticker_options,
                        help="Select from search results"
                    )
                    
                    # Extract ticker symbols
                    all_tickers = [option.split(' - ')[0] for option in selected_indices]
                else:
                    st.info("No tickers found matching your search.")
            else:
                st.info("Enter a search term to find tickers.")
                
        else:  # Custom Input
            # Custom ticker input
            custom_tickers = st.text_input(
                "Custom Tickers",
                placeholder="Enter custom tickers separated by commas (e.g., MSFT, AMZN)",
                help="Add any tickers not in the lists above"
            )
            
            if custom_tickers:
                all_tickers = [ticker.strip().upper() for ticker in custom_tickers.split(',') if ticker.strip()]
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        
        # Display selected tickers
        if all_tickers:
            st.write("**Selected Tickers:**")
            for ticker in all_tickers:
                st.write(f"• {ticker}")
        
        st.markdown("---")
        
        # Information section
        st.subheader("ℹ️ About")
        st.markdown("""
        This application analyzes the sentiment of recent financial news 
        for selected stock tickers using AI-powered sentiment analysis.
        
        **Features:**
        - **Comprehensive Ticker Database**: Access to all NYSE & NASDAQ stocks
        - **Smart Search**: Find tickers by symbol or company name
        - **Real-time Analysis**: Live sentiment analysis with confidence scores
        - **Interactive Visualizations**: Charts and graphs for insights
        - **24-hour Filtering**: Only recent news articles
        - **Export Results**: Download analysis as CSV
        """)
        
        # Show ticker database stats
        if st.session_state.all_tickers:
            st.subheader("📊 Database Stats")
            st.write(f"**Total Tickers:** {len(st.session_state.all_tickers):,}")
            
            # Count by exchange
            nasdaq_count = len([t for t in st.session_state.all_tickers if t.get('exchange') == 'NASDAQ'])
            nyse_count = len([t for t in st.session_state.all_tickers if t.get('exchange') == 'NYSE'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("NASDAQ", f"{nasdaq_count:,}")
            with col2:
                st.metric("NYSE", f"{nyse_count:,}")
    
    # Main content area
    if analyze_button and all_tickers:
        st.session_state.analysis_results = analyze_tickers(all_tickers)
    
    # Display results
    if st.session_state.analysis_results:
        st.header("📊 Analysis Results")
        
        # Create tabs for each ticker
        ticker_tabs = st.tabs([f"📈 {ticker}" for ticker in st.session_state.analysis_results.keys()])
        
        for i, (ticker, results) in enumerate(st.session_state.analysis_results.items()):
            with ticker_tabs[i]:
                display_ticker_results(ticker, results)
        
        # Overall summary
        st.header("📋 Overall Summary")
        
        # Create summary metrics
        total_tickers = len(st.session_state.analysis_results)
        successful_tickers = sum(1 for results in st.session_state.analysis_results.values() 
                               if results.get('articles'))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tickers Analyzed", total_tickers)
        
        with col2:
            st.metric("Successful Analyses", successful_tickers)
        
        with col3:
            success_rate = (successful_tickers / total_tickers * 100) if total_tickers > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Export results
        st.subheader("💾 Export Results")
        
        # Create downloadable data
        export_data = []
        for ticker, results in st.session_state.analysis_results.items():
            for article in results.get('articles', []):
                export_data.append({
                    'Ticker': ticker,
                    'Title': article['title'],
                    'Sentiment': article['sentiment']['label'],
                    'Confidence': f"{article['sentiment']['score']*100:.1f}%",
                    'Source': article.get('source', 'Unknown'),
                    'Published': article.get('time_published', 'Unknown')
                })
        
        if export_data:
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"stock_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    elif not st.session_state.analysis_results:
        # Welcome message
        st.markdown("""
        ## 🎯 Welcome to Stock Sentiment Analysis
        
        This application helps you analyze the sentiment of financial news for your selected stock tickers.
        
        **To get started:**
        1. Select stock tickers from the sidebar
        2. Add any custom tickers if needed
        3. Click "Analyze Sentiment" to begin
        
        **What you'll get:**
        - Sentiment analysis for recent news articles
        - Confidence scores for each prediction
        - Interactive visualizations
        - Exportable results
        """)
        
        # Sample visualization
        st.subheader("📊 Sample Analysis")
        
        # Create a sample pie chart
        sample_data = {'POSITIVE': 15, 'NEGATIVE': 8, 'NEUTRAL': 3}
        fig = create_sentiment_distribution_chart(sample_data)
        fig.update_layout(title="Sample Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
