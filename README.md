# AI-Powered Financial News Analysis Cloud Function

An intelligent Google Cloud Function that automatically discovers trending stocks, analyzes financial news sentiment using AI agents, and sends comprehensive daily email summaries. The system uses LangChain ReAct agents with custom tools to provide sophisticated market analysis.

## 🚀 **Current Status: Production Ready**

This project has evolved from a Streamlit web application into an advanced AI-powered serverless Cloud Function that runs daily and sends intelligent market analysis via Resend.

## ✨ **Features**

- **🤖 AI Agent Analysis**: LangChain ReAct agents with custom tools for sophisticated reasoning
- **📈 Dynamic Market Discovery**: Uses Financial Modeling Prep API to find trending stocks
- **🧠 Intelligent Sentiment Analysis**: Hugging Face DistilBERT + OpenAI GPT-4o hybrid approach
- **📧 Beautiful Email Reports**: HTML-formatted summaries with comprehensive market insights
- **🛡️ Robust Error Handling**: Circuit breaker pattern, retry logic, rate limit handling
- **🔐 Secure Secret Management**: Google Cloud Secret Manager integration
- **💰 Cost-Effective**: Uses Resend's generous free tier (3,000 emails/month)
- **⚡ Optimized API Usage**: Smart fallback mechanism reduces Alpha Vantage API calls by ~50%
- **🔄 Automatic Retry Logic**: Intelligently tries tickers until finding 3 with valid news data

## 🏗️ **Architecture**

- **Google Cloud Functions**: Serverless execution environment (2GB memory, 9min timeout)
- **LangChain ReAct Agents**: AI reasoning with custom tools for market analysis
- **Financial Modeling Prep**: Real-time trending stock discovery
- **Alpha Vantage API**: Financial news data and stock performance
- **OpenAI GPT-4o**: Advanced reasoning and analysis synthesis
- **Hugging Face Transformers**: Local sentiment analysis (DistilBERT)
- **Resend**: Email delivery service (3,000 free emails/month)
- **Google Secret Manager**: Secure API key storage

## 📁 **Project Structure**

```
StockAnalysis/
├── main.py                          # AI Agent Cloud Function entry point
├── financial_news_analyzer.py       # Core analysis logic (legacy)
├── requirements.txt                  # Dependencies for Cloud Functions
├── test_ai_agent.py                 # Local AI agent testing script
├── .env                             # Environment variables (local only)
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🚀 **Quick Start**

### **1. Local Testing**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with your API keys:
# ALPHA_VANTAGE_API_KEY=your_key_here
# FMP_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# RESEND_API_KEY=your_key_here

# Test the AI agent locally
python test_ai_agent.py
```

### **2. Deploy to Google Cloud Functions**
```bash
# Store secrets in Google Cloud Secret Manager
gcloud secrets create ALPHA_VANTAGE_API_KEY --data-file=- <<< "your_key_here"
gcloud secrets create FMP_API_KEY --data-file=- <<< "your_key_here"
gcloud secrets create OPENAI_API_KEY --data-file=- <<< "your_key_here"
gcloud secrets create RESEND_API_KEY --data-file=- <<< "your_key_here"

# Deploy the AI agent function
gcloud functions deploy run-daily-analysis \
    --gen2 \
    --runtime=python311 \
    --source=. \
    --entry-point=run_daily_analysis \
    --trigger-http \
    --allow-unauthenticated \
    --memory=2GB \
    --timeout=540s \
    --region=us-central1 \
    --set-secrets="ALPHA_VANTAGE_API_KEY=ALPHA_VANTAGE_API_KEY:latest,FMP_API_KEY=FMP_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,RESEND_API_KEY=RESEND_API_KEY:latest"

# Set up daily scheduling (8 AM Eastern)
gcloud scheduler jobs create http daily-financial-analysis \
    --schedule="0 13 * * *" \
    --uri="https://us-central1-stockanalysis-472822.cloudfunctions.net/run-daily-analysis" \
    --http-method=POST \
    --time-zone="America/New_York"
```

## 📊 **AI Agent Analysis Process**

The system uses a sophisticated AI agent workflow with optimized API usage:

1. **🔍 Market Discovery**: Scans Financial Modeling Prep's "most-actives" API for trending stocks (returns 6-8 tickers from NYSE/NASDAQ)
2. **📰 Smart News Fetching with Fallback**: 
   - Uses `fetch_stock_news_with_fallback` tool that automatically tries tickers in order
   - Handles errors gracefully (rate limits, invalid tickers, empty responses)
   - Stops when 3 valid tickers with news are found (saves API calls)
   - **No pre-validation needed** - validation happens during actual news fetch
3. **🧠 Sentiment Analysis**: Uses batch processing for efficient sentiment analysis
4. **📈 Performance Check**: Gets current stock prices and daily changes for the 3 successful tickers
5. **🤖 AI Synthesis**: GPT-4o agent synthesizes all data into comprehensive insights

**API Call Optimization**: 
- **Before**: ~12-16 calls/day (4-8 validation + 4 news + 4 performance)
- **After**: ~6-9 calls/day (0 validation + 3-6 news + 3 performance)
- **Savings**: ~50% reduction in Alpha Vantage API calls

**Fallback System**: If trending stocks fail, the system automatically tries the next ticker in the list until 3 valid ones are found. If all trending stocks fail, falls back to major stocks (AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA)

Each daily email includes:
- **Market Overview**: AI-generated summary of trending stocks
- **Individual Analysis**: Per-ticker sentiment and performance analysis
- **News Headlines**: Recent headlines with sentiment scores
- **Stock Performance**: Current prices and daily changes
- **Professional HTML**: Beautiful formatting with color-coded indicators

## 🔧 **Configuration**

### **Email Settings** (in main.py):
```python
SENDER_EMAIL = "onboarding@resend.dev"  # Resend test domain
RECIPIENT_EMAIL = "your-email@example.com"  # Your email
EMAIL_SUBJECT = "AI Agent Market Report"
```

### **AI Agent Settings** (in main.py):
```python
# Agent configuration
max_iterations = 30  # Allow sufficient steps for analysis
master_prompt = """Your mission is to create a daily market trends summary email..."""
```

## 🔐 **API Keys Required**

1. **Alpha Vantage API Key**: Get from [alphavantage.co](https://www.alphavantage.co/support/#api-key) (25 free requests/day)
2. **Financial Modeling Prep API Key**: Get from [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs) (250 free requests/day)
3. **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys) (pay-per-use)
4. **Resend API Key**: Get from [resend.com](https://resend.com/) (3,000 free emails/month)

## 📚 **Development History**

This project evolved through several phases:

1. **Streamlit Web App**: Initial development and testing platform
   - Tested sentiment analysis algorithms
   - Validated API integrations
   - Developed user interface
   - Debugged and refined analysis logic

2. **Cloud Function Migration**: Refactored for production
   - Converted to serverless architecture
   - Added email delivery via Resend
   - Implemented Google Cloud Secret Manager

3. **AI Agent Enhancement**: Advanced intelligence layer
   - Integrated LangChain ReAct agents
   - Added Financial Modeling Prep for market discovery
   - Implemented batch sentiment analysis
   - Enhanced with OpenAI GPT-4o reasoning

## 🛠️ **Technical Details**

- **Language**: Python 3.11
- **AI Framework**: LangChain + OpenAI GPT-4o
- **ML Framework**: Hugging Face Transformers (DistilBERT)
- **Market Data**: Financial Modeling Prep + Alpha Vantage
- **Email Service**: Resend (3,000 free emails/month)
- **Deployment**: Google Cloud Functions (2GB memory, 9min timeout)
- **Scheduling**: Google Cloud Scheduler
- **Secrets**: Google Cloud Secret Manager

## 🎯 **Key Capabilities**

- **Dynamic Market Discovery**: Finds trending stocks automatically from Financial Modeling Prep
- **Smart Fallback Mechanism**: Automatically tries tickers until finding 3 with valid news (no wasted API calls)
- **Intelligent Analysis**: AI agents provide sophisticated reasoning with GPT-4o
- **Optimized API Usage**: Reduces Alpha Vantage API calls by ~50% through intelligent retry logic
- **Batch Processing**: Efficient sentiment analysis with batch processing
- **Robust Error Handling**: Graceful handling of rate limits, invalid tickers, and API errors
- **Production Ready**: Scalable serverless architecture with comprehensive logging

## 📄 **License**

This project is for educational and personal use. Please respect API terms of service for all integrated services.