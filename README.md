# Financial News Analysis Cloud Function

A Google Cloud Function that automatically analyzes financial news sentiment and sends daily email summaries. The system fetches real-time financial news from Alpha Vantage API, performs AI-powered sentiment analysis, and delivers beautiful HTML email reports.

## 🚀 **Current Status: Production Ready**

This project has been refactored from a Streamlit web application into a serverless Cloud Function that runs daily and sends email summaries via Resend.

## ✨ **Features**

- **Automated Daily Analysis**: Runs on a schedule via Google Cloud Scheduler
- **AI-Powered Sentiment Analysis**: Uses Hugging Face's DistilBERT model
- **Real-time News Fetching**: Alpha Vantage API integration
- **Beautiful Email Reports**: HTML-formatted summaries with sentiment visualization
- **Robust Error Handling**: Circuit breaker pattern, retry logic, rate limit handling
- **Secure Secret Management**: Google Cloud Secret Manager integration
- **Cost-Effective**: Uses Resend's generous free tier (3,000 emails/month)

## 🏗️ **Architecture**

- **Google Cloud Functions**: Serverless execution environment
- **Resend**: Email delivery service (better free plan than SendGrid)
- **Alpha Vantage API**: Financial news data source
- **Hugging Face Transformers**: AI sentiment analysis
- **Google Secret Manager**: Secure API key storage

## 📁 **Project Structure**

```
StockAnalysis/
├── main.py                          # Cloud Function entry point
├── financial_news_analyzer.py       # Core analysis logic
├── requirements_cloud.txt           # Dependencies for Cloud Functions
├── test_resend_function.py          # Local testing script
├── env_resend_template.txt          # Environment variables template
├── DEPLOYMENT_GUIDE_RESEND.md       # Complete deployment guide
└── README.md                        # This file
```

## 🚀 **Quick Start**

### **1. Local Testing**
```bash
# Install dependencies
pip install -r requirements_cloud.txt

# Set up environment variables
cp env_resend_template.txt .env
# Edit .env with your API keys

# Test locally
python test_resend_function.py
```

### **2. Deploy to Google Cloud Functions**
```bash
# Deploy the function
gcloud functions deploy financial-news-analysis \
    --runtime python39 \
    --trigger-http \
    --allow-unauthenticated \
    --source . \
    --entry-point run_daily_analysis \
    --memory 1GB \
    --timeout 540s

# Set up daily scheduling
gcloud scheduler jobs create pubsub daily-financial-analysis \
    --schedule="0 9 * * *" \
    --topic=daily-analysis \
    --message-body='{"trigger": "daily"}' \
    --time-zone="America/New_York"
```

## 📊 **Analysis Output**

The system analyzes these tickers by default:
- **AAPL** (Apple)
- **GOOGL** (Google/Alphabet)  
- **TSLA** (Tesla)

Each daily email includes:
- Overall sentiment summary per ticker
- Individual article headlines with sentiment scores
- Color-coded sentiment indicators
- Confidence percentages
- Professional HTML formatting

## 🔧 **Configuration**

### **Email Settings** (in main.py):
```python
SENDER_EMAIL = "onboarding@resend.dev"  # Resend test domain
RECIPIENT_EMAIL = "your-email@example.com"  # Your email
EMAIL_SUBJECT = "Daily Financial News Analysis"
```

### **Tickers** (in main.py):
```python
TICKERS = ['AAPL', 'GOOGL', 'TSLA']  # Add your preferred tickers
```

## 🔐 **API Keys Required**

1. **Alpha Vantage API Key**: Get from [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. **Resend API Key**: Get from [resend.com](https://resend.com/) (free tier: 3,000 emails/month)

## 📚 **Development History**

This project started as a **Streamlit web application** for testing and developing the financial news analysis functionality. The Streamlit app was used to:
- Test sentiment analysis algorithms
- Validate API integrations
- Develop the user interface
- Debug and refine the analysis logic

Once the core functionality was proven, the project was refactored into a production-ready Cloud Function for automated daily email delivery.

## 🛠️ **Technical Details**

- **Language**: Python 3.9+
- **ML Framework**: Hugging Face Transformers
- **Email Service**: Resend (3,000 free emails/month)
- **News API**: Alpha Vantage (25 free requests/day)
- **Deployment**: Google Cloud Functions
- **Scheduling**: Google Cloud Scheduler
- **Secrets**: Google Cloud Secret Manager

## 📖 **Documentation**

- **Complete Deployment Guide**: See `DEPLOYMENT_GUIDE_RESEND.md`
- **Local Testing**: Run `python test_resend_function.py`
- **Environment Setup**: Use `env_resend_template.txt`

## 🎯 **Next Steps**

1. **Deploy**: Follow the deployment guide
2. **Customize**: Update tickers and email settings
3. **Monitor**: Check Cloud Functions logs
4. **Scale**: Upgrade API plans as needed

## 📄 **License**

This project is for educational and personal use. Please respect API terms of service for Alpha Vantage and Resend.