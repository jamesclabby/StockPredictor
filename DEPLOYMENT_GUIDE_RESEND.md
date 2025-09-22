# Google Cloud Functions Deployment Guide (Resend Version)

## Overview
This guide explains how to deploy the refactored financial news analysis system to Google Cloud Functions using Resend for email delivery.

## Why Resend?
- **Better Free Plan**: 3,000 emails/month vs SendGrid's 100 emails/day
- **Simpler API**: Cleaner, more modern email API
- **Better Developer Experience**: Easier setup and configuration
- **No Credit Card Required**: Free plan doesn't require payment info

## Files Created
- `main_resend.py` - Main Cloud Function code with Resend integration
- `requirements_resend.txt` - Python dependencies for Cloud Functions
- `financial_news_analyzer.py` - Core analysis logic (existing)
- `test_resend_function.py` - Local testing script

## Prerequisites

### 1. Google Cloud Setup
```bash
# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Resend Setup (Much Easier!)
1. Create account at [Resend](https://resend.com/)
2. Verify your domain (or use their test domain for testing)
3. Generate API key from dashboard
4. Store API key in Google Secret Manager

### 3. Alpha Vantage API
1. Get API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Store API key in Google Secret Manager

## Secret Management

### Store Secrets in Google Secret Manager
```bash
# Store Alpha Vantage API key
echo "YOUR_ALPHA_VANTAGE_API_KEY" | gcloud secrets create ALPHA_VANTAGE_API_KEY --data-file=-

# Store Resend API key
echo "YOUR_RESEND_API_KEY" | gcloud secrets create RESEND_API_KEY --data-file=-

# Grant Cloud Functions access to secrets
gcloud secrets add-iam-policy-binding ALPHA_VANTAGE_API_KEY \
    --member="serviceAccount:YOUR_PROJECT_ID@appspot.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding RESEND_API_KEY \
    --member="serviceAccount:YOUR_PROJECT_ID@appspot.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Configuration

### Update Email Settings in main_resend.py
```python
# Update these values in main_resend.py
SENDER_EMAIL = "your-verified-email@yourdomain.com"  # Must be verified in Resend
RECIPIENT_EMAIL = "recipient@example.com"  # Where to send the analysis
EMAIL_SUBJECT = "Daily Financial News Analysis"
```

### Update Tickers (Optional)
```python
# Modify the TICKERS list in main_resend.py
TICKERS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']  # Add your preferred tickers
```

## Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements_resend.txt
```

### 2. Set Up Environment Variables
```bash
# Copy the template
cp env_resend_template.txt .env

# Edit .env with your actual values
# ALPHA_VANTAGE_API_KEY=your_key_here
# RESEND_API_KEY=your_resend_key_here
```

### 3. Test Locally
```bash
python test_resend_function.py
```

## Deployment

### 1. Deploy the Cloud Function
```bash
# Deploy with HTTP trigger (for testing)
gcloud functions deploy financial-news-analysis \
    --runtime python39 \
    --trigger-http \
    --allow-unauthenticated \
    --source . \
    --entry-point run_daily_analysis \
    --memory 1GB \
    --timeout 540s

# Deploy with Cloud Scheduler trigger (for daily runs)
gcloud functions deploy financial-news-analysis \
    --runtime python39 \
    --trigger-topic daily-analysis \
    --source . \
    --entry-point run_daily_analysis \
    --memory 1GB \
    --timeout 540s
```

### 2. Set Up Cloud Scheduler (for daily runs)
```bash
# Create a Pub/Sub topic
gcloud pubsub topics create daily-analysis

# Create a Cloud Scheduler job
gcloud scheduler jobs create pubsub daily-financial-analysis \
    --schedule="0 9 * * *" \
    --topic=daily-analysis \
    --message-body='{"trigger": "daily"}' \
    --time-zone="America/New_York"
```

## Testing

### 1. Test HTTP Trigger
```bash
# Test HTTP trigger
curl -X POST https://REGION-PROJECT_ID.cloudfunctions.net/financial-news-analysis

# Check logs
gcloud functions logs read financial-news-analysis --limit 50
```

### 2. Test Email Delivery
- Check your email inbox for the analysis
- Verify Resend dashboard for delivery statistics
- Check Cloud Functions logs for any errors

## Monitoring

### View Logs
```bash
# Real-time logs
gcloud functions logs tail financial-news-analysis

# Historical logs
gcloud functions logs read financial-news-analysis --limit 100
```

### Monitor Performance
- Check Cloud Functions metrics in Google Cloud Console
- Monitor API usage in Alpha Vantage dashboard
- Check Resend delivery statistics and analytics

## Troubleshooting

### Common Issues

1. **Rate Limit Exceeded**
   - Alpha Vantage free tier: 25 requests/day
   - Consider upgrading to premium plan
   - Implement request caching

2. **Email Not Sending**
   - Verify sender email in Resend dashboard
   - Check Resend API key permissions
   - Review Resend activity feed

3. **Secret Access Denied**
   - Verify IAM permissions for Cloud Functions service account
   - Check secret names match exactly

4. **Function Timeout**
   - Increase timeout in deployment command
   - Optimize analysis logic
   - Consider reducing number of tickers

### Debug Commands
```bash
# Check function status
gcloud functions describe financial-news-analysis

# View function configuration
gcloud functions describe financial-news-analysis --format="value(httpsTrigger.url)"

# Test with specific payload
gcloud functions call financial-news-analysis --data='{"test": true}'
```

## Resend vs SendGrid Comparison

| Feature | Resend | SendGrid |
|---------|--------|----------|
| Free Emails | 3,000/month | 100/day |
| Setup Complexity | Simple | Complex |
| API Design | Modern | Legacy |
| Documentation | Excellent | Good |
| Credit Card Required | No | Yes (for some features) |

## Cost Optimization

1. **Memory**: Start with 1GB, adjust based on usage
2. **Timeout**: Set appropriate timeout to avoid unnecessary costs
3. **Frequency**: Consider running less frequently if rate limits are an issue
4. **Caching**: Implement result caching to reduce API calls

## Security Best Practices

1. **Secrets**: Never hardcode API keys in code
2. **IAM**: Use least privilege principle for service accounts
3. **Network**: Consider VPC connector if needed
4. **Monitoring**: Set up alerts for failures and anomalies
5. **Email Security**: Use verified domains for sender addresses
