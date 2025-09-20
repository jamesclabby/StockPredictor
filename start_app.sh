#!/bin/bash
# Simple startup script for the Stock Sentiment Analysis app

echo "📈 Stock Sentiment Analysis - Quick Start"
echo "========================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "📝 Please create a .env file with your Alpha Vantage API key:"
    echo "   echo 'ALPHA_VANTAGE_API_KEY=your_api_key_here' > .env"
    exit 1
fi

# Check if API key is set
if grep -q "your_api_key_here" .env; then
    echo "⚠️  Please update your .env file with your actual API key"
    exit 1
fi

echo "✅ .env file found with API key"
echo "🚀 Starting Streamlit app..."
echo "🌐 The app will open at: http://localhost:8501"
echo ""

# Start Streamlit
streamlit run streamlit_app.py --server.headless true --browser.gatherUsageStats false
