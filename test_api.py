#!/usr/bin/env python3
"""
Test Alpha Vantage API directly
"""

import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_alpha_vantage_api():
    """Test Alpha Vantage API directly"""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        print("❌ No API key found in .env file")
        return
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    # Test with AAPL
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': 'AAPL',
        'apikey': api_key
    }
    
    print(f"🔍 Making API call...")
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"✅ Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response keys: {list(data.keys())}")
            
            if 'feed' in data:
                articles = data['feed']
                print(f"✅ Found {len(articles)} articles")
                
                if articles:
                    print(f"✅ First article: {articles[0].get('title', 'No title')}")
                    print(f"✅ First article time: {articles[0].get('time_published', 'No time')}")
                else:
                    print("❌ No articles in feed")
            else:
                print("❌ No 'feed' key in response")
                print(f"Response content: {json.dumps(data, indent=2)[:500]}...")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response content: {response.text[:500]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_alpha_vantage_api()
