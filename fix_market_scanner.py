# Fix 1: Market Scanner Tool - Return string instead of list, limit to 5 tickers each

@tool
def scan_market_for_trending_tickers() -> str:
    """
    Scans the market using the Alpha Vantage API to find the day's top gainers and top losers.
    Returns a comma-separated string of the top 5 gainers and top 5 losers to be analyzed.
    """
    try:
        api_key = get_secret('ALPHA_VANTAGE_API_KEY')
        url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Information' in data:
            if 'rate limit' in data['Information'].lower():
                logger.warning("API rate limit exceeded for market scanner")
                return "No trending tickers available due to API rate limit"
            else:
                logger.error(f"API error in market scanner: {data['Information']}")
                return "No trending tickers available due to API error"
        
        if 'Note' in data:
            if 'rate limit' in data['Note'].lower():
                logger.warning("API rate limit exceeded for market scanner")
                return "No trending tickers available due to API rate limit"
            else:
                logger.error(f"API error in market scanner: {data['Note']}")
                return "No trending tickers available due to API error"
        
        # Extract top gainers and losers (limit to 5 each to avoid rate limits)
        top_gainers = [g['ticker'] for g in data.get('top_gainers', [])[:5]]
        top_losers = [l['ticker'] for l in data.get('top_losers', [])[:5]]
        
        trending_tickers = top_gainers + top_losers
        
        logger.info(f"Dynamically found trending tickers: {trending_tickers}")
        
        # Return as comma-separated string for LangChain compatibility
        if trending_tickers:
            return ", ".join(trending_tickers)
        else:
            return "No trending tickers found today"
        
    except Exception as e:
        logger.exception("An error occurred in scan_market_for_trending_tickers.")
        return "No trending tickers available due to technical error"
