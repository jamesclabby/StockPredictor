#!/usr/bin/env python3
"""
Ticker Data Manager

Fetches and manages comprehensive lists of NYSE and NASDAQ stock tickers.
Provides caching and search functionality for the Streamlit app.
"""

import os
import json
import csv
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickerDataManager:
    """Manages ticker data fetching, caching, and searching."""
    
    def __init__(self, cache_dir: str = "ticker_cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "tickers.json")
        self.cache_duration = timedelta(days=1)  # Cache Alpha Vantage data for 1 day
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                cache_time = datetime.fromisoformat(data.get('timestamp', ''))
                return datetime.now() - cache_time < self.cache_duration
        except (json.JSONDecodeError, ValueError, KeyError):
            return False
    
    def _fetch_nasdaq_tickers(self) -> List[Dict[str, str]]:
        """Fetch NASDAQ tickers from Alpha Vantage LISTING_STATUS API."""
        tickers = []
        
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            
            if not api_key or api_key == 'your_api_key_here':
                logger.warning("Alpha Vantage API key not found, using static database")
                return self._get_comprehensive_ticker_list()
            
            # Use Alpha Vantage LISTING_STATUS API
            alpha_vantage_tickers = self._fetch_from_alpha_vantage_listing_status(api_key)
            if alpha_vantage_tickers:
                tickers.extend(alpha_vantage_tickers)
                logger.info(f"Fetched {len(tickers)} tickers from Alpha Vantage LISTING_STATUS API")
            else:
                logger.warning("Alpha Vantage API returned no data, using static database")
                return self._get_comprehensive_ticker_list()
                
        except Exception as e:
            logger.warning(f"Alpha Vantage API failed: {e}, using static database")
            return self._get_comprehensive_ticker_list()
        
        return tickers
    
    def _fetch_from_nasdaq_ftp(self) -> List[Dict[str, str]]:
        """Fetch current tickers from NASDAQ FTP server."""
        try:
            import ftplib
            import io
            
            tickers = []
            
            # Connect to NASDAQ FTP server
            ftp = ftplib.FTP('ftp.nasdaqtrader.com')
            ftp.login()  # Anonymous login
            
            # Get NASDAQ listed companies
            nasdaq_data = io.BytesIO()
            ftp.retrbinary('RETR SymbolDirectory/nasdaqlisted.txt', nasdaq_data.write)
            nasdaq_data.seek(0)
            
            # Parse NASDAQ data
            nasdaq_lines = nasdaq_data.read().decode('utf-8').split('\n')
            for line in nasdaq_lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split('|')
                    if len(parts) >= 2 and parts[0] and parts[0] != 'Symbol':
                        tickers.append({
                            'symbol': parts[0].strip(),
                            'name': parts[1].strip(),
                            'exchange': 'NASDAQ',
                            'sector': parts[2].strip() if len(parts) > 2 else '',
                            'industry': parts[3].strip() if len(parts) > 3 else ''
                        })
            
            # Get other listed companies (NYSE, etc.)
            other_data = io.BytesIO()
            ftp.retrbinary('RETR SymbolDirectory/otherlisted.txt', other_data.write)
            other_data.seek(0)
            
            # Parse other exchanges data
            other_lines = other_data.read().decode('utf-8').split('\n')
            for line in other_lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split('|')
                    if len(parts) >= 2 and parts[0] and parts[0] != 'ACT Symbol':
                        exchange = 'NYSE' if 'NYSE' in parts[2] else 'Other'
                        tickers.append({
                            'symbol': parts[0].strip(),
                            'name': parts[1].strip(),
                            'exchange': exchange,
                            'sector': parts[3].strip() if len(parts) > 3 else '',
                            'industry': parts[4].strip() if len(parts) > 4 else ''
                        })
            
            ftp.quit()
            logger.info(f"Fetched {len(tickers)} tickers from NASDAQ FTP")
            return tickers
            
        except Exception as e:
            logger.warning(f"NASDAQ FTP fetch failed: {e}")
            return []
    
    def _fetch_from_yfinance_sp500(self) -> List[Dict[str, str]]:
        """Fetch S&P 500 tickers using Wikipedia with SSL handling."""
        try:
            import pandas as pd
            import ssl
            import urllib.request
            
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Get S&P 500 companies list from Wikipedia
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            
            # Download the page content with SSL context
            request = urllib.request.Request(sp500_url)
            response = urllib.request.urlopen(request, context=ssl_context)
            html_content = response.read().decode('utf-8')
            
            # Read tables from HTML content
            tables = pd.read_html(html_content)
            sp500_table = tables[0]
            
            tickers = []
            for _, row in sp500_table.iterrows():
                # Determine exchange based on symbol patterns and known exchanges
                symbol = row['Symbol']
                exchange = 'NYSE'  # Default to NYSE
                
                # Some known NASDAQ patterns
                nasdaq_patterns = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE']
                if symbol in nasdaq_patterns or len(symbol) <= 4:
                    exchange = 'NASDAQ'
                
                tickers.append({
                    'symbol': symbol,
                    'name': row['Security'],
                    'exchange': exchange,
                    'sector': row.get('GICS Sector', ''),
                    'industry': row.get('GICS Sub Industry', '')
                })
            
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers
            
        except Exception as e:
            logger.warning(f"S&P 500 Wikipedia fetch failed: {e}")
            return []
    
    def _fetch_from_polygon(self) -> List[Dict[str, str]]:
        """Fetch from Polygon.io (free tier)."""
        try:
            # Polygon.io free tier endpoint
            url = "https://api.polygon.io/v3/reference/tickers"
            params = {
                'market': 'stocks',
                'active': 'true',
                'limit': 1000,
                'apikey': 'demo'  # Free demo key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                tickers = []
                for item in data.get('results', []):
                    if item.get('market') == 'stocks':
                        tickers.append({
                            'symbol': item.get('ticker', ''),
                            'name': item.get('name', ''),
                            'exchange': 'NASDAQ' if 'NASDAQ' in item.get('market', '') else 'NYSE',
                            'sector': item.get('sic_description', ''),
                            'industry': item.get('sic_description', '')
                        })
                return tickers
        except Exception as e:
            logger.warning(f"Polygon fetch failed: {e}")
        return []
    
    def _fetch_from_nasdaq_direct(self) -> List[Dict[str, str]]:
        """Try direct NASDAQ endpoints."""
        urls = [
            "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download",
            "https://www.nasdaq.com/api/v1/screener?page=1&pageSize=1000&exchange=nasdaq"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                if 'csv' in response.headers.get('content-type', '').lower():
                    csv_data = response.text
                    lines = csv_data.strip().split('\n')
                    
                    if len(lines) > 1:
                        reader = csv.DictReader(lines)
                        tickers = []
                        for row in reader:
                            if row.get('Symbol') and row.get('Symbol') != 'Symbol':
                                tickers.append({
                                    'symbol': row['Symbol'].strip(),
                                    'name': row.get('Name', '').strip(),
                                    'exchange': 'NASDAQ',
                                    'sector': row.get('Sector', '').strip(),
                                    'industry': row.get('industry', '').strip()
                                })
                        return tickers
            except Exception as e:
                logger.warning(f"NASDAQ direct fetch failed from {url}: {e}")
                continue
        return []
    
    def _fetch_from_alpha_vantage_listing_status(self, api_key: str) -> List[Dict[str, str]]:
        """Fetch tickers from Alpha Vantage LISTING_STATUS API."""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'LISTING_STATUS',
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Alpha Vantage returns CSV data for LISTING_STATUS
            csv_data = response.text
            lines = csv_data.strip().split('\n')
            
            if len(lines) < 2:
                logger.warning("Alpha Vantage returned insufficient data")
                return []
            
            tickers = []
            reader = csv.DictReader(lines)
            
            for row in reader:
                # Filter for NYSE and NASDAQ only
                exchange = row.get('exchange', '').upper()
                if exchange in ['NYSE', 'NASDAQ']:
                    # Skip ETFs and other non-stock assets, only include active stocks
                    asset_type = row.get('assetType', '').upper()
                    status = row.get('status', '').upper()
                    if asset_type == 'STOCK' and status == 'ACTIVE':
                        tickers.append({
                            'symbol': row.get('symbol', '').strip(),
                            'name': row.get('name', '').strip(),
                            'exchange': exchange,
                            'sector': '',  # Not provided by LISTING_STATUS
                            'industry': '',  # Not provided by LISTING_STATUS
                            'ipo_date': row.get('ipoDate', '').strip()
                        })
            
            logger.info(f"Fetched {len(tickers)} NYSE/NASDAQ stocks from Alpha Vantage")
            return tickers
            
        except Exception as e:
            logger.warning(f"Alpha Vantage LISTING_STATUS fetch failed: {e}")
            return []
    
    def _fetch_from_alpha_vantage(self) -> List[Dict[str, str]]:
        """Fetch from Alpha Vantage (if API key available)."""
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            
            if not api_key or api_key == 'your_api_key_here':
                return []
            
            # Use the new LISTING_STATUS method
            return self._fetch_from_alpha_vantage_listing_status(api_key)
        except Exception as e:
            logger.warning(f"Alpha Vantage fetch failed: {e}")
        return []
    
    def _fetch_nyse_tickers(self) -> List[Dict[str, str]]:
        """Fetch NYSE tickers from Alpha Vantage LISTING_STATUS API."""
        # NYSE tickers are now included in the Alpha Vantage LISTING_STATUS method
        # This method is kept for compatibility but delegates to the Alpha Vantage method
        return self._fetch_nasdaq_tickers()
    
    def _get_comprehensive_ticker_list(self) -> List[Dict[str, str]]:
        """Get a comprehensive list of major tickers as fallback."""
        from static_ticker_data import get_comprehensive_ticker_database
        return get_comprehensive_ticker_database()

    def _fetch_popular_tickers(self) -> List[Dict[str, str]]:
        """Get a curated list of popular/well-known tickers."""
        from static_ticker_data import get_popular_tickers, get_comprehensive_ticker_database
        
        popular_symbols = get_popular_tickers()
        all_tickers = get_comprehensive_ticker_database()
        
        # Filter comprehensive database to only include popular tickers
        popular_tickers = []
        for ticker in all_tickers:
            if ticker['symbol'] in popular_symbols:
                popular_tickers.append(ticker)
        
        return popular_tickers
    
    def fetch_all_tickers(self) -> List[Dict[str, str]]:
        """Fetch all tickers from various sources."""
        logger.info("Fetching ticker data from all sources...")
        
        all_tickers = []
        
        # Try to fetch from external sources first
        try:
            all_tickers.extend(self._fetch_nasdaq_tickers())
            all_tickers.extend(self._fetch_nyse_tickers())
        except Exception as e:
            logger.warning(f"Error fetching from external sources: {e}")
        
        # If we didn't get much data, use our comprehensive fallback
        if len(all_tickers) < 100:
            logger.info("Using comprehensive fallback ticker list...")
            all_tickers = self._get_comprehensive_ticker_list()
        else:
            # Remove duplicates based on symbol
            seen_symbols = set()
            unique_tickers = []
            
            for ticker in all_tickers:
                symbol = ticker['symbol'].upper()
                if symbol not in seen_symbols and len(symbol) <= 5:  # Reasonable ticker length
                    seen_symbols.add(symbol)
                    unique_tickers.append(ticker)
            
            all_tickers = unique_tickers
        
        logger.info(f"Total unique tickers fetched: {len(all_tickers)}")
        return all_tickers
    
    def get_tickers(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """Get ticker data, using cache if available and valid."""
        
        # Check if we should use cached data
        if not force_refresh and self._is_cache_valid():
            logger.info("Using cached ticker data")
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('tickers', [])
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        # Fetch fresh data
        logger.info("Fetching fresh ticker data...")
        tickers = self.fetch_all_tickers()
        
        # Cache the data
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'tickers': tickers,
            'count': len(tickers)
        }
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached {len(tickers)} tickers")
        except Exception as e:
            logger.error(f"Error caching data: {e}")
        
        return tickers
    
    def search_tickers(self, query: str, tickers: List[Dict[str, str]], limit: int = 50) -> List[Dict[str, str]]:
        """Search tickers by symbol or name."""
        if not query:
            return tickers[:limit]
        
        query = query.upper().strip()
        results = []
        
        for ticker in tickers:
            symbol = ticker.get('symbol', '').upper()
            name = ticker.get('name', '').upper()
            
            # Exact symbol match (highest priority)
            if symbol == query:
                results.insert(0, ticker)
            # Symbol starts with query
            elif symbol.startswith(query):
                results.append(ticker)
            # Name contains query
            elif query in name:
                results.append(ticker)
            # Symbol contains query
            elif query in symbol:
                results.append(ticker)
        
        return results[:limit]
    
    def get_popular_tickers(self) -> List[str]:
        """Get a list of popular ticker symbols."""
        popular = self._fetch_popular_tickers()
        return [ticker['symbol'] for ticker in popular]
    
    def get_exchange_tickers(self, exchange: str, tickers: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter tickers by exchange."""
        return [ticker for ticker in tickers if ticker.get('exchange', '').upper() == exchange.upper()]
    
    def get_sector_tickers(self, sector: str, tickers: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter tickers by sector."""
        return [ticker for ticker in tickers if sector.upper() in ticker.get('sector', '').upper()]

def main():
    """Test the ticker data manager."""
    manager = TickerDataManager()
    
    print("🔍 Testing Ticker Data Manager...")
    
    # Test fetching tickers
    tickers = manager.get_tickers()
    print(f"✅ Fetched {len(tickers)} tickers")
    
    # Test popular tickers
    popular = manager.get_popular_tickers()
    print(f"✅ Popular tickers: {len(popular)}")
    
    # Test search
    search_results = manager.search_tickers("AAPL", tickers, limit=5)
    print(f"✅ Search results for 'AAPL': {len(search_results)}")
    
    # Test exchange filtering
    nasdaq_tickers = manager.get_exchange_tickers("NASDAQ", tickers)
    print(f"✅ NASDAQ tickers: {len(nasdaq_tickers)}")
    
    print("🎉 Ticker Data Manager test completed!")

if __name__ == "__main__":
    main()
