# Financial News Analysis Agent

A Python script that fetches financial news for specified stock tickers and performs sentiment analysis on the news headlines using Hugging Face's transformers library.

## Features

- Fetches real-time financial news from Alpha Vantage API
- Performs sentiment analysis on news headlines using DistilBERT
- Filters news to only include articles from the last 24 hours
- Provides clean, formatted output grouped by stock ticker
- **Robust Error Handling:**
  - Custom exception classes with detailed error classification
  - Retry logic with exponential backoff for transient failures
  - Circuit breaker pattern to prevent API abuse during outages
  - Intelligent rate limit detection and handling
  - Comprehensive logging system with structured error reporting
  - Graceful degradation with user-friendly error messages

## Prerequisites

- Python 3.9 or higher
- Alpha Vantage API key (free tier available)

## Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get an Alpha Vantage API key:**
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up for a free account
   - Copy your API key

4. **Create a `.env` file:**
   ```bash
   # Create .env file in the project root
   echo "ALPHA_VANTAGE_API_KEY=your_actual_api_key_here" > .env
   ```
   
   Or manually create a `.env` file with:
   ```
   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
   ```

## Usage

### Command Line Interface

Run the script:
```bash
python financial_news_analyzer.py
```

### Web Interface (Streamlit)

For an interactive web interface with visualizations:

1. **Install additional dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the web app:**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the web interface:**
   - The app will automatically open in your browser
   - If not, go to: http://localhost:8501

### Web App Features

- **Interactive Ticker Selection**: Multi-select dropdown with popular stocks
- **Custom Ticker Input**: Add any ticker not in the dropdown
- **Real-time Analysis**: Live sentiment analysis with progress indicators
- **Interactive Visualizations**: 
  - Sentiment distribution pie charts
  - Confidence score bar charts
  - Color-coded results (green=positive, red=negative)
- **Detailed Results**: Individual article analysis with confidence scores
- **Export Functionality**: Download results as CSV
- **Responsive Design**: Works on desktop and mobile devices

## Configuration

### Stock Tickers
The script is configured to analyze these stock tickers by default:
- AAPL (Apple)
- GOOGL (Google/Alphabet)
- TSLA (Tesla)

To change the tickers, edit the `tickers` list in the `main()` function:
```python
tickers = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']  # Add your preferred tickers
```

### Sentiment Analysis Model
The script uses `distilbert-base-uncased-finetuned-sst-2-english` for sentiment analysis. This model:
- Is lightweight and fast
- Provides good accuracy for general sentiment analysis
- Works well with financial news headlines

## Output Format

The script provides a clean, formatted output like this:

```
=== Financial News Analysis Agent ===

Initializing sentiment analysis model...
✓ Sentiment analysis model loaded successfully

Fetching news for AAPL...
Found 15 articles for AAPL
Found 3 articles from last 24 hours
Analyzed 2 articles with clear sentiment

--- Daily Financial News Summary ---

Ticker: AAPL
[POSITIVE] (confidence: 0.95) Apple announces record-breaking iPhone sales for the quarter.
[NEGATIVE] (confidence: 0.87) EU regulators launch new probe into Apple's App Store policies.

Ticker: GOOGL
[POSITIVE] (confidence: 0.92) Google's new AI features drive stock price higher.
```

## API Rate Limits

- Alpha Vantage free tier: 25 requests per day
- The script makes one request per ticker
- With 3 default tickers, you can run the script ~8 times per day

## Error Handling

The script includes comprehensive error handling with the following features:

### Error Classification System
- **Network Errors**: Connection timeouts, DNS failures, network unreachable
- **API Errors**: Server errors (5xx), invalid requests (4xx), malformed responses  
- **Rate Limiting**: Different types of rate limits (per minute, per day, burst limits)
- **Authentication Errors**: Invalid API keys, expired tokens
- **Data Parsing Errors**: Malformed JSON, missing required fields
- **Model Errors**: Sentiment analysis failures, model loading issues

### Retry Strategy
- **Exponential Backoff**: 3 retries with increasing delays (1s, 2s, 4s)
- **Smart Retry Logic**: Different strategies for different error types
- **Rate Limit Respect**: Honors Retry-After headers from API responses
- **Non-Retryable Errors**: Immediate failure for auth/config errors

### Circuit Breaker Pattern
- **Failure Tracking**: Monitors consecutive failures per ticker
- **Auto-Recovery**: Opens circuit after 5 consecutive failures
- **Half-Open State**: Tests recovery before fully reopening
- **Cooldown Period**: 5-minute recovery timeout

### User Experience
- **Clear Error Messages**: User-friendly feedback with emoji indicators
- **Graceful Degradation**: Continues processing other tickers if one fails
- **Progress Tracking**: Shows success/failure statistics
- **Detailed Logging**: Comprehensive logs saved to `financial_news_analyzer.log`

## Dependencies

### Core Dependencies
- `requests`: HTTP library for API calls
- `python-dotenv`: Environment variable management
- `transformers`: Hugging Face transformers library
- `torch`: PyTorch backend for transformers

### Web Interface Dependencies
- `streamlit`: Web application framework
- `plotly`: Interactive visualizations
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing

## File Structure

```
StockAnalysis/
├── financial_news_analyzer.py    # Core analysis engine
├── streamlit_app.py              # Web interface
├── run_app.py                    # Launch script
├── requirements.txt              # Python dependencies
├── .env                          # API key (not tracked)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Web App Architecture

The Streamlit application (`streamlit_app.py`) provides:

### Frontend Components
- **Multi-select Dropdown**: Popular stock tickers (AAPL, GOOGL, TSLA, etc.)
- **Custom Input Field**: Add any ticker not in the dropdown
- **Progress Indicators**: Real-time analysis progress
- **Interactive Tabs**: Results organized by ticker
- **Export Functionality**: CSV download with timestamps

### Visualization Features
- **Sentiment Distribution**: Pie charts showing positive/negative breakdown
- **Confidence Scores**: Bar charts with color-coded confidence levels
- **Color Coding**: Green for positive, red for negative sentiment
- **Responsive Design**: Adapts to different screen sizes

### Integration
- **Session State**: Maintains analysis results across interactions
- **Error Handling**: Graceful handling of API errors and timeouts
- **Circuit Breaker**: Prevents API abuse during outages
- **Real-time Updates**: Live progress and status updates

## Troubleshooting

### Common Issues

1. **"Please set your ALPHA_VANTAGE_API_KEY" error:**
   - Make sure you have a `.env` file in the project root
   - Verify the API key is correct and active

2. **"API rate limit reached" error:**
   - You've exceeded the free tier limit (25 requests/day)
   - Wait until the next day or upgrade to a paid plan

3. **Model loading errors:**
   - Ensure you have a stable internet connection
   - The model will be downloaded on first run (~500MB)

4. **No articles found:**
   - Check if the ticker symbols are correct
   - Verify your API key has access to news data

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.
