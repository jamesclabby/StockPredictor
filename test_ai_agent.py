#!/usr/bin/env python3
"""
Test script for AI Agent functionality
Loads API keys from .env file for local testing
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and test
try:
    from main import run_daily_analysis
    print("🚀 Starting AI Agent test...")
    result = run_daily_analysis()
    print("✅ Test completed!")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
