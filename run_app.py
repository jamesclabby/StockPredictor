#!/usr/bin/env python3
"""
Launch script for the Stock Sentiment Analysis Streamlit app.

This script checks dependencies and launches the Streamlit application.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'requests', 'dotenv', 'transformers', 'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_env_file():
    """Check if .env file exists and has API key."""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📝 Create a .env file with your Alpha Vantage API key:")
        print("   echo 'ALPHA_VANTAGE_API_KEY=your_api_key_here' > .env")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'your_api_key_here' in content or 'ALPHA_VANTAGE_API_KEY=' not in content:
            print("⚠️  .env file exists but API key may not be set properly")
            print("📝 Make sure your .env file contains:")
            print("   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here")
            return False
    
    print("✅ .env file found with API key!")
    return True

def launch_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Launching Stock Sentiment Analysis app...")
    print("🌐 The app will open in your default web browser")
    print("📱 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        # Use headless mode to avoid email prompt
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    
    return True

def main():
    """Main function."""
    print("📈 Stock Sentiment Analysis - Launch Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check .env file
    if not check_env_file():
        return 1
    
    # Launch Streamlit
    if not launch_streamlit():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
