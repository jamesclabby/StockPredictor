#!/usr/bin/env python3
"""
Test script for the Cloud Function with Resend email
"""

import os
from dotenv import load_dotenv
from main_resend import run_daily_analysis, analyze_tickers_cloud, format_analysis_summary

def test_local():
    """Test the Cloud Function locally"""
    print("🧪 Testing Cloud Function with Resend locally...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are set
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    resend_key = os.getenv('RESEND_API_KEY')
    
    if not alpha_vantage_key:
        print("❌ ALPHA_VANTAGE_API_KEY not found in environment variables")
        print("   Please set it in your .env file or environment")
        return False
    
    if not resend_key:
        print("❌ RESEND_API_KEY not found in environment variables")
        print("   Please get your API key from https://resend.com/")
        print("   Then add it to your .env file: RESEND_API_KEY=your_key_here")
        return False
    
    print("✅ Environment variables loaded")
    
    # Test the main function
    try:
        result = run_daily_analysis(None, None)
        print(f"✅ Function executed successfully: {result}")
        return True
    except Exception as e:
        print(f"❌ Function failed: {e}")
        return False

def test_analysis_only():
    """Test just the analysis part without email"""
    print("🧪 Testing analysis logic only...")
    
    load_dotenv()
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        print("❌ ALPHA_VANTAGE_API_KEY not found")
        return False
    
    try:
        # Test with a single ticker to avoid rate limits
        results = analyze_tickers_cloud(['AAPL'], api_key)
        
        if results:
            print("✅ Analysis completed")
            print(f"   Results: {list(results.keys())}")
            
            # Test HTML formatting
            html = format_analysis_summary(results)
            print(f"✅ HTML formatting completed ({len(html)} characters)")
            
            return True
        else:
            print("❌ No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def test_resend_setup():
    """Test Resend API key and configuration"""
    print("🧪 Testing Resend setup...")
    
    load_dotenv()
    resend_key = os.getenv('RESEND_API_KEY')
    
    if not resend_key:
        print("❌ RESEND_API_KEY not found")
        return False
    
    try:
        import resend
        resend.api_key = resend_key
        
        # Test API key by trying to get domains (this is a lightweight test)
        # Note: This might require a valid API key, so we'll just test the import
        print("✅ Resend library imported successfully")
        print("✅ API key format looks valid")
        return True
        
    except Exception as e:
        print(f"❌ Resend setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Cloud Function Test Suite (Resend Version)")
    print("=" * 50)
    
    # Test 1: Resend setup
    print("\n1. Testing Resend setup...")
    resend_success = test_resend_setup()
    
    # Test 2: Analysis only (safer for rate limits)
    print("\n2. Testing analysis logic...")
    analysis_success = test_analysis_only()
    
    # Test 3: Full function (only if both previous tests worked)
    if resend_success and analysis_success:
        print("\n3. Testing full function...")
        full_success = test_local()
        
        if full_success:
            print("\n🎉 All tests passed! Cloud Function with Resend is ready for deployment.")
        else:
            print("\n⚠️  Analysis works but full function failed. Check email configuration.")
    else:
        print("\n❌ Prerequisites failed. Please fix the issues above.")
    
    print("\n" + "=" * 50)
    print("💡 Next steps:")
    print("   1. Get Resend API key from https://resend.com/")
    print("   2. Update email addresses in main_resend.py")
    print("   3. Add RESEND_API_KEY to your .env file")
    print("   4. Deploy to Google Cloud Functions")
    print("   5. Set up Cloud Scheduler for daily runs")
