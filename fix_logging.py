# Fix 4: Enhanced Logging for Better Debugging

def run_daily_analysis(event=None, context=None):
    """
    Main entry point for Google Cloud Functions.
    This function is triggered by Cloud Scheduler or HTTP.
    Enhanced with comprehensive logging.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting AI-powered financial news analysis")
        logger.info(f"Event: {event}")
        logger.info(f"Context: {context}")
        logger.info("=" * 60)
        
        # Run AI agent analysis with market scanner
        logger.info("Calling run_ai_agent_analysis with empty ticker list")
        results = run_ai_agent_analysis([])  # Empty list since agent will discover tickers dynamically
        
        logger.info(f"Agent analysis results: {results}")
        
        # Format results as HTML
        logger.info("Formatting results as HTML")
        summary_html = format_agent_analysis_summary(results)
        
        logger.info(f"HTML summary length: {len(summary_html)} characters")
        
        # Send email
        logger.info("Sending summary email")
        email_sent = send_summary_email(summary_html)
        
        if email_sent:
            logger.info("✅ AI agent analysis completed successfully")
            return "AI-powered analysis completed and email sent successfully."
        else:
            logger.error("❌ Failed to send summary email")
            return "AI analysis completed but failed to send email."
            
    except Exception as e:
        logger.error(f"❌ Error in daily analysis: {e}")
        logger.exception("Full traceback:")
        return f"Error: {str(e)}"
