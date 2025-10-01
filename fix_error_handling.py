# Fix 2: Comprehensive Error Handling for Agent Execution

def run_ai_agent_analysis(tickers: List[str]) -> Dict:
    """
    Run AI agent analysis using LangChain ReAct agent with market scanner.
    Enhanced with comprehensive error handling.
    """
    try:
        # Initialize AI components
        initialize_ai_components()
        
        # Define tools
        tools = [scan_market_for_trending_tickers, fetch_stock_news, analyze_headline_sentiment, get_stock_performance]
        
        # Get the ReAct prompt template
        prompt = hub.pull("hwchase17/react")
        
        # Create the agent and executor with error handling
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,  # Handle LLM output parsing errors
            max_iterations=10,  # Limit iterations to prevent infinite loops
            early_stopping_method="generate"  # Stop early if agent gets stuck
        )
        
        # Create the new, high-level prompt for the agent
        master_prompt = """
        Your mission is to create a daily market trends summary email.

        First, you MUST use the 'scan_market_for_trending_tickers' tool to discover which stocks are the most significant market movers today. If the tool returns "No trending tickers found today" or similar, your final answer should be a simple message like 'No trending tickers were found today.'

        Then, for each of the tickers returned by that tool, you must perform a full analysis by sequentially using your other tools: fetch its news, analyze the sentiment of that news, and get its recent stock performance.

        Finally, compile all the individual analyses into a single, comprehensive report formatted as a string, ready to be sent as an email. Structure the report with clear headings for each ticker.
        """
        
        # Invoke the agent with comprehensive error handling
        logger.info("Running AI agent with market scanner for dynamic ticker discovery")
        
        try:
            final_report_dict = agent_executor.invoke({"input": master_prompt})
            final_report_string = final_report_dict['output']
            
            logger.info(f"Agent execution completed successfully. Output length: {len(final_report_string)}")
            
        except StopIteration as e:
            logger.error(f"Agent execution stopped with StopIteration: {e}")
            final_report_string = "Market analysis could not be completed due to technical issues with the AI agent."
            
        except Exception as e:
            logger.error(f"Agent execution failed with error: {e}")
            final_report_string = f"Market analysis could not be completed due to technical error: {str(e)}"
        
        # Return the report as a single entry for the email formatter
        return {
            'market_analysis': {
                'summary': final_report_string,
                'status': 'success'
            }
        }
        
    except Exception as e:
        logger.error(f"Error running AI agent with market scanner: {e}")
        return {
            'market_analysis': {
                'error': f"AI agent analysis failed: {str(e)}",
                'status': 'error'
            }
        }
