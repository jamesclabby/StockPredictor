#!/usr/bin/env python3
"""
Simple test to debug button click issues
"""

import streamlit as st

st.title("Button Click Test")

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

# Simple button test
if st.button("Test Button"):
    st.write("✅ Button was clicked!")
    st.session_state.test_results.append("Button clicked")
else:
    st.write("❌ Button not clicked")

# Display results
st.write("Session state results:", st.session_state.test_results)

# Test with tickers
tickers = ['AAPL', 'GOOGL']
selected = st.multiselect("Select tickers", tickers, default=tickers)

if st.button("Analyze Selected Tickers"):
    st.write(f"Analyzing: {selected}")
    if selected:
        st.write("✅ Would call analyze function here")
    else:
        st.write("❌ No tickers selected")
