import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Hello",
    page_icon="🤖",
    layout = "centered"
)
st.sidebar.success("Select a demo above.")

st.write("# Welcome to IntelliChat!👋")

st.markdown(
    """
    IntelliChat is an app built specifically for
    Chat based and Conversational AI projects.

    *Who is this for?* - Businesses and Enterprises who
    are looking to optimize and enhance their
    Conversational AI Workflow.
    
    **👈 Select a demo from the sidebar** to see some examples
    of what IntelliChat can do!
"""
)
