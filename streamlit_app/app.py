"""
DrugQuery Streamlit Frontend.

Run with: streamlit run streamlit_app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="DrugQuery - FDA Drug Information",
    page_icon="💊",
    layout="wide",
)

st.title("💊 DrugQuery")
st.markdown("*AI-powered FDA drug information assistant*")

st.warning("⚠️ This frontend will be implemented in Phase 6.")

st.markdown("""
## Planned Features

- Natural language Q&A about medications
- Citations to original FDA sources
- Drug name filtering
- Query statistics

## Coming Soon

Check back after completing Phases 1-5!
""")
