"""
FADA - Fetal Anomaly Detection Algorithm
Streamlit Chatbot Interface

Main application entry point. UI components are modularized in web/components/.
"""

import streamlit as st
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for model imports
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from components.session_state import init_session_state
from components.model_loader import load_model, load_vqa_model, MODEL_AVAILABLE, VQA_AVAILABLE
from components.tabs import (
    render_analysis_tab,
    render_comparison_tab,
    render_about_tab,
    render_help_tab
)

# Page configuration
st.set_page_config(
    page_title="FADA - Ultrasound Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'FADA - Fetal Anomaly Detection Algorithm'
    }
)

# Initialize session state
init_session_state()

# Load models once at startup
if MODEL_AVAILABLE and st.session_state.model is None:
    st.session_state.model = load_model()
    if st.session_state.model is None:
        st.error("WARNING: Classification model failed to load.")

if VQA_AVAILABLE and st.session_state.vqa_model is None:
    st.session_state.vqa_model = load_vqa_model()

# Header
st.markdown("# FADA - Fetal Anomaly Detection Algorithm")
st.warning("RESEARCH PROTOTYPE ONLY -- Not for clinical use. All results require verification by qualified medical professionals.")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Analyses", st.session_state.total_analyses)
with col2:
    st.metric("Classification", "EfficientNet-B0")
with col3:
    st.metric("Test Accuracy", "90%")
with col4:
    vqa_status = "Enabled" if st.session_state.vqa_model else "Disabled"
    st.metric("VQA", vqa_status)

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Model Comparison", "About", "Help"])

with tab1:
    render_analysis_tab()

with tab2:
    render_comparison_tab()

with tab3:
    render_about_tab()

with tab4:
    render_help_tab()

# Footer
st.divider()
st.caption("FADA v3.1")
