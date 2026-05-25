"""
Session State Initialization for FADA Web App
"""

import streamlit as st


def init_session_state():
    """Initialize all session state variables"""

    # Chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Welcome to FADA Ultrasound Analysis System. Upload an ultrasound image to begin analysis."
        })

    # Models
    if "model" not in st.session_state:
        st.session_state.model = None

    if "vqa_model" not in st.session_state:
        st.session_state.vqa_model = None

    if "vqa_enabled" not in st.session_state:
        st.session_state.vqa_enabled = False

    # Analysis state
    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False

    if "total_analyses" not in st.session_state:
        st.session_state.total_analyses = 0

    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

    if "last_classification" not in st.session_state:
        st.session_state.last_classification = None

    if "debug_info" not in st.session_state:
        st.session_state.debug_info = []

    # Multi-model comparison state
    if "vlm_manager" not in st.session_state:
        st.session_state.vlm_manager = None

    if "vlm_responses" not in st.session_state:
        st.session_state.vlm_responses = {}

    if "vlm_selected_models" not in st.session_state:
        st.session_state.vlm_selected_models = ["minicpm", "moondream"]

    if "vlm_user_selections" not in st.session_state:
        st.session_state.vlm_user_selections = {}

    if "vlm_analysis_complete" not in st.session_state:
        st.session_state.vlm_analysis_complete = False

    if "vlm_comparison_image" not in st.session_state:
        st.session_state.vlm_comparison_image = None
