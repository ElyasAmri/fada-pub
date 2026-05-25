"""
About Tab Component for FADA Web App
"""

import streamlit as st


def render_about_tab():
    """Render the about section tab"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## About FADA")
        st.markdown("""
        FADA (Fetal Anomaly Detection Algorithm) is an automated analysis system for fetal
        ultrasound images using deep learning technology.

        ### Key Features
        - **Automated Analysis**: Instant classification of ultrasound views
        - **Multi-Class Detection**: Supports 12 different anatomical views
        - **Visual Question Answering**: Ask medical questions about Non-standard NT images
        - **Confidence Scoring**: Provides reliability metrics for each prediction
        - **Interactive Interface**: Chat-based interaction for ease of use

        ### Technical Architecture
        **Phase 1 - Classification**:
        - Model: EfficientNet-B0 backbone with custom classification heads
        - Accuracy: ~90% on 12-class ultrasound view classification

        **Phase 2 - Visual Question Answering (VQA)**:
        - Model: BLIP-2 OPT-2.7B with LoRA fine-tuning
        - Supports 8 medical questions for Non-standard NT images
        - Trained on 50 annotated images with expert annotations

        ### Framework
        - PyTorch for deep learning
        - Streamlit for web interface
        - Input: 224x224 RGB ultrasound images
        - Output: Classification + medical Q&A
        """)

    with col2:
        st.markdown("### System Information")

        info_container = st.container(border=True)
        with info_container:
            st.markdown("**Version:** 3.1")
            st.markdown("**Classification:** EfficientNet-B0")
            st.markdown("**VQA:** BLIP-2 OPT-2.7B")
            st.markdown("**Classes:** 12")
            st.markdown("**Updated:** October 2025")

        st.markdown("### Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "~90%")
        with col2:
            st.metric("Speed", "< 1s")
