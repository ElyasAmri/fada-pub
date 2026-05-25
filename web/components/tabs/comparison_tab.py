"""
Model Comparison Tab Component for FADA Web App
"""

import streamlit as st
from PIL import Image
import logging

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_MB = 10

# Check VLM availability
try:
    from src.inference.vlm_factory import create_top_vlms
    from src.data.question_loader import get_question_loader
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logger.warning("VLM interface not available")

# Available models for comparison
AVAILABLE_VLM_MODELS = {
    "minicpm": "MiniCPM-V-2.6",
    "moondream": "Moondream2",
    "internvl2_2b": "InternVL2-2B",
}


def render_comparison_tab():
    """Render the multi-model VLM comparison tab"""
    st.markdown("## Multi-Model VLM Comparison")
    st.markdown("**Research Tool** - Compare responses from multiple VLM models")

    if not VLM_AVAILABLE:
        st.warning("VLM interface not available. Install required packages.")
        return

    comp_col1, comp_col2 = st.columns([1, 2])

    with comp_col1:
        _render_upload_section()

    with comp_col2:
        _render_results_section()


def _render_upload_section():
    """Render the image upload and model selection"""
    st.markdown("### 1. Upload Image")

    vlm_uploaded_file = st.file_uploader(
        "Choose ultrasound image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload fetal ultrasound image for multi-model analysis",
        key="vlm_upload"
    )

    if vlm_uploaded_file is not None:
        if vlm_uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            st.error(f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit. Please upload a smaller image.")
            return
        vlm_image = Image.open(vlm_uploaded_file).convert('RGB')
        st.session_state.vlm_comparison_image = vlm_image
        st.image(vlm_image, caption=vlm_uploaded_file.name, use_container_width=True)

        # Model selection
        st.markdown("### Select Models (1-3)")
        selected = st.multiselect(
            "Models",
            options=list(AVAILABLE_VLM_MODELS.keys()),
            default=st.session_state.vlm_selected_models,
            format_func=lambda x: AVAILABLE_VLM_MODELS[x],
            max_selections=3,
            help="Choose 1-3 models to compare"
        )
        if selected:
            st.session_state.vlm_selected_models = selected

        # Run analysis button
        if st.button("Run Multi-Model Analysis", type="primary", use_container_width=True):
            _run_vlm_analysis(vlm_image)
    else:
        st.info("Upload an image to begin multi-model comparison")


def _run_vlm_analysis(image):
    """Run VLM analysis on the image"""
    if not st.session_state.vlm_selected_models:
        st.error("Please select at least one model")
        return

    with st.spinner("Initializing VLM models..."):
        if st.session_state.vlm_manager is None:
            st.session_state.vlm_manager = create_top_vlms(use_api=False)

    question_loader = get_question_loader()
    questions = question_loader.get_questions()

    progress_bar = st.progress(0)
    status_text = st.empty()

    responses = {}
    for idx, model_key in enumerate(st.session_state.vlm_selected_models):
        model = st.session_state.vlm_manager.get_model(model_key)
        status_text.text(f"Loading {model.model_name}...")
        model.load()

        status_text.text(f"Analyzing with {model.model_name}...")
        try:
            answers = model.answer_batch(image, questions)
            responses[model_key] = answers
        except Exception as e:
            st.error(f"Error with {model.model_name}: {e}")
            responses[model_key] = [f"Error: {str(e)}"] * len(questions)

        model.unload()
        progress_bar.progress((idx + 1) / len(st.session_state.vlm_selected_models))

    progress_bar.empty()
    status_text.empty()

    st.session_state.vlm_responses = responses
    st.session_state.vlm_analysis_complete = True
    st.session_state.vlm_user_selections = {i: None for i in range(len(questions))}
    st.success("Analysis complete!")
    st.rerun()


def _render_results_section():
    """Render the model comparison results"""
    st.markdown("### 2. Compare Model Responses")

    if not st.session_state.vlm_analysis_complete:
        st.info("Upload an image and run analysis to see results")
        return

    question_loader = get_question_loader()
    questions = question_loader.get_questions()
    short_names = question_loader.get_question_short_names()

    for q_idx, (short_name, question) in enumerate(zip(short_names, questions)):
        st.markdown(f"#### {short_name}")
        st.markdown(f"*{question}*")

        # Create columns for each model
        num_models = len(st.session_state.vlm_selected_models)
        cols = st.columns(num_models)

        for col, model_key in zip(cols, st.session_state.vlm_selected_models):
            with col:
                answer = st.session_state.vlm_responses.get(model_key, ["N/A"] * len(questions))[q_idx]
                model_name = AVAILABLE_VLM_MODELS.get(model_key, model_key)
                st.markdown(f"**{model_name}**")
                st.markdown(f"> {answer}")

        # Selection for best answer
        selection = st.radio(
            f"Best answer for {short_name}",
            options=st.session_state.vlm_selected_models,
            format_func=lambda x: AVAILABLE_VLM_MODELS.get(x, x),
            key=f"vlm_q{q_idx}",
            horizontal=True
        )
        st.session_state.vlm_user_selections[q_idx] = selection
        st.divider()

    # Summary and reset buttons
    if st.button("Show Selection Summary", use_container_width=True):
        st.markdown("### Selection Summary")
        for model_key in st.session_state.vlm_selected_models:
            count = sum(1 for v in st.session_state.vlm_user_selections.values() if v == model_key)
            st.write(f"**{AVAILABLE_VLM_MODELS.get(model_key, model_key)}**: {count} selections")

    if st.button("Reset Comparison", use_container_width=True):
        st.session_state.vlm_responses = {}
        st.session_state.vlm_analysis_complete = False
        st.session_state.vlm_user_selections = {}
        st.session_state.vlm_comparison_image = None
        st.rerun()
