"""
Analysis Tab Component for FADA Web App
"""

import streamlit as st
from PIL import Image
import logging

from ..analysis import analyze_ultrasound, generate_response
from ..model_loader import load_category_vqa, VQA_AVAILABLE

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_MB = 10


def render_analysis_tab():
    """Render the main analysis tab"""
    chat_col, preview_col = st.columns([2, 1])

    with chat_col:
        _render_chat_area()

    with preview_col:
        _render_preview_area()


def _render_chat_area():
    """Render the chat interface"""
    chat_container = st.container(height=500)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "image" in message and message.get("show_image", False):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(message["image"], width=150)
                st.markdown(message["content"])

    # Input area
    col1, col2 = st.columns([3, 1])

    with col1:
        prompt = st.chat_input("Ask about the ultrasound analysis...")

    with col2:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            label_visibility="collapsed",
            key="upload_widget"
        )

    # Process text input
    if prompt:
        _handle_text_input(prompt)

    # Process image upload
    if uploaded_file is not None and not st.session_state.analyzing:
        if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            st.error(f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit. Please upload a smaller image.")
        else:
            _handle_image_upload(uploaded_file)


def _handle_text_input(prompt):
    """Handle text input from chat"""
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vqa_enabled and st.session_state.vqa_model and st.session_state.current_image:
        medical_keywords = ["what", "which", "where", "how", "describe", "identify", "assess", "evaluate", "measure", "show", "visible"]
        if any(keyword in prompt.lower() for keyword in medical_keywords):
            with st.spinner("Loading VQA model and analyzing image..."):
                try:
                    answer = st.session_state.vqa_model.answer_question(
                        st.session_state.current_image,
                        prompt
                    )
                    response = answer
                except Exception as e:
                    response = f"VQA error: {str(e)}"
        else:
            response = "Upload an ultrasound image for detailed analysis. "
            response += "You can ask specific medical questions about the image."
    else:
        if "explain" in prompt.lower() or "what" in prompt.lower():
            response = "The analysis uses deep learning models trained on fetal ultrasound data. "
            response += "Each anatomical view has specific features that the model identifies."
        elif "measure" in prompt.lower() or "size" in prompt.lower():
            response = "Biometric measurements vary by anatomical view and gestational age. "
            response += "The system provides classification and confidence scores."
        else:
            response = "Upload an ultrasound image for detailed analysis. "
            response += "The system can identify anatomical views and detect patterns."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


def _handle_image_upload(uploaded_file):
    """Handle image upload and analysis"""
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if file_id != st.session_state.last_uploaded_file:
        st.session_state.analyzing = True
        st.session_state.last_uploaded_file = file_id

        image = Image.open(uploaded_file)
        st.session_state.current_image = image

        st.session_state.messages.append({
            "role": "user",
            "content": f"Uploaded: {uploaded_file.name}",
            "image": image,
            "show_image": True
        })

        with st.spinner("Analyzing ultrasound image..."):
            analysis_results = analyze_ultrasound(image, st.session_state.model)
            response = generate_response(analysis_results)

            st.session_state.debug_info.append({
                'image': uploaded_file.name,
                'organ': analysis_results.get('organ'),
                'confidence': analysis_results.get('confidence', 0),
                'model_available': st.session_state.model is not None
            })

            st.session_state.last_classification = analysis_results.get("organ", "")

            detected_category = st.session_state.last_classification
            if VQA_AVAILABLE and detected_category:
                try:
                    category_vqa = load_category_vqa(detected_category)
                    if category_vqa:
                        st.session_state.vqa_model = category_vqa
                        st.session_state.vqa_enabled = True
                        response += "\n\n---\n\n### Visual Question Answering Available\n"
                        response += f"Loaded specialized VQA model for {detected_category}. Click the question buttons below or ask your own questions in the chat."
                    else:
                        st.session_state.vqa_enabled = False
                except Exception as e:
                    logger.error(f"VQA loading error: {e}")
                    st.session_state.vqa_enabled = False
            else:
                st.session_state.vqa_enabled = False

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            st.session_state.total_analyses += 1
            st.session_state.analyzing = False
            st.rerun()


def _render_preview_area():
    """Render the image preview and VQA buttons"""
    st.markdown("### Current Analysis")

    if st.session_state.current_image:
        st.image(st.session_state.current_image, use_container_width=True)

        if st.session_state.vqa_enabled and st.session_state.vqa_model:
            st.markdown("#### Ask Medical Questions")
            vqa_shortcuts = st.session_state.vqa_model.get_question_shortcuts()

            with st.expander("Standard Questions", expanded=True):
                for short_name, full_question in vqa_shortcuts.items():
                    if st.button(short_name, key=f"vqa_btn_{short_name}", use_container_width=True):
                        st.session_state.messages.append({
                            "role": "user",
                            "content": full_question
                        })

                        with st.spinner(f"Loading VQA model and answering: {short_name}..."):
                            try:
                                answer = st.session_state.vqa_model.answer_question(
                                    st.session_state.current_image,
                                    full_question
                                )
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"**{short_name}**\n\n{answer}"
                                })
                            except Exception as e:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error loading VQA model: {str(e)}"
                                })

                        st.rerun()

        # Quick actions
        st.markdown("#### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear", use_container_width=True):
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "Chat cleared. Upload a new image to begin."
                }]
                st.session_state.current_image = None
                st.session_state.last_uploaded_file = None
                st.session_state.vqa_enabled = False
                st.session_state.last_classification = None
                st.rerun()
        with col2:
            if st.button("Export", use_container_width=True):
                st.info("Export feature coming soon")
    else:
        st.info("Upload an ultrasound image to see preview and analysis")

        with st.expander("Supported Views (12 Classes)", expanded=False):
            st.markdown("""
            - Abdomen
            - Aortic Arch
            - Cervical View
            - Cervix
            - Femur
            - Non-standard NT
            - Fetal Head Position
            - Standard NT
            - Thorax
            - Transcerebellar Plane
            - Transthalamic Plane
            - Transventricular Plane
            """)
