"""
Help Tab Component for FADA Web App
"""

import streamlit as st

from ..model_loader import MODEL_AVAILABLE, VQA_AVAILABLE


def render_help_tab():
    """Render the help and documentation tab"""
    st.markdown("## Help & Documentation")

    _render_debug_section()
    _render_getting_started()
    _render_supported_formats()
    _render_understanding_results()
    _render_troubleshooting()


def _render_debug_section():
    """Render debug information expander"""
    with st.expander("Debug Information", expanded=False):
        st.markdown("### Import Status")
        st.write(f"MODEL_AVAILABLE: {MODEL_AVAILABLE}")
        st.write(f"VQA_AVAILABLE: {VQA_AVAILABLE}")

        st.markdown("### Model Status")
        st.write(f"Classification model loaded: {st.session_state.model is not None}")
        st.write(f"VQA model instance: {st.session_state.vqa_model is not None}")

        try:
            from src.models.vqa_model import UltrasoundVQA as TestVQA
            st.success("VQA module can be imported directly")
        except Exception as e:
            st.error(f"VQA import error: {e}")

        if st.session_state.debug_info:
            st.markdown("### Recent Classifications")
            for idx, info in enumerate(st.session_state.debug_info[-5:]):
                st.write(f"**{idx+1}. {info['image']}**")
                st.write(f"   - Organ: {info['organ']}")
                st.write(f"   - Confidence: {info['confidence']:.1%}")
                st.write(f"   - Model: {'Yes' if info['model_available'] else 'No'}")


def _render_getting_started():
    """Render getting started section"""
    with st.expander("Getting Started", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        1. **Upload an Image**: Click the "Upload Image" button in the Analysis tab
        2. **Wait for Processing**: The system will analyze the ultrasound automatically
        3. **Review Results**: Check the detected view, confidence score, and assessment
        4. **Ask Questions**: Use the chat input or question buttons (if VQA is enabled)

        ### Using Visual Question Answering (VQA)
        When you upload a Non-standard NT image:
        - VQA mode will be automatically enabled
        - 8 standard medical question buttons will appear on the right
        - Click any button to get instant answers
        - Or type your own medical questions in the chat
        - VQA provides detailed medical analysis beyond classification
        """)


def _render_supported_formats():
    """Render supported formats section"""
    with st.expander("Supported Image Formats"):
        st.markdown("""
        The system accepts the following image formats:
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        - BMP (.bmp)
        - TIFF (.tiff)

        **Recommended**: Images should be at least 224x224 pixels for best results.
        """)


def _render_understanding_results():
    """Render results explanation section"""
    with st.expander("Understanding Results"):
        st.markdown("""
        ### Analysis Output
        - **Detected View**: The anatomical structure identified
        - **Confidence**: How certain the model is (0-100%)
        - **Image Quality**: Assessment of the input image quality
        - **Orientation**: The viewing angle of the ultrasound

        ### Confidence Interpretation
        - **> 90%**: High confidence
        - **70-90%**: Moderate confidence
        - **< 70%**: Low confidence - consider re-uploading
        """)


def _render_troubleshooting():
    """Render troubleshooting section"""
    with st.expander("Troubleshooting"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Common Issues
            **Image not uploading?**
            - Check file format and size
            - Refresh the page

            **Unexpected results?**
            - Ensure good image quality
            - Try different viewing angles
            """)

        with col2:
            st.markdown("""
            ### Tips for Best Results
            - Use clear, high-contrast images
            - Avoid heavily cropped images
            - Standard clinical views work best
            - Multiple angles improve accuracy
            """)
