import streamlit as st
from src.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from src.utils import is_image, is_video, encode_image, encode_video


def file_upload():
    # Sidebar header
    st.sidebar.header("Upload Files")
    uploaded_file, file_object = None, None

    # File uploader with improved grammar and standardized code
    uploaded_file = st.sidebar.file_uploader(
        "Please select an image or video file...",
        type=IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
    )

    if uploaded_file is not None:
        # check if the file is image or video
        if is_image(uploaded_file.name):
            st.sidebar.image(uploaded_file, use_column_width=True)
            file_object = [encode_image(uploaded_file)]

        elif is_video(uploaded_file.name):
            with st.sidebar.status("Processing video..."):
                file_object = encode_video(uploaded_file)
                st.sidebar.video(uploaded_file)
        else:
            st.sidebar.warning(
                "Unsupported file type. Please upload an image or video file."
            )

    return uploaded_file, file_object


def header():
    # CSS to center crop the image in a circle
    circle_image_css = """
    <style>
    .center-cropped {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 50%;
        width: 96px;
        height: 96px;
        object-fit: cover;
    }
    </style>
    """

    # Inject CSS
    st.markdown(circle_image_css, unsafe_allow_html=True)

    st.markdown(
        """
        <img src="https://github.com/user-attachments/assets/f648d3bc-fd96-4102-9302-7d549fcf5eaa" class="center-cropped">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align: center;'>Chat with Vision LLM</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align: center; margin-bottom:4'>"
        "<h5 style='text-align: center;'>A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone</h5>"
        "<b>MiniCPM-V 2.6</b> is the latest and most capable model in the MiniCPM-V series. <a href='https://huggingface.co/openbmb/MiniCPM-V-2_6' target='_blank'>Read more</a>"
        "<br>"
        # "<p>Made with ❤️ by <a href='https://github.com/bhimrazy' target='_blank'>Bhimraj Yadav</a></p>"
        "</div>",
        unsafe_allow_html=True,
    )
