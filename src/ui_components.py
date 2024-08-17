import streamlit as st
from src.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from src.utils import is_image, is_video, encode_image, encode_video


def file_upload():
    # Sidebar header
    st.sidebar.header("Upload Files")
    uploaded_file = None

    # File uploader with improved grammar and standardized code
    uploaded_file = st.sidebar.file_uploader(
        "Please select an image or video file...",
        type=IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
    )

    if uploaded_file is not None:
        # check if the file is image or video
        if is_image(uploaded_file.name):
            st.sidebar.image(uploaded_file, use_column_width=True)
            image_object = encode_image(uploaded_file)
            return uploaded_file, [image_object]

        elif is_video(uploaded_file.name):
            st.sidebar.video(uploaded_file)
            video_object = encode_video(uploaded_file)
            return uploaded_file, video_object
        else:
            st.sidebar.warning(
                "Unsupported file type. Please upload an image or video file."
            )
    return None, None


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
        <img src="https://azure.microsoft.com/en-us/blog/wp-content/uploads/2024/05/Azure_Blog_Isometric_Illustration-12_1260x708.jpg" class="center-cropped">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align: center;'>Chat with Phi-3-vision-128k-instruct</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='text-align: center; margin-bottom:4'>"
        "Phi-3-vision is a lightweight, state-of-the-art 4.2 billion parameter multimodal model with language and vision capabilities, available with a 128k context length. <a href='https://huggingface.co/microsoft/Phi-3-vision-128k-instruct' target='_blank'>Read more</a>"
        "<br>"
        "<p>Made with ❤️ by <a href='https://github.com/bhimrazy' target='_blank'>Bhimraj Yadav</a></p>"
        "</div>",
        unsafe_allow_html=True,
    )
