
import streamlit as st

from api import Phi3VisionAI
from src.ui_components import file_upload, header

# Define Phi3 model client
client = Phi3VisionAI()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"}


# Define path to store chat history file
CHAT_HISTORY_FILE = "messages.json"
IMAGE_DIR = "uploaded_images"


def main():
    # Title section
    header()

    # Sidebar section for file upload
    uploaded_file, file_object = file_upload()

    print(uploaded_file)
    print(file_object.__len__())

    # Load chat history from file
    messages = None
    if messages is None:
        messages = []

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = messages

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # skip if role is system
        if message["role"] == "system":
            continue
        # Display chat message in chat message container
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input(
        "Ask something", disabled=uploaded_file is None, key="prompt"
    ):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # with st.spinner("Thinking..."):
        # Placeholder function to send message to Phi3 model API
        stream = client.chat(st.session_state.messages, file_object)

        # Display Phi3 response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        # Add Phi3 response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history before closing the app
    # save_chat_history(st.session_state.messages)

    # Made with section
    # st.markdown("---")
    # st.markdown("Made with ❤️ by [Bhimraj Yadav](https://github.com/bhimrazy)")





if __name__ == "__main__":
    main()
