import streamlit as st

from src.api import client
from src.config import MODEL, SYSTEM_MESSAGE
from src.ui_components import file_upload, header
from src.utils import prepare_content_with_images, is_image, is_video

def main():
    # Title section
    header()

    # Add input field for system prompt
    st.sidebar.header("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Modify the prompt here.", value=SYSTEM_MESSAGE["content"]
    )
    SYSTEM_MESSAGE["content"] = system_prompt

    # Sidebar section for file upload
    uploaded_file, file_object = file_upload()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    if "messages" in st.session_state.keys() and len(st.session_state.messages) > 0:
        # add clear chat history button to sidebar
        st.sidebar.button(
            "Clear Chat History",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Display chat message in chat message container
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, list):
                st.markdown(content[0]["text"])
                caption = "Thumbnail for Video" if len(content) > 2 else ""
                st.image(
                    content[1]["image_url"]["url"],
                    width=200,
                    caption=caption,
                )
            else:
                st.markdown(content)

    if prompt := st.chat_input("Ask something", key="prompt"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            content = (
                prepare_content_with_images(prompt, file_object)
                if file_object
                else prompt
            )
            if file_object:
                if is_image(uploaded_file.name):
                    caption = "Thumbnail of Video" if len(file_object) > 1 else ""
                    st.image(
                        file_object[0]["image_url"]["url"], width=200, caption=caption
                    )

                elif is_video(uploaded_file.name):
                    st.video(uploaded_file, autoplay=True)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": content})

        # Get response from the assistant
        with st.chat_message("assistant"):
            messages = [SYSTEM_MESSAGE, *st.session_state.messages]
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, stream=True
            )
            response = st.write_stream(stream)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
