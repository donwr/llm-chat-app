from backend.core import run_llm
import streamlit as st
from typing import Set

# Add custom CSS for chat container and messages
st.markdown(
    """
    <style>
    .chat-container {
        display: flex !important;
        flex-direction: column;
        align-items: flex-start;
    }
    .user-message, .bot-message {
        max-width: 80%;
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
        overflow-wrap: break-word;
        word-wrap: break-word;
        hyphens: auto;
    }
    .user-message {
        align-self: flex-end;
        background-color: #2c2a31;
    }
    .bot-message {
        background-color: #1e1e1e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Langchain Course - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

# Initialize the session state
if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []

# Chat Answer History
if "chat_answer_history" not in st.session_state:
    st.session_state.chat_answer_history = []

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Take the list of the urls and print them
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response"):
        # Generate the response
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state.chat_history
        )

        # Get the sources
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        # Format the response
        formatted_response = (
            f"{generated_response['answer']}\n\n {create_sources_string(sources)}"
        )

        # Add the prompt and response to the chat history
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answer_history.append(formatted_response)
        st.session_state.chat_history.append((prompt, generated_response["answer"]))

# Custom chat UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(
        st.session_state.chat_answer_history, st.session_state.user_prompt_history
    ):
        st.markdown(f"<div class='user-message'>User: {user_query}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-message'>Bot: {generated_response}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)