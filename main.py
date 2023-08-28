from backend.core import run_llm
import streamlit as st
from typing import Set
from streamlit_chat import message

st.header("Langchain Course - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

# Initialize the session state
if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []

# Chat Answer History
if "chat_answer_history" not in st.session_state:
    st.session_state.chat_answer_history = []



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
        generated_response = run_llm(query=prompt)

        # Get the sources
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        # Format the response
        formatted_response = (f"{generated_response['result']}\n\n {create_sources_string(sources)}")

        # Add the prompt and response to the chat history
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answer_history.append(formatted_response)

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(st.session_state.chat_answer_history, st.session_state.user_prompt_history):
        message(user_query, is_user=True)
        message(generated_response, is_user=False)