import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

st.title("Welcome to URL dicussion Chatbot")

st.write("Enter URL here")
url = st.text_input("Enter URL for discussion")
url2 = st.text_input("Enter the second URL it is optional")


use_advance = st.sidebar.checkbox("use advance model")
choose_llm = st.sidebar.selectbox("Choose LLM provider"("OpenAI", "Claude"))


# Model chooice

model_choice = "gpt-5.1" if choose_llm == 'OpenAI' else 'Claude-opus-5'

#system prompt built from URL(s), never discarded

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = None

if url or url2:
    content_parts = []
if url:
        content1 = read_url_content(url)
if content1:
            content_parts.append(f"Content from {url}:\n{content1}")
if url2:
        content2 = read_url_content(url2)
if content2:
            content_parts.append(f"Content from {url2}:\n{content2}")

combined_content = "\n\n".join(content_parts)
st.session_state.system_prompt = (
        "You are a helpful assistant answering questions about the following "
        f"webpage content. Use it as your primary source of truth:\n\n{combined_content}"
    )


# ---------- Requirement 5: buffer of 6 messages (3 user-assistant exchanges) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # holds ONLY user/assistant turns, never the system prompt

# Display past conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Ask a question about the URL(s)")

if prompt:
    if not st.session_state.system_prompt:
        st.warning("Please enter at least one URL in the sidebar first.")
    else:
        # Add user's message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Keep only the last 6 messages (3 exchanges) — system prompt is NOT part of this list
        st.session_state.messages = st.session_state.messages[-6:]

        # Rebuild the full message list fresh on every call
        full_messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ] + st.session_state.messages

        with st.chat_message("assistant"):
            if choose_llm == "OpenAI":
                client = OpenAI(api_key=st.secrets["openai_api_key"])
                stream = client.chat.completions.create(
                    model=model_choice,
                    messages=full_messages,
                    stream=True,
                )
                response = st.write_stream(stream)

            elif choose_llm == "Claude":
                client = Anthropic(
                    api_key=st.secrets["claude_api_key"],
                    default_headers={"anthropic-workspace-id": st.secrets["claude_workspace_id"]},
                )
                # Claude takes "system" as its own top-level param, not inside messages
                claude_messages = [m for m in full_messages if m["role"] != "system"]
                with client.messages.stream(
                    model=model_choice,
                    max_tokens=1024,
                    system=st.session_state.system_prompt,
                    messages=claude_messages,
                ) as stream:
                    response = st.write_stream(stream.text_stream)

        # Save assistant's reply to history too
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages = st.session_state.messages[-6:]