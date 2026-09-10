
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup


@st.cache_data(show_spinner=False)
def read_url_content(url):
    """Fetch a URL and return its visible text. Cached so reruns don't refetch."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


st.title("Welcome to URL Discussion Chatbot")

url = st.text_input("Enter URL for discussion")
url2 = st.text_input("Enter a second URL (optional)")

# ---------- Sidebar controls ----------
choose_llm = st.sidebar.selectbox("Choose LLM provider", ("OpenAI", "Claude"))
use_advance = st.sidebar.checkbox("Use advanced model")

# Verify these IDs against each provider's current model list before submitting.
MODELS = {
    "OpenAI": {False: "gpt-5.1-mini", True: "gpt-5.1"},
    "Claude": {False: "claude-haiku-4-5-20251001", True: "claude-opus-5"},
}
model_choice = MODELS[choose_llm][use_advance]

# ---------- System prompt built from URL(s), never discarded ----------
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

    if content_parts:
        combined_content = "\n\n".join(content_parts)
        st.session_state.system_prompt = (
            "You are a helpful assistant answering questions about the following "
            "webpage content. Use it as your primary source of truth:\n\n"
            f"{combined_content}"
        )

# ---------- Buffer of 6 messages (3 user/assistant exchanges) ----------
MAX_MESSAGES = 6

if "messages" not in st.session_state:
    st.session_state.messages = []  # user/assistant turns only, never the system prompt


def trim_history():
    """Keep the last 6 turns, and never start on an assistant turn (Claude rejects that)."""
    msgs = st.session_state.messages[-MAX_MESSAGES:]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    st.session_state.messages = msgs


# Display past conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Ask a question about the URL(s)")

if prompt:
    if not st.session_state.system_prompt:
        st.warning("Please enter at least one URL above first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        trim_history()

        with st.chat_message("assistant"):
            if choose_llm == "OpenAI":
                client = OpenAI(api_key=st.secrets["openai_api_key"])
                stream = client.chat.completions.create(
                    model=model_choice,
                    messages=(
                        [{"role": "system", "content": st.session_state.system_prompt}]
                        + st.session_state.messages
                    ),
                    stream=True,
                )
                response = st.write_stream(stream)

            else:  # Claude
                client = Anthropic(api_key=st.secrets["claude_api_key"])
                # Claude takes "system" as a top-level param, not inside messages
                with client.messages.stream(
                    model=model_choice,
                    max_tokens=1024,
                    system=st.session_state.system_prompt,
                    messages=st.session_state.messages,
                ) as stream:
                    response = st.write_stream(stream.text_stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
