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


# Show title and description.
st.title("Url Summarisation ")

st.write(
    " Please enter your Url! " 
)
url = st.text_input("Enter a Url to summarise")

summary_type = st.sidebar.selectbox(
    'Summary Type',
    ('Summarize in 100 words',
     'Summarize in 2 connecting paragraphs',
     'Summarize in 5 bullet points'
)
)
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

use_advanced = st.sidebar.checkbox("Use advanced model")
Choose_llm = st.sidebar.selectbox("Choose LLM Provider", ("OpenAI", "Claude", "Gemini"))

# - API keys -
openai_api_key = st.secrets["My_newkey"]
claude_api_key = st.secrets["My_newclaudekey"]


language = st.sidebar.selectbox('Output Language', ('English', 'French', 'Chinese'))

# - Set up client + model based on selection -

if Choose_llm == "OpenAI":
    client = OpenAI(api_key=openai_api_key)
    model_choice = "gpt-5" if use_advanced else "gpt-5-mini"
else:  # Claude
    client = Anthropic(api_key=claude_api_key)
    model_choice = "claude-opus-4-1" if use_advanced else "claude-haiku-4-5"

# Main logic -
if url and summary_type and language:
    document = read_url_content(url)

    if document is None:
        st.error("Could not read content from that URL. Please check the link and try again.")
    else:
        prompt = f"Here's the content from a webpage: {document}\n\n---\n\n{summary_type}. Please respond in {language}."

        if Choose_llm == "OpenAI":
            stream = client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            st.write_stream(stream)

        elif Choose_llm == "Claude":
            with client.messages.stream(
                model=model_choice,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                st.write_stream(stream.text_stream)

        else:  # Gemini
            response = client.models.generate_content(
                model=model_choice,
                contents=prompt,
            )
            st.write(response.text)