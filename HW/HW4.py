import streamlit as st
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3 not installed locally — fine, system sqlite3 works

from openai import OpenAI
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup

st.title("Syracuse Student Organization Chatbot)")


# Create the OpenAI client once and store it in session_state

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["My_newkey"])
client = st.session_state.client

# Model selector mini vs regular
openai_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openai_model == "mini" else "gpt-4o"

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def extract_text_from_html(html_path):
    """Read an HTML file and return its visible text, stripped of tags/scripts."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script", "style"]):
          tag.decompose()
 
    text = soup.get_text(separator="\n")
    # Collapse excess blank lines left over from stripped tags
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
def chunk_text(text, n_chunks=2):
    """
    Chunking method used: FIXED-SIZE CHUNKING (by character count), splitting
    each document into exactly `n_chunks` roughly equal pieces.
 
    Why this method: the assignment requires two mini-documents per HTML
    page, so a simple, predictable split is all that's needed here rather
    than a more complex semantic/paragraph-aware splitter. Fixed-size
    chunking is the simplest approach to implement, produces evenly-sized
    chunks (useful since our chunk count is fixed at 2 per doc), and is
    good enough for these relatively short, single-topic student-org pages
    where splitting mid-paragraph is unlikely to separate unrelated ideas.
    The trade-off (acceptable here) is that a chunk boundary could fall in
    the middle of a sentence, since we're not searching for a natural
    semantic break point.
    """
    text = text.strip()
    if not text:
        return [""] * n_chunks
 
    chunk_size = max(1, len(text) // n_chunks)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        # last chunk takes any remainder so nothing gets truncated
        end = len(text) if i == n_chunks - 1 else start + chunk_size
        chunks.append(text[start:end])
    return chunks
 
 
def add_to_collection(collection, text, file_name):
    """Chunk the document into 2 mini-documents and store each separately,
    with metadata linking it back to its source file and chunk number."""
    chunks = chunk_text(text, n_chunks=2)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        chunk_id = f"{file_name}::chunk{i + 1}"
        collection.add(
            documents=[chunk],
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[{"source_file": file_name, "chunk_number": i + 1}],
        ) 
 
def load_html_to_collection(folder_path, collection):
    for html_file in sorted(Path(folder_path).glob("*.html")):
        text = extract_text_from_html(str(html_file))
        add_to_collection(collection, text, html_file.name)
 
def get_relevant_context(collection, query_text, n_results=3):
    """Embed the query, search the collection, and return the combined
    text of the top matches plus their source filenames (for transparency)."""
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    context_text = "\n\n".join(
        f"[Source: {meta['source_file']}, chunk {meta['chunk_number']}]\n{doc}"
        for doc, meta in zip(docs, metadatas)
    )
    source_labels = [
        f"{meta['source_file']} (chunk {meta['chunk_number']})" for meta in metadatas
    ]
    return context_text, source_labels
 

# Part 2: Build (or reuse) the ChromaDB vector database from HTML files.
# collection.count() == 0 means we only embed/populate once — on later runs
# the persistent DB already has data, so this block is skipped entirely.

if "HW4_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_HW4")
    collection = chroma_client.get_or_create_collection(name="HW4Collection")
    if collection.count() == 0:
        load_html_to_collection("./HW-04-Data/", collection)
    st.session_state.HW4_VectorDB = collection
else:
    collection = st.session_state.HW4_VectorDB
 
# Part 3-4: The RAG chatbot with a 5-interaction memory buffer

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about student "
    "organizations at the iSchool, using the context provided below, which "
    "was retrieved from a vector database of the organizations' web pages.\n\n"
    "IMPORTANT: If you use information from the provided context to answer, "
    "explicitly say so at the start of your answer, e.g. 'Based on the "
    "student organization pages I found...'. If the context doesn't contain "
    "relevant information, say so and answer from general knowledge "
    "instead, making clear you are not using the retrieved data.\n\n"
    "Context from student organization pages:\n{context}"
)
 
# An "interaction" = one user message + one assistant reply.
# We keep the last 5 interactions => at most 10 messages in memory.
MAX_INTERACTIONS = 5
MAX_MESSAGES = MAX_INTERACTIONS * 2
 
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask me about iSchool student organizations!"}
    ]
 
# Display existing chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])
 
# React to new user input
if prompt := st.chat_input("Ask about a student organization..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # --- RAG step: retrieve relevant context for this prompt ---
    context_text, source_labels = get_relevant_context(collection, prompt, n_results=3)
    system_prompt = {
        "role": "system",
        "content": BASE_SYSTEM_PROMPT.format(context=context_text),
    }
 
    # Trim conversation memory to the last MAX_INTERACTIONS exchanges
    # (this happens BEFORE sending to the LLM, and we also trim the stored
    # history itself so it never grows past the 5-interaction limit)
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
 
    messages_to_send = [system_prompt] + st.session_state.messages
 
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True,
    )
 
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
        with st.expander("Sources used for this answer"):
            st.write(", ".join(source_labels))
 
    st.session_state.messages.append({"role": "assistant", "content": response})
 
    # Trim again after adding the assistant's reply, to enforce the cap
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
 