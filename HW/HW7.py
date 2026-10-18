import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import re
import chromadb
from openai import OpenAI

st.title("News Reporting Chatbot")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# - Load and clean the CSV --
@st.cache_data
def load_news():
    df = pd.read_csv("news.csv")
    df["Document"] = df["Document"].fillna("").astype(str)
    # same article listed under several clients -> one row
    df = (df.groupby("URL", as_index=False)
            .agg(companies=("company_name", lambda s: "; ".join(sorted(set(s)))),
                 Date=("Date", "first"),
                 Document=("Document", "first")))
    df["title"] = df["Document"].str.split(" Description:").str[0].str.strip()
    df["id"] = [f"doc{i}" for i in range(len(df))]
    return df

# --Build the vector DB once, then just load it ---
@st.cache_resource
def get_collection(_df):
    chroma = chromadb.PersistentClient(path="news_db")   # saved to a folder
    col = chroma.get_or_create_collection("news", metadata={"hnsw:space": "cosine"})
    if col.count() == 0:                                   # only the first time
        for i in range(0, len(_df), 100):
            batch = _df.iloc[i:i + 100]
            emb = client.embeddings.create(model="text-embedding-3-small",
                                           input=batch["Document"].str[:3000].tolist())
            col.add(ids=batch["id"].tolist(),
                    documents=batch["Document"].tolist(),
                    embeddings=[e.embedding for e in emb.data],
                    metadatas=batch[["companies", "Date", "URL", "title"]].to_dict("records"))
    return col

news_df = load_news()
collection = get_collection(news_df)
st.sidebar.caption(f"{len(pd.read_csv('news.csv'))} rows in CSV → {len(news_df)} unique articles · {collection.count()} in vector DB")


def embed(text):
    return client.embeddings.create(model="text-embedding-3-small",
                                    input=text).data[0].embedding

def search_news(query, company=None, n_results=8):
    """Find articles about a topic, optionally for one client company."""
    if company:
        rows = news_df[news_df["companies"].str.contains(company, case=False, regex=False)]
        if len(rows) > 0:
            got = collection.get(ids=rows["id"].tolist(), include=["embeddings"])
            emb = np.array(got["embeddings"])
            q = np.array(embed(query))
            sims = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q))
            order = np.argsort(-sims)[:n_results]
            ids = [got["ids"][i] for i in order]
            return news_df.set_index("id").loc[ids].reset_index().to_dict("records")

    res = collection.query(query_embeddings=[embed(query)], n_results=n_results)
    return news_df.set_index("id").loc[res["ids"][0]].reset_index().to_dict("records")

# --Interest score (law-firm view) --
CATEGORIES = {
    "litigation/regulatory": (3, [r"lawsuit", r"\bsue[sd]?\b", r"settle", r"class action", r"antitrust",
                                  r"\bsec\b", r"\bftc\b", r"\bdoj\b", r"\bcfpb\b", r"regulator", r"probe",
                                  r"investigat", r"\bfined\b", r"penalt", r"\bcourts?\b", r"judge",
                                  r"ruling", r"fraud", r"legal", r"patent"]),
    "deal/corporate":        (2, [r"merger", r"acqui", r"takeover", r"buyout", r"\bipo\b", r"stake",
                                  r"bankrupt", r"restructur", r"layoff", r"job cuts"]),
    "risk event":            (2, [r"breach", r"hack", r"cyber", r"outage", r"recall", r"scandal",
                                  r"strike", r"leak"]),
    "leadership":            (1.5, [r"\bceo\b", r"resign", r"steps? down", r"appoint", r"ousted"]),
    "market-moving":         (1, [r"earnings", r"revenue", r"profit", r"forecast", r"plunge",
                                  r"tumble", r"soar", r"downgrade"]),
}
NOISE = [r"otcmkts", r"1-year (low|high)", r"52-week", r"shares (sold|bought) by", r"position in",
         r"price target", r"review:", r"price cut", r"discount", r"short interest"]

def score_article(text):
    t = text.split(" content:")[0].lower()        # headline + description only
    score, reasons = 0.0, []
    for name, (weight, patterns) in CATEGORIES.items():
        if any(re.search(p, t) for p in patterns):
            score += weight
            reasons.append(name)
    if any(re.search(p, t) for p in NOISE):
        score -= 2
        reasons.append("routine stock/product item")
    return score, ", ".join(reasons) or "no strong signal"

scores = news_df["Document"].apply(score_article)
news_df["score"] = [s for s, _ in scores]
news_df["reasons"] = [r for _, r in scores]
dates = pd.to_datetime(news_df["Date"], utc=True, format="ISO8601")
news_df["score"] += (dates - dates.min()) / (dates.max() - dates.min())   # 0-1 recency boost
news_df["score"] += news_df["companies"].str.count(";") * 0.5             # several clients

def rank_interesting_news(top_n=10, company=None):
    """Top candidate stories by interest score (the LLM re-ranks them later)."""
    pool = news_df
    if company:
        pool = news_df[news_df["companies"].str.contains(company, case=False, regex=False)]
    top = pool.sort_values("score", ascending=False).head(top_n * 2)
    return top[["title", "companies", "Date", "URL", "score", "reasons"]].to_dict("records")

# = Step 5: Chatbot with function calling ==
import json

MODELS = {"gpt-4o-mini (low cost)": "gpt-4o-mini",
          "gpt-4o (high cost)": "gpt-4o"}

def short(a):
    """Keep tool results small: only what the LLM needs."""
    return {"title": a["title"], "companies": a["companies"], "date": str(a["Date"])[:10],
            "url": a["URL"], **({"summary": a["Document"].split(" content:")[0][:400]} if "Document" in a else {}),
            **({"score": round(float(a["score"]), 2), "signals": a["reasons"]} if "score" in a else {})}

def run_tool(name, args):
    if name == "search_news":
        return [short(a) for a in search_news(args.get("query", ""), args.get("company"),
                                              int(args.get("n_results", 8)))]
    if name == "rank_interesting_news":
        return [short(a) for a in rank_interesting_news(int(args.get("top_n", 10)), args.get("company"))]
    return {"error": "unknown tool"}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_news",
        "description": "Find articles about a company, topic, person or event. Use for 'find news about X'.",
        "parameters": {"type": "object", "properties": {
            "query":     {"type": "string", "description": "What to search for, e.g. 'antitrust lawsuit'"},
            "company":   {"type": "string", "description": "Client company name if the user named one"},
            "n_results": {"type": "integer", "description": "Number of articles (default 8)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "rank_interesting_news",
        "description": "Get candidate stories ranked by importance to a law firm. "
                       "Use for 'most interesting news', 'top stories'.",
        "parameters": {"type": "object", "properties": {
            "top_n":   {"type": "integer", "description": "How many stories (default 10)"},
            "company": {"type": "string", "description": "Optional client to restrict to"}}}}},
]

SYSTEM_PROMPT = """You are a news-monitoring assistant for a large global law firm.
Use ONLY the articles returned by your tools. Never use outside knowledge and never
invent articles, dates or URLs. Always call a tool before answering about the news.

"Interesting" to the firm, most important first:
1. litigation, regulatory action, investigations, antitrust, fines
2. M&A, deals, restructuring, bankruptcy, layoffs
3. risk events: fraud, cyber breaches, recalls, scandals
4. leadership changes
5. major market-moving financial news
Routine stock-price items, product reviews and shopping deals are NOT interesting.

For "most interesting news": call rank_interesting_news, then RE-RANK the candidates
yourself with the criteria above (the score is only a keyword heuristic - drop false
positives). Give a numbered list, most interesting first, each with:
**Title** - client(s), date
Why it matters: 1-2 sentences of legal/business context for the firm.
Source: URL

For "news about X": call search_news and list only articles truly about X, newest
first, each with a one-line summary, date and URL. If nothing relevant, say so."""

def chat(messages, model):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[-8:]   # short memory
    for _ in range(4):                                  # allow a few tool rounds
        resp = client.chat.completions.create(model=model, messages=msgs,
                                              tools=TOOLS, temperature=0.2)
        msg = resp.choices[0].message
        if not msg.tool_calls:                          # final answer
            return msg.content
        msgs.append(msg)
        for call in msg.tool_calls:                     # run each tool the LLM asked for
            result = run_tool(call.function.name, json.loads(call.function.arguments or "{}"))
            msgs.append({"role": "tool", "tool_call_id": call.id,
                         "content": json.dumps(result, default=str)})
    return client.chat.completions.create(model=model, messages=msgs).choices[0].message.content

#
with st.sidebar:
    model = MODELS[st.radio("Model", list(MODELS))]
    st.markdown("Try:\n- Find the most interesting news\n- Find news about Nvidia\n"
                "- Any regulatory issues for big tech?")
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"].replace("$", "\\$"))   # stop $ being read as math

if prompt := st.chat_input("Ask about the news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model}..."):
            answer = chat(st.session_state.messages, model)
        st.markdown(answer.replace("$", "\\$"))         # stop $ being read as math
    st.session_state.messages.append({"role": "assistant", "content": answer})
