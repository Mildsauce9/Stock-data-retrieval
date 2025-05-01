import os
import re
from datetime import datetime, timedelta

import dateparser
import pinecone
from openai import OpenAI
import openai
import google.generativeai as genai
from dotenv import load_dotenv

# ─── Load environment ─────────────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY        = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT    = os.getenv("PINECONE_ENVIRONMENT")
NEWS_INDEX              = os.getenv("PINECONE_NEWS_INDEX_NAME")
TICKER_INDEX            = os.getenv("PINECONE_TICKER_INDEX_NAME")
OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
# ─── Initialize Pinecone, OpenAI, Gemini ──────────────────────────────────────
from pinecone import Pinecone

# Initialize the Pinecone client (using your API key; environment is auto-detected)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to your existing indexes
news_index   = pc.Index(NEWS_INDEX)
ticker_index = pc.Index(TICKER_INDEX)
client = OpenAI()

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def extract_date_range_and_clean_query(prompt: str):
    """
    Finds substrings like:
      • "between X and Y"
      • "from X to Y" (or X - Y)
      • "in <period>"  (e.g. "in July 2020" or "in the last week")
    Returns: (start_iso, end_iso, cleaned_prompt). If no dates found,
             start_iso/end_iso = None and cleaned_prompt == prompt.
    """
    patterns = [
        r'between\s+(?P<start>.*?)\s+and\s+(?P<end>.*?)(?=$|\?)',
        r'from\s+(?P<start>.*?)\s+(?:to|-)\s+(?P<end>.*?)(?=$|\?)',
        r'\bin\s+(?P<period>.*?)(?=$|\?)',
    ]
    for pat in patterns:
        m = re.search(pat, prompt, flags=re.IGNORECASE)
        if not m:
            continue
        cleaned = (prompt[:m.start()] + prompt[m.end():]).strip(" ,?")
        if m.groupdict().get("start"):
            # explicit start/end
            s, e = m.group("start"), m.group("end")
            start_dt = dateparser.parse(s, settings={'PREFER_DAY_OF_MONTH':'first'})
            end_dt   = dateparser.parse(e, settings={'PREFER_DAY_OF_MONTH':'last'})
        else:
            # single period like "in July 2020" or "in the last week"
            period = m.group("period").strip()
            dt = dateparser.parse(period, settings={'PREFER_DAY_OF_MONTH':'first'})
            if re.match(r'^[A-Za-z]+\s+\d{4}$', period):
                # whole month
                start_dt = dt.replace(day=1)
                nxt = (start_dt.replace(day=28) + timedelta(days=4)).replace(day=1)
                end_dt   = nxt - timedelta(days=1)
            elif "last week" in period.lower():
                today = datetime.today()
                start_dt = (today - timedelta(days=today.weekday()+7)).replace(hour=0, minute=0)
                end_dt   = start_dt + timedelta(days=6, hours=23, minutes=59)
            elif "last month" in period.lower():
                today = datetime.today()
                first_this = today.replace(day=1)
                end_dt    = first_this - timedelta(seconds=1)
                start_dt  = end_dt.replace(day=1, hour=0, minute=0)
            else:
                start_dt = dt
                end_dt   = dt
        return start_dt.date().isoformat(), end_dt.date().isoformat(), cleaned
    return None, None, prompt

def get_embedding(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
        encoding_format="float"
    )
    return resp.data[0].embedding

def query_pinecone(index, vector, start_date=None, end_date=None, top_k=5):
    filt = {}
    if start_date and end_date:
        # convert ISO date-strings into numeric timestamps for Pinecone
        sd = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        ed = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        filt = {
            "date": {
                "$gte": int(sd),
                "$lte": int(ed)
            }
        }
    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filt
    )
    return res["matches"]

def process_user_query(prompt: str) -> str:
    # 1) parse date + clean question
    start, end, clean_q = extract_date_range_and_clean_query(prompt)
    print(start, end, clean_q)
    # 2) embed & fetch from both indexes
    emb       = get_embedding(clean_q)
    news_hits = query_pinecone(news_index, emb, start, end)
    tick_hits = query_pinecone(ticker_index, emb, start, end)

    print(news_hits)
    print(tick_hits)
    # 3) build contexts
    news_ctx   = "\n".join(f"{h['metadata']['date']}: sector : {h['metadata']['sectors']}, feeling very {h['metadata']['sentiment']} about this, {h['metadata'].get('summary','')} they are talking about {h['metadata']['tickers']}" for h in news_hits)
    ticker_ctx = "\n".join(f"{h['metadata']['date']}: {h['metadata']['text']}"     for h in tick_hits)

    user_msg = (
        f"""
        SYSTEM INSTRUCTIONS
        You are a specialized financial analysis system with access to historical news events and stock ticker data. Your task is to provide insightful, accurate, and well-reasoned analyses of market events by connecting news coverage with price movements. Maintain a professional, balanced tone appropriate for financial analysis.
        CONTEXT
        The user has requested an analysis regarding: {clean_q}
        Time Period: from {start} to {end}
        DATA SOURCES
        News Articles
        {news_ctx}
        Market Data
        {ticker_ctx}
        """
    )
    print(user_msg)
    # run our prompt through Google Generative AI
    llm = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = llm.generate_content(
        user_msg
    )
    return response.text
    return None
