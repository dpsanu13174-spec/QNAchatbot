from urllib.parse import urlparse

import streamlit as st
from groq import APIStatusError, AuthenticationError
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROMPT = PromptTemplate.from_template(
    """
You are an expert summarization assistant.
Summarize the following content clearly in 1-2 concise paragraphs.
Ignore ads/navigation and focus only on meaningful information.

Content:
{text}
"""
)


def normalize_key(value: str) -> str:
    return (value or "").strip().strip('"').strip("'")


def is_youtube_url(value: str) -> bool:
    host = urlparse(value).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def load_documents(url: str):
    if is_youtube_url(url):
        try:
            return YoutubeLoader.from_youtube_url(
                url, add_video_info=False, language=["en", "hi"]
            ).load()
        except Exception as err:
            st.warning(f"YouTube transcript failed ({err}). Falling back to page extraction.")
    return WebBaseLoader(url).load()


st.set_page_config(page_title="URL / YouTube Summarizer", page_icon="📝", layout="wide")
st.title("Text Summarization with LangChain + Groq")

st.sidebar.header("Configuration")
api_key = normalize_key(st.sidebar.text_input("Groq API Key", type="password"))
model = st.sidebar.selectbox(
    "Groq Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

url = st.text_input(
    "Enter YouTube or Website URL",
    placeholder="https://www.youtube.com/watch?v=... or https://example.com/article",
)

if st.button("Summarize"):
    if not api_key:
        st.error("Please enter your Groq API key.")
        st.stop()
    if not api_key.startswith("gsk_"):
        st.error("Invalid Groq API key format. It should start with 'gsk_'.")
        st.stop()
    if not url.startswith(("http://", "https://")):
        st.error("Please enter a valid URL starting with http:// or https://")
        st.stop()

    try:
        with st.spinner("Loading content..."):
            docs = load_documents(url)
        if not docs:
            st.error("No content could be loaded from this URL.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
        text = "\n\n".join(d.page_content for d in splitter.split_documents(docs))

        llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        chain = PROMPT | llm | StrOutputParser()

        with st.spinner("Generating summary..."):
            summary = chain.invoke({"text": text})

        st.subheader("Summary")
        st.write(summary)

    except AuthenticationError:
        st.error("Groq authentication failed (401). Please use a valid active API key.")
    except APIStatusError as err:
        if getattr(err, "status_code", None) == 401:
            st.error("Groq authentication failed (401). Please regenerate your API key.")
        else:
            st.exception(err)
    except Exception as err:
        st.exception(err)
