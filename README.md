URL & YouTube Content Summarizer

An LLM-powered summarization tool that generates concise summaries from webpages and YouTube videos using LangChain and Groq models.
The application extracts content from a given URL, processes the text, and produces a clear summary through a large language model.

Built with Python, Streamlit, LangChain, and Groq API.

Features

Summarize web articles from any valid URL

Summarize YouTube videos using transcript extraction

Clean and preprocess webpage content automatically

Handle long documents using text chunking

Fast LLM inference using Groq models

Simple and interactive Streamlit UI

Tech Stack

Python

LangChain

Groq API

Streamlit

BeautifulSoup / WebBaseLoader

YouTube Transcript Loader

RecursiveCharacterTextSplitter

Project Architecture
User Input (URL)
        │
        ▼
Content Loader
(WebBaseLoader / YouTubeLoader)
        │
        ▼
Text Processing
(RecursiveCharacterTextSplitter)
        │
        ▼
LLM Summarization
(Groq + LangChain)
        │
        ▼
Streamlit Interface
(Display Summary)
Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/url-summarizer.git
cd url-summarizer
2. Install dependencies
pip install -r requirements.txt
3. Run the application
streamlit run app.py
Usage

Open the Streamlit app.

Enter your Groq API Key in the sidebar.

Paste a website URL or YouTube link.

Click Summarize.

The AI model will generate a concise summary of the content.

Example Inputs

Website:

https://example.com/article

YouTube:

https://www.youtube.com/watch?v=video_id
Example Output

The system generates a 1–2 paragraph summary highlighting the key information from the webpage or video transcript.

Future Improvements

Multi-language summarization

Bullet-point summaries

PDF and document summarization

Browser extension support

Summary export options
