import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "QNACHATBOTWITHGROQ"

##prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Hey you are a helpful assistant please response to the user queries."),
        ("user","question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    os.environ["GROQ_API_KEY"] = api_key
    llm=ChatGroq(model="openai/gpt-oss-120B")
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


###streamlit
st.title("Settings")
api_key = st.sidebar.text_input(
    "Please Enter your Groq API key:",
    type="password"
)


llm=st.sidebar.selectbox("Select a model",["openai/gpt-oss-120B","Mixtral-8x7B","openai/gpt-oss-20B"])


temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

###question
st.write("go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
    
else:
    st.write("Please provide a query")    


