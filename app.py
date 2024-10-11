import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Simple Q&A Chatbot with OpenAI'

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, openai_api_key, model, temperature, max_tokens):
    model = ChatOpenAI(model_name=model, temperature=temperature, max_tokens=max_tokens, openai_api_key=openai_api_key)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Simple Q&A Chatbot with OpenAI")

# Sidebar
st.sidebar.title("Settings")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Dropdown for OpenAI models
model = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Slider for temperature
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

# Slider for max tokens
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 100)

# Text Input for user question
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if openai_api_key:
        answer = generate_response(question, openai_api_key, model, temperature, max_tokens)
        st.write(answer)
    else:
        st.write("Please enter your OpenAI API Key")

