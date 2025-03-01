import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From Website", page_icon="🌐")
st.title("🌐 LangChain: Summarize Text From Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

generic_url = st.text_input("URL", label_visibility="collapsed")

# Groq LLM Configuration
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])



# Summarization Button
if st.button("Summarize the Content from Website"):
    # Validate all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It should be a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load website data
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                docs = loader.load()

                # Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")