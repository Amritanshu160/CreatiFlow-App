import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config
)

# Page Configuration
st.set_page_config(layout="wide", page_title="BlogCraft", page_icon="✍️")

# App Title
st.title("✍️🤖 BlogCraft: Your AI Writing Companion")
st.subheader("Now you can craft perfect blogs with the help of AI - BlogCraft is your new AI blog companion.")

# Sidebar for inputs
with st.sidebar:
    st.title("Input Your Blog Details")
    st.subheader("Enter Details of the Blog You Want to Generate")

    # User inputs
    blog_title = st.text_input("Blog Title", placeholder="Enter your blog title here")
    keywords = st.text_area("Keywords (comma-separated)")
    num_words = st.slider("Number of words", min_value=250, max_value=2500, step=50, value=500)

    prompt_parts = [
        f"Generate a comprehensive,engaging blog post relevant to the given title \"{blog_title}\" and keywords \"{keywords}\".Make ensure to incorporate these keywords in the blog post.The blog should be approximately {num_words} words."
    ]

    # Submit button
    submit_button = st.button("Generate Blog")

if submit_button:
    response = model.generate_content(prompt_parts)
    st.write(response.text)    
