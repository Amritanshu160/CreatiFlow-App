import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure Google Generative AI
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
    response = client.models.generate_content(
        model= "gemini-2.5-flash",
        contents= prompt_parts
    )
    st.write(response.text)    
