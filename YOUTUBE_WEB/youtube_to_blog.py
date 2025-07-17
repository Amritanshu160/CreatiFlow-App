import streamlit as st
from dotenv import load_dotenv
load_dotenv() ##load all the nevironment variables
import os
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))

prompt = """You are a professional blog writer. Your task is to convert the following YouTube video transcript into a well-structured, engaging, and informative blog post. The blog should accurately reflect the content of the video while enhancing it with a natural writing style, clear headings, smooth transitions, and paragraph formatting.

Please follow these guidelines:
- Retain all key information, insights, and examples from the video.
- Use a conversational yet informative tone.
- Break the content into logical sections with appropriate subheadings (like Introduction, Key Points, Examples, Conclusion, etc.).
- Avoid any redundant timestamps, filler words, or speech disfluencies.
- Ensure the blog is SEO-friendly, uses relevant keywords naturally, and is suitable for readers unfamiliar with the video format.
- The output should be comprehensive and capture the full essence of the video within 700–1000 words if possible.

Here is the transcript of the video you need to convert into a blog post:
"""

## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]

        all_languages = [
            'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'zh', 'zh-Hans', 'zh-Hant',
            'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha',
            'haw', 'he', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'rw', 'ko', 'ku',
            'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ny',
            'or', 'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so',
            'es', 'su', 'sw', 'sv', 'tl', 'tg', 'ta', 'tt', 'te', 'th', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy',
            'xh', 'yi', 'yo', 'zu'
        ]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id,languages=all_languages)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    
## getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text,prompt):
    response = client.models.generate_content(
        model= "gemini-2.5-flash",
        contents = prompt + transcript_text
    )
    return response.text

st.title("YouTube Videos To Blog Generator")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Convert To Blog"):
    transcript_text=extract_transcript_details(youtube_link)

    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("Blog:")
        st.write(summary)