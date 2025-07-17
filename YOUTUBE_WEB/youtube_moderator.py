import streamlit as st
from dotenv import load_dotenv
load_dotenv() ##load all the nevironment variables
import os
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))

prompt="""You are Yotube video content moderator. You will be taking the transcript text
and moderating the entire video content as per youtube content policies and providing the content moderation recommendation
or the places where the content in video is violating the policies in points
within 250 words. Please provide the same for the text given here:  """


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

st.title("YouTube Content Moderator")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get AI Recommendation"):
    transcript_text=extract_transcript_details(youtube_link)

    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("## Moderation Recommendation:")
        st.write(summary)