# main.py
import streamlit as st
import os
from groq_translation import groq_translate
from gtts import gTTS
from gtts.lang import tts_langs
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

from dotenv import load_dotenv

load_dotenv()

# Configure Google API for audio summarization
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(translation,prompt):
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt+translation.text,
    )
    return response.text

prompt = """
You are a highly meticulous and intelligent YouTube video note-taker assistant. You will be given the full transcript of a YouTube video — it can be any type: lecture, podcast, tutorial, explainer, interview, motivational talk, documentary, etc.

Your task is to create **exhaustive, structured notes** from the transcript, ensuring that **every single detail — no matter how small — is captured**, also make ensure its strictly in its original language(The text's original language).

Here’s what you must do:

1. **Capture ALL key points and sub-points** in the exact order they appear (chronological flow). Do NOT skip or merge anything.
2. **Extract every useful detail** — including:
   - Examples
   - Analogies
   - Case studies
   - Statistics or data points
   - Quotes (preserve exact wording if impactful)
   - Steps in a process
   - Actionable advice
   - Definitions of terms
   - Tools, methods, or frameworks mentioned
   - Minor but relevant observations or tips
   - Speaker’s personal opinions or reflective comments
   - Any interjections or humorous asides that add value or tone
3. **Organize the output into well-structured notes**, with:
   - Clear **section headings** for topic shifts
   - **Bullet points** for ideas and sub-ideas
   - **Numbered lists** for ordered steps or sequences
   - **Bold formatting** for key terms, tools, names, or technical concepts
   - **Indented sub-bullets** for additional layers of detail
4. These notes must be **highly detailed** — no generalizations, shortening, or abstraction. Do NOT leave out any valuable content.
5. Maintain the speaker’s **intent, tone, personality, and content richness**.
6. The output should be:
   - Clean, easy to read, and logically organized
   - Structured like a **comprehensive classroom notebook**
   - Suitable for **study, revision, or referencing**
   - Fully self-contained — the reader should understand the entire video from the notes

Here is the transcript to work with:
"""


# Set page config
st.set_page_config(page_title='YouTube Transcript Translator', page_icon='🎤')

# Set page title
st.title('YouTube Transcript Translator')

# Text to speech
def text_to_speech(translated_text, language):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text, lang=language)
    my_obj.save(file_name)
    return file_name

# Getting the transcript data from YouTube videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        
        # List of common ISO 639-1 language codes (can be extended further)
        all_languages = [
            'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'zh', 'zh-Hans', 'zh-Hant',
            'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha',
            'haw', 'he', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'rw', 'ko', 'ku',
            'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ny',
            'or', 'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so',
            'es', 'su', 'sw', 'sv', 'tl', 'tg', 'ta', 'tt', 'te', 'th', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy',
            'xh', 'yi', 'yo', 'zu'
        ]
        
        # Fetch the transcript in any available language from the list
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=all_languages)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

# Get supported languages and create a mapping of language names to codes
supported_langs = tts_langs()
# Create a sorted list of (code, name) tuples
sorted_langs = sorted(supported_langs.items(), key=lambda x: x[1])
# Create a dictionary for display names to codes
lang_display_to_code = {name: code for code, name in sorted_langs}

# Language selection
selected_lang_name = st.selectbox(
    "Language to translate to:",
    options=list(lang_display_to_code.keys()),
    index=None,
    placeholder="Select language...",
)
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Translate"):
    with st.spinner("Translating"):
        transcript_text = extract_transcript_details(youtube_link)
        lang_code = lang_display_to_code[selected_lang_name]
        translation = groq_translate(transcript_text,lang_code)
    st.subheader('Translated Transcript to ' + selected_lang_name)
    st.write(translation.text)
    st.divider()
    st.subheader(f"Transcript Notes in {selected_lang_name}:")
    transcript_notes = get_gemini_response(translation,prompt)
    st.write(transcript_notes)

    with st.spinner("Generating Audio"):
        audio_file = text_to_speech(translation.text, lang_code)
        st.audio(audio_file, format="audio/mp3")