import streamlit as st
from google import genai
from youtube_search import YoutubeSearch
import nltk
from collections import Counter
import re
from nltk.corpus import stopwords
from googleapiclient.discovery import build

YOUTUBE_API_KEY = "YOUTUBE_API_KEY"


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Configure Gemini
GEMINI_API_KEY = "GOOGLE_API_KEY"
client = genai.Client(api_key=GEMINI_API_KEY)

# Utils

def extract_keywords(text):
    # Tokenize the text
    words = nltk.word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # POS tagging to filter for nouns and proper nouns
    tagged_words = nltk.pos_tag(words)
    filtered_words = [word for word, tag in tagged_words if tag in ['NN', 'NNS', 'NNP', 'NNPS']]  # Nouns and proper nouns
    
    # Return the most common words
    return Counter(filtered_words).most_common(15)

from googleapiclient.discovery import build

def fetch_trending_videos(topic):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    # Step 1: Search by relevance
    search_response = youtube.search().list(
        q=topic,
        type='video',
        part='id,snippet',
        maxResults=10,
        order='relevance'
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]

    # Step 2: Get statistics for those videos
    stats_response = youtube.videos().list(
        part='statistics,snippet',
        id=",".join(video_ids)
    ).execute()

    videos = []
    for item in stats_response['items']:
        videos.append({
            'title': item['snippet']['title'],
            'long_desc': item['snippet'].get('description', ''),
            'video_id': item['id'],
            'view_count': int(item['statistics'].get('viewCount', 0)),
            'url_suffix': f"/watch?v={item['id']}"
        })

    # Step 3: Sort manually by view count
    videos.sort(key=lambda x: x['view_count'], reverse=True)

    return videos


def get_combined_text(videos):
    combined = "\n".join([v['title'] + ". " + (v.get('long_desc', '') or '') for v in videos if 'title' in v])
    return combined

def generate_youtube_script(topic, keywords):
    prompt = f"""
    You are a YouTube content strategist.
    Given the topic "{topic}" and the trending keywords {keywords},
    generate:
    - 3 catchy video title ideas
    - A complete YouTube video script
    - Hashtag suggestions
    - A video structure breakdown (intro, hook, main, CTA, outro)
    """
    response = client.models.generate_content(
        model= "gemini-2.5-flash",
        contents= prompt
    )
    return response.text

# Streamlit UI
st.set_page_config(page_title="YouTube Trend AI", layout="wide")
st.title("🚀 YouTube Trend Research AI")
st.markdown("Find the trendiest YouTube topics + get content strategy from Gemini!")

query = st.text_input("Enter a topic/niche:", "AI tools")

if st.button("Search"):
    with st.spinner("Fetching trending videos..."):
        videos = fetch_trending_videos(query)

    if not videos:
        st.error("No trending videos found for this topic.")
    else:
        combined_text = get_combined_text(videos)
        keywords = extract_keywords(combined_text)
        keywords_list = [kw for kw, _ in keywords]

        st.subheader("🔥 Top Keywords")
        st.write(", ".join(keywords_list))

        with st.spinner("Generating content strategy via Gemini..."):
            script = generate_youtube_script(query, keywords_list)

        st.subheader("📽️ Gemini Content Strategy")
        st.markdown(script)

        st.subheader("📊 Trending Videos")
        for video in videos:
            title = video.get('title', 'No title')
            desc = video.get('long_desc', '')
            url_suffix = video.get('url_suffix', '')
            views = video.get('views', 'N/A')
            duration = video.get('duration', 'N/A')
            publish_time = video.get('publish_time', 'N/A')
            channel = video.get('channel', 'Unknown')

            # Extract video ID for thumbnail
            if "watch?v=" in url_suffix:
                video_id = url_suffix.split("watch?v=")[-1]
                thumbnail_url = f"http://img.youtube.com/vi/{video_id}/0.jpg"
                st.image(thumbnail_url, use_column_width=True)

            st.markdown(f"### 🎬 {title}")
            st.markdown(f"**Channel**: {channel}  \n**Published**: {publish_time}  \n**Views**: {views}  \n**Duration**: {duration}")
            st.write(desc)
            st.markdown(f"[▶️ Watch on YouTube](https://www.youtube.com{url_suffix})")
            st.markdown("---")

