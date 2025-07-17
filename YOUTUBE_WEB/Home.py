import streamlit as st

# Set page config once at the beginning of the script
st.set_page_config(page_title="CreatiFlow", layout="wide",page_icon="🎨")


def front():
    import streamlit as st

    # Title and description
    st.title("🎨 CreatiFlow - Your Creative Companion")
    st.write("Welcome to **CreatiFlow**! Unleash your creativity with these powerful tools designed for content creators, marketers, and digital enthusiasts.")

    with st.expander("📺 YouTube Tools"):
        st.markdown("""
        1. **Chat With YouTube Videos**: Chat with any YouTube video using its transcript and Gemini RAG.  
        2. **YouTube Content Moderator**: Analyze YouTube video transcripts for content moderation and policy violations.  
        3. **YouTube Video Summarizer**: Get detailed notes and summaries from YouTube video transcripts.  
        4. **YouTube Trend Analyzer**: Discover trending YouTube videos on any topic and generate content strategy using Gemini AI.
        5. **YouTube Translator**: Translate YouTube videos into multiple languages.              
        6. **YouTube To Blog Generator**: Generate blogs from YouTube videos.  
        """)

    with st.expander("🎨 Creative Tools"):
        st.markdown("""
        1. **Thumbnail Generator**: Create stunning YouTube thumbnails with customizable colors, text, and images.  
        2. **Video to GIF Converter**: Convert your videos into high-quality GIFs with adjustable speed and resolution.  
        3. **Image Transformation Toolset**: Generate/Edit images and also apply various transformations on the images.  
        4. **Avatar Generator**: Design unique and personalized avatars for your projects or profiles.  
        """)

    with st.expander("🌐 Website Tools"):
        st.markdown("""
        1. **Blog Generator**: Craft engaging and SEO-friendly blog posts with AI assistance.  
        2. **Website Summarizer**: Summarize any website content into concise, easy-to-read points.
        3. **Website Chat**: Chat with any website's content in real-time using RAG and LLMs, with full chat history support.              
        """)

    # Footer
    st.write("---")
    st.write("Made with ❤️ by Amritanshu Bhardwaj")
    st.write("© 2025 CreatiFlow. All rights reserved.")

# Define the functions for each app
def thumbnail_generator():
    import numpy as np
    import random
    import sys
    from main import main
    from PIL import Image, ImageColor, ImageFont, ImageDraw
    from PIL.Image import Resampling
    from rembg import remove
    from io import BytesIO

    # Page title
    pagetitle = '🏞️ Thumbnail Image Generator'
    st.title(pagetitle)
    st.info('This app allows you to create a thumbnail image for a YouTube video.')

    img_path = 'renders'

    # Initialize session state
    if 'color1' not in st.session_state:
        st.session_state.color1 = '#06D0DE'
    if 'color2' not in st.session_state:
        st.session_state.color2 = '#FE31CD'

    # Generate a random HEX color
    def generate_random_hex_color():
        # Color 1
        hex1 = '%06x' % random.randint(0, 0xFFFFFF)
        hex1 = '#' + hex1
        rgb_color_1 = ImageColor.getcolor(hex1, 'RGB')
        # Complementary of Color 1
        baseline_color = (255, 255, 255)
        tuple_color = tuple(np.subtract(baseline_color, rgb_color_1))
        hex_color = '#' + rgb_to_hex(tuple_color)
        st.session_state.color1 = hex1
        st.session_state.color2 = hex_color

    # Convert RGB to HEX color code
    def rgb_to_hex(rgb):
        return '%02x%02x%02x' % rgb

    # Convert the image to BytesIO so we can download it!
    def convert_image(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im

    # Sidebar input widgets
    with st.sidebar:
        st.header('⚙️ Settings')

        # Color selection
        st.subheader('Wallpaper Color Selection')
        with st.expander('Expand', expanded=True):
            color1 = st.color_picker('Choose the first color', key='color1')
            color2 = st.color_picker('Choose the second color', key='color2')
            st.button('Random complementary colors', on_click=generate_random_hex_color)
        
        # Add title text
        st.subheader('Title Text')
        with st.expander('Expand'):
            st.markdown('### Line 1 Text')
            title_text_1 = st.text_input('Enter text', 'Text 1')
            title_font_1 = st.slider('Font size', 10, 200, 150, step=10)
            bounding_box_1 = st.checkbox('Black bounding box for text', value=True, key='bounding_box_1')
            left_margin_number_1 = st.number_input('Left margin', 0, 800, 50, step=10, key='left_margin_number_1')
            top_margin_number_1 = st.number_input('Top margin', 0, 800, 340, step=10, key='top_margin_number_1')
            box_width_1 = st.number_input('Box width', 0, 1200, 750, step=10, key='box_width_1')
            box_height_1 = st.number_input('Box height', 0, 800, 520, step=10, key='box_height_1')

            st.markdown('### Line 2 Text')
            title_text_2 = st.text_input('Enter text', 'Text 2')
            title_font_2 = st.slider('Font size', 10, 200, 120, step=10)
            bounding_box_2 = st.checkbox('Black bounding box for text', value=True, key='bounding_box_2')
            left_margin_number_2 = st.number_input('Left margin', 0, 800, 50, step=10, key='left_margin_number_2')
            top_margin_number_2 = st.number_input('Top margin', 0, 800, 540, step=10, key='top_margin_number_2')
            box_width_2 = st.number_input('Box width', 0, 1200, 1010, step=10, key='box_width_2')
            box_height_2 = st.number_input('Box height', 0, 800, 700, step=10, key='box_height_2')
            
        # Image upload
        st.subheader('Image upload')
        with st.expander('Expand'):
            image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            image_resize = st.slider('Image size', 0.0, 8.0, 1.0, step=0.1)
            image_vertical_placement = st.slider('Vertical placement', 0, 1000, 0, step=25)
            image_horizontal_placement = st.slider('Horizontal placement', -1000, 1000, 0, step=25)

        # Add Streamlit logo
        st.subheader('Streamlit logo')
        with st.expander('Expand'):
            streamlit_logo = st.checkbox('Add Streamlit logo', value=True, key='streamlit_logo')
            logo_width = st.slider('Image width', 0, 500, 180, step=10)
            logo_vertical_placement = st.slider('Vertical placement', 0, 1000, 900, step=10)
            logo_horizontal_placement = st.slider('Horizontal placement', 0, 1800, 20, step=10)

    # Render wallpaper
    col1, col2 = st.columns(2)
    # with col1:
    with st.expander('See Rendered Wallpaper', expanded=True):
        st.subheader('Rendered Wallpaper')
        # Generate RGB color code from selected colors
        rgb_color1 = ImageColor.getcolor(color1, 'RGB')
        rgb_color2 = ImageColor.getcolor(color2, 'RGB')
        # Generate wallpaper
        main(rgb_color1, rgb_color2)
        with Image.open(f'{img_path}/wallpaper.png') as img:
            st.image(img)

    # Add text to wallpaper
    # with col2:
    with st.expander('See Wallpaper with Text', expanded=True):
        st.subheader('Wallpaper with Text')
        with Image.open(f'{img_path}/wallpaper.png') as img:
            title_font_1 = ImageFont.truetype('font/Montserrat-BlackItalic.ttf', title_font_1)
            title_font_2 = ImageFont.truetype('font/Montserrat-BlackItalic.ttf', title_font_2)

            img_edit = ImageDraw.Draw(img)
            if bounding_box_1:
                #img_edit.rectangle(((50, 340), (750, 520)), fill="black")
                img_edit.rectangle(((left_margin_number_1, top_margin_number_1), (box_width_1, box_height_1)), fill="black")
            if bounding_box_2:
                img_edit.rectangle(((left_margin_number_2, top_margin_number_2), (box_width_2, box_height_2)), fill="black")
            img_edit.text((85,340), title_text_1, (255, 255, 255), font=title_font_1)
            img_edit.text((85,550), title_text_2, (255, 255, 255), font=title_font_2)
            
            if streamlit_logo:
                logo_img = Image.open('streamlit-logo.png').convert('RGBA')
                logo_img.thumbnail([sys.maxsize, logo_width], Resampling.LANCZOS)
                img.paste(logo_img, (logo_horizontal_placement, logo_vertical_placement), logo_img)
                
            img.save(f'{img_path}/thumbnail.png')
            st.image(img)
            downloadable_thumbnail = convert_image(img)
            st.download_button("Download image", downloadable_thumbnail, "thumbnail.png", "image/png")

    # Remove background from photo
    if image_upload:
        st.subheader('Photo overlayed on Wallpaper')
        image = Image.open(image_upload)

        new_width = int(image.width * image_resize)
        new_height = int(image.height * image_resize)
        resized_image = image.resize((new_width, new_height))
        fixed = remove(resized_image)
        
        #fixed = remove(image)
        fixed.save(f'{img_path}/photo.png')

        # Overlay photo on wallpaper
        base_img = Image.open(f'{img_path}/thumbnail.png').convert('RGBA')
        photo_img = Image.open(f'{img_path}/photo.png').convert('RGBA')
    
        base_img.paste(photo_img, (image_horizontal_placement, image_vertical_placement), photo_img)
        base_img.save(f'{img_path}/final.png')

        final_img = Image.open(f'{img_path}/final.png')
        st.image(final_img)

        # Download final thumbnail image
        downloadable_image = convert_image(final_img)
        st.download_button("Download final image", downloadable_image, "thumbnail_image.png", "image/png")

def video_to_gif():
    import os
    import base64
    import tempfile
    from PIL import Image
    import numpy as np
    from moviepy.editor import VideoFileClip
    import moviepy.video.fx.all as vfx

    ## Session state ##
    if 'clip_width' not in st.session_state:
        st.session_state.clip_width = 0
    if 'clip_height' not in st.session_state:
        st.session_state.clip_height = 0
    if 'clip_duration' not in st.session_state:
        st.session_state.clip_duration = 0
    if 'clip_fps' not in st.session_state:
        st.session_state.clip_fps = 0
    if 'clip_total_frames' not in st.session_state:
        st.session_state.clip_total_frames = 0  
        
    st.title('🎈 Animated GIF Maker')

    ## Upload file ##
    st.sidebar.header('Upload file')
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['mov', 'mp4'])

    ## Display gif generation parameters once file has been uploaded ##
    if uploaded_file is not None:
        ## Save to temp file ##
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        ## Open file ##
        clip = VideoFileClip(tfile.name)
        
        st.session_state.clip_duration = clip.duration
        
        ## Input widgets ##
        st.sidebar.header('Input parameters')
        selected_resolution_scaling = st.sidebar.slider('Scaling of video resolution', 0.0, 1.0, 0.5 )
        selected_speedx = st.sidebar.slider('Playback speed', 0.1, 10.0, 5.0)
        selected_export_range = st.sidebar.slider('Duration range to export', 0, int(st.session_state.clip_duration), (0, int(st.session_state.clip_duration) ))
        
        ## Resizing of video ##
        clip = clip.resize(selected_resolution_scaling)
        
        st.session_state.clip_width = clip.w
        st.session_state.clip_height = clip.h
        st.session_state.clip_duration = clip.duration
        st.session_state.clip_total_frames = clip.duration * clip.fps
        st.session_state.clip_fps = st.sidebar.slider('FPS', 10, 60, 20)
        
        ## Display output ##
        st.subheader('Metrics')
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Width', st.session_state.clip_width, 'pixels')
        col2.metric('Height', st.session_state.clip_height, 'pixels')
        col3.metric('Duration', st.session_state.clip_duration, 'seconds')
        col4.metric('FPS', st.session_state.clip_fps, '')
        col5.metric('Total Frames', st.session_state.clip_total_frames, 'frames')

        # Extract video frame as a display image
        st.subheader('Preview')

        with st.expander('Show image'):
            selected_frame = st.slider('Preview a time frame (s)', 0, int(st.session_state.clip_duration), int(np.median(st.session_state.clip_duration)) )
            clip.save_frame('frame.gif', t=selected_frame)
            frame_image = Image.open('frame.gif')
            st.image(frame_image)

        ## Print image parameters ##
        st.subheader('Image parameters')
        with st.expander('Show image parameters'):
            st.write(f'File name: `{uploaded_file.name}`')
            st.write('Image size:', frame_image.size)
            st.write('Video resolution scaling', selected_resolution_scaling)
            st.write('Speed playback:', selected_speedx)
            st.write('Export duration:', selected_export_range)
            st.write('Frames per second (FPS):', st.session_state.clip_fps)
        
        ## Export animated GIF ##
        st.subheader('Generate GIF')
        generate_gif = st.button('Generate Animated GIF')
        
        if generate_gif:
            clip = clip.subclip(selected_export_range[0], selected_export_range[1]).speedx(selected_speedx)
            
            frames = []
            for frame in clip.iter_frames():
                frames.append(np.array(frame))
            
            image_list = []

            for frame in frames:
                im = Image.fromarray(frame)
                image_list.append(im)

            image_list[0].save('export.gif', format = 'GIF', save_all = True, loop = 0, append_images = image_list)
            
            #clip.write_gif('export.gif', fps=st.session_state.clip_fps)
            
            ## Download ##
            st.subheader('Download')
            
            #video_file = open('export.gif', 'rb')
            #video_bytes = video_file.read()
            #st.video(video_bytes)
            
            file_ = open('export.gif', 'rb')
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True,
            )
            
            fsize = round(os.path.getsize('export.gif')/(1024*1024), 1)
            st.info(f'File size of generated GIF: {fsize} MB', icon='💾')
            
            fname = uploaded_file.name.split('.')[0]
            with open('export.gif', 'rb') as file:
                btn = st.download_button(
                    label='Download image',
                    data=file,
                    file_name=f'{fname}_scaling-{selected_resolution_scaling}_fps-{st.session_state.clip_fps}_speed-{selected_speedx}_duration-{selected_export_range[0]}-{selected_export_range[1]}.gif',
                    mime='image/gif'
                )

    ## Default page ##
    else:
        st.warning('👈 Upload a video file')

def website_summarizer():
    import validators
    from langchain.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langchain.chains.summarize import load_summarize_chain
    from langchain.schema import Document
    from langchain_community.document_loaders import UnstructuredURLLoader

    # Streamlit App Configuration
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
    Provide detailed notes of the following content without missing anything:
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

def youtube_content_moderator():
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
    

def youtube_video_summarizer():
    from dotenv import load_dotenv
    load_dotenv() ##load all the nevironment variables
    import os
    from google import genai
    from youtube_transcript_api import YouTubeTranscriptApi

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = """
    You are a highly meticulous and intelligent YouTube video note-taker assistant. You will be given the full transcript of a YouTube video — it can be any type: lecture, podcast, tutorial, explainer, interview, motivational talk, documentary, etc.

    Your task is to create **exhaustive, structured notes** from the transcript, ensuring that **every single detail — no matter how small — is captured**.

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

    def generate_gemini_content(transcript_text,prompt):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents= prompt+transcript_text,
        )
        return response.text

    st.title("YouTube Transcript to Detailed Notes Converter")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes"):
        transcript_text=extract_transcript_details(youtube_link)

        if transcript_text:
            summary=generate_gemini_content(transcript_text,prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)
    

def blog_generator():
    from google import genai
    import os
    from dotenv import load_dotenv
    from PIL import Image
    from io import BytesIO

    # Load environment variables
    load_dotenv()

    # Configure Google Generative AI
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
    

def youtube_trend_analyzer():
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

def youtube_to_blog():
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

def chat_with_youtube():
    import os
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from google import genai
    from langchain.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv

    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Step 1: Extract transcript exactly as in your original code
    def extract_transcript_details(youtube_video_url):
        try:
            video_id = youtube_video_url.split("=")[1]

            all_languages = [
                'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'zh', 'zh-Hans', 'zh-Hant',
                'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha',
                'haw', 'he', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'rw', 'ko', 'ku',
                'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ny',
                'or', 'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so',
                'es', 'su', 'sw', 'sv', 'tl', 'tg', 'ta', 'tt', 'te', 'th', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy',
                'xh', 'yi', 'yo', 'zu'
            ]

            transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=all_languages)
            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]
            return transcript
        except Exception as e:
            raise e

    # Step 2: Process transcript into chunks like Chat with Files
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_yt_index")

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
        If the answer is not in the provided context, just say, "Answer is not available in the context." 
        Don't provide a wrong answer.
        
        Context:\n{context}\n
        Question:\n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def generate_response(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_yt_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        return response["output_text"]

    # Streamlit app
    def main():
        st.title("Chat with YouTube Video using Gemini + RAG 🎥🧠")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello! Paste a YouTube URL, and ask anything based on its content!"}
            ]

        # Display chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        youtube_link = st.text_input("Enter YouTube Video URL:")
        if youtube_link:
            try:
                video_id = youtube_link.split("=")[1]
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
            except:
                st.warning("Invalid YouTube URL format.")

        if st.button("Process Video"):
            if youtube_link:
                with st.spinner("Extracting and processing transcript..."):
                    try:
                        transcript_text = extract_transcript_details(youtube_link)
                        text_chunks = get_text_chunks(transcript_text)
                        get_vector_store(text_chunks)
                        st.success("Transcript processed and indexed successfully!")
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
            else:
                st.warning("Please enter a valid YouTube URL.")

        if prompt := st.chat_input("Ask a question based on the YouTube video..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = generate_response(prompt)
                    except Exception as e:
                        response = f"Error: {e}"

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)

    if __name__ == "__main__":
        main()

def image():
    import cv2
    import numpy as np
    from PIL import Image
    from rembg import remove
    from cartooner import cartoonize
    from io import BytesIO

    st.title("🖼️ Image Transformation Toolset")
    st.markdown("Apply various transformations like background removal, cartoonization, sketching, and more!")

    # Helper functions
    def convert_image(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im

    def dodgeV2(x, y):
        return cv2.divide(x, 255 - y, scale=256)

    def pencil_sketch(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_invert = cv2.bitwise_not(img_gray)
        img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
        final_img = dodgeV2(img_gray, img_smoothing)
        return final_img

    def apply_sepia(img):
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(img, sepia_filter)
        return np.clip(sepia_img, 0, 255).astype(np.uint8)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📤 Upload","🖌️ Text to Image", "🖍️ Image editor", "🧼 Background Remover", "🧑‍🎨 Cartoonizer", "✏️ Pencil Sketch", "🎨 Image Colorizer", "🎞️ Effects"])

    # File uploader shared across tabs
    with tab1:
        uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'])
        st.session_state['file_uploaded'] = uploaded_file is not None
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success("Image uploaded successfully!")

    with tab2:
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        import os

        # Set your API key here or use an environment variable
        GOOGLE_API_KEY = "GOOGLE_API_KEY" # Replace with your own if needed

        # Initialize Gemini client
        client = genai.Client(api_key=GOOGLE_API_KEY)

        prompt = st.text_area("Enter a creative prompt for image generation:")

        if st.button("Generate Image"):
            if prompt.strip():
                with st.spinner("Generating image using Gemini..."):
                    try:
                        contents = prompt

                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp-image-generation",
                            contents=contents,
                            config=types.GenerateContentConfig(
                                response_modalities=['TEXT', 'IMAGE']
                            )
                        )

                        for part in response.candidates[0].content.parts:
                            if part.text is not None:
                                st.markdown("**Gemini's Response:**")
                                st.write(part.text)
                            elif part.inline_data is not None:
                                image = Image.open(BytesIO(part.inline_data.data))
                                st.markdown("**Generated Image:**")
                                st.image(image, use_column_width=True)
                                image.save("generated_image.png")
                                with open("generated_image.png", "rb") as img_file:
                                    st.download_button("⬇️ Download Image", img_file, file_name="generated_image.png")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a valid prompt.")

    with tab3:
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        import os

        # Set your API key here or use an environment variable
        GOOGLE_API_KEY = "AIzaSyBWc7Ym6qMSo04uD-KtfT1JSin5AqhtyNg" # Replace with your own if needed

        # Initialize Gemini client
        client = genai.Client(api_key=GOOGLE_API_KEY)

        uploaded_image = st.file_uploader("Upload an image to edit", type=["png", "jpg", "jpeg"])
        edit_prompt = st.text_area("Describe the edit you want (e.g., 'Add a llama next to me')")

        if st.button("Edit Image"):
            if uploaded_image and edit_prompt.strip():
                with st.spinner("Editing image using Gemini..."):
                    try:
                        image = Image.open(uploaded_image)

                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp-image-generation",
                            contents=[edit_prompt, image],
                            config=types.GenerateContentConfig(
                                response_modalities=['TEXT', 'IMAGE']
                            )
                        )

                        for part in response.candidates[0].content.parts:
                            if part.text is not None:
                                st.markdown("**Gemini's Response:**")
                                st.write(part.text)
                            elif part.inline_data is not None:
                                edited_image = Image.open(BytesIO(part.inline_data.data))
                                st.markdown("**Edited Image:**")
                                st.image(edited_image, use_column_width=True)
                                edited_image.save("edited_image.png")
                                with open("edited_image.png", "rb") as img_file:
                                    st.download_button("⬇️ Download Edited Image", img_file, file_name="edited_image.png")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload an image and provide an edit prompt.")

    with tab4:
        st.header("Remove Background")
        if st.session_state.get('file_uploaded', False):
            alpha_matting = st.checkbox("Use Alpha Matting", value=True)
            threshold = st.slider("Background Threshold", 0, 255, 50, step=5)

            if st.button("Remove Background"):
                output = remove(image, alpha_matting=alpha_matting, alpha_matting_foreground_threshold=threshold)
                st.image(output, caption="Background Removed", use_column_width=True)
                st.download_button("Download", convert_image(output), "bg_removed.png", "image/png")
        else:
            st.warning("Please upload an image in the Upload tab.")

    # Cartoonize
    with tab5:
        import numpy as np
        import cv2
        from PIL import Image
        import io

        st.header("Cartoonize Images")

        def convert_comic(img):
            return cv2.stylization(img, sigma_s=150, sigma_r=0.25)

        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(image)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            st.image(image, caption="Original Image", use_column_width=True)
            
            with st.spinner("Processing image..."):
                output_np = convert_comic(img_np)
                output_rgb = cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)
                output_pil = Image.fromarray(output_rgb)

                st.image(output_pil, caption="Comic Style Image", use_column_width=True)

                # Download button
                buf = io.BytesIO()
                output_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="📥 Download Comic Image",
                    data=byte_im,
                    file_name="comic_style.png",
                    mime="image/png"
                )
        else:
            st.warning("Please upload an image in the Upload tab.")    


    # Pencil Sketch
    with tab6:
        st.header("Pencil Sketch Effect")
        if st.session_state.get('file_uploaded', False):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            sketch = pencil_sketch(img_cv)
            st.image(sketch, channels="GRAY", caption="Pencil Sketch", use_column_width=True)

            pil_sketch = Image.fromarray(sketch)
            st.download_button("Download", convert_image(pil_sketch), "sketch.png", "image/png")
        else:
            st.warning("Please upload an image in the Upload tab.")

    with tab7:
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO

        # Set your Gemini API Key
        GOOGLE_API_KEY = "GOOGLE_API_KEY"  # Replace this with your key

        client = genai.Client(api_key=GOOGLE_API_KEY)

        st.header("Black & White Image Colorifier")

        # Upload grayscale image
        uploaded_image = st.file_uploader("Upload a black & white image", type=["png", "jpg", "jpeg"])

        if uploaded_image:
            st.image(uploaded_image, caption="Original Black & White Image", use_column_width=True)

            if st.button("Colorify Image"):
                with st.spinner("Colorizing using Gemini..."):
                    try:
                        image = Image.open(uploaded_image)

                        # Prompt to Gemini for colorizing
                        prompt = "This is a black and white image. Please colorize it naturally."

                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp-image-generation",
                            contents=[prompt, image],
                            config=types.GenerateContentConfig(
                                response_modalities=['TEXT', 'IMAGE']
                            )
                        )

                        for part in response.candidates[0].content.parts:
                            if part.text is not None:
                                st.markdown("**Gemini's Note:**")
                                st.write(part.text)
                            elif part.inline_data is not None:
                                color_image = Image.open(BytesIO(part.inline_data.data))
                                st.markdown("**Colorized Image:**")
                                st.image(color_image, use_column_width=True)
                                color_image.save("colorized_image.png")
                                with open("colorized_image.png", "rb") as img_file:
                                    st.download_button("⬇️ Download Colorized Image", img_file, file_name="colorized_image.png")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.info("Please upload a black and white image to begin.")
            

    # Effects: Grayscale, Sepia
    with tab8:
        st.header("Image Effects")
        if st.session_state.get('file_uploaded', False):
            effect = st.selectbox("Choose effect", ["Grayscale", "Sepia"])

            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            if effect == "Grayscale":
                result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                st.image(result, channels="GRAY", caption="Grayscale", use_column_width=True)
                pil = Image.fromarray(result)
            elif effect == "Sepia":
                result = apply_sepia(img_cv)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result, caption="Sepia", use_column_width=True)
                pil = Image.fromarray(result)

            st.download_button("Download", convert_image(pil), f"{effect.lower()}.png", "image/png")
        else:
            st.warning("Please upload an image in the Upload tab.")               

        

def avatar_generator():
    import py_avataaars as pa
    from PIL import Image
    import base64
    from random import randrange

    # Page title
    st.markdown("""
    # Avatar Maker

    This app allows you to build your own custom avatars based on modular templates provided herein.
    """)

    # Sidebar menu for customizing the avatar
    st.sidebar.header('Customize your avatar')

    option_style = st.sidebar.selectbox('Style', ('CIRCLE', 'TRANSPARENT'))

    list_skin_color = ['TANNED', 'YELLOW', 'PALE', 'LIGHT', 'BROWN', 'DARK_BROWN', 'BLACK']
    list_top_type = ['NO_HAIR', 'EYE_PATCH', 'HAT', 'HIJAB', 'TURBAN',
                     'WINTER_HAT1', 'WINTER_HAT2', 'WINTER_HAT3',
                     'WINTER_HAT4', 'LONG_HAIR_BIG_HAIR', 'LONG_HAIR_BOB',
                     'LONG_HAIR_BUN', 'LONG_HAIR_CURLY', 'LONG_HAIR_CURVY',
                     'LONG_HAIR_DREADS', 'LONG_HAIR_FRIDA', 'LONG_HAIR_FRO',
                     'LONG_HAIR_FRO_BAND', 'LONG_HAIR_NOT_TOO_LONG',
                     'LONG_HAIR_SHAVED_SIDES', 'LONG_HAIR_MIA_WALLACE',
                     'LONG_HAIR_STRAIGHT', 'LONG_HAIR_STRAIGHT2',
                     'LONG_HAIR_STRAIGHT_STRAND', 'SHORT_HAIR_DREADS_01',
                     'SHORT_HAIR_DREADS_02', 'SHORT_HAIR_FRIZZLE',
                     'SHORT_HAIR_SHAGGY_MULLET', 'SHORT_HAIR_SHORT_CURLY',
                     'SHORT_HAIR_SHORT_FLAT', 'SHORT_HAIR_SHORT_ROUND',
                     'SHORT_HAIR_SHORT_WAVED', 'SHORT_HAIR_SIDES',
                     'SHORT_HAIR_THE_CAESAR', 'SHORT_HAIR_THE_CAESAR_SIDE_PART']
    list_hair_color = ['AUBURN', 'BLACK', 'BLONDE', 'BLONDE_GOLDEN', 'BROWN',
                       'BROWN_DARK', 'PASTEL_PINK', 'PLATINUM', 'RED', 'SILVER_GRAY']
    list_hat_color = ['BLACK', 'BLUE_01', 'BLUE_02', 'BLUE_03', 'GRAY_01', 'GRAY_02',
                      'HEATHER', 'PASTEL_BLUE', 'PASTEL_GREEN', 'PASTEL_ORANGE',
                      'PASTEL_RED', 'PASTEL_YELLOW', 'PINK', 'RED', 'WHITE']

    list_facial_hair_type = ['DEFAULT', 'BEARD_MEDIUM', 'BEARD_LIGHT', 'BEARD_MAJESTIC', 'MOUSTACHE_FANCY', 'MOUSTACHE_MAGNUM']
    list_facial_hair_color = ['AUBURN', 'BLACK', 'BLONDE', 'BLONDE_GOLDEN', 'BROWN', 'BROWN_DARK', 'PLATINUM', 'RED']
    list_mouth_type = ['DEFAULT', 'CONCERNED', 'DISBELIEF', 'EATING', 'GRIMACE', 'SAD', 'SCREAM_OPEN', 'SERIOUS', 'SMILE', 'TONGUE', 'TWINKLE', 'VOMIT']
    list_eye_type = ['DEFAULT', 'CLOSE', 'CRY', 'DIZZY', 'EYE_ROLL', 'HAPPY', 'HEARTS', 'SIDE', 'SQUINT', 'SURPRISED', 'WINK', 'WINK_WACKY']
    list_eyebrow_type = ['DEFAULT', 'DEFAULT_NATURAL', 'ANGRY', 'ANGRY_NATURAL', 'FLAT_NATURAL', 'RAISED_EXCITED', 'RAISED_EXCITED_NATURAL', 'SAD_CONCERNED', 'SAD_CONCERNED_NATURAL', 'UNI_BROW_NATURAL', 'UP_DOWN', 'UP_DOWN_NATURAL', 'FROWN_NATURAL']
    list_accessories_type = ['DEFAULT', 'KURT', 'PRESCRIPTION_01', 'PRESCRIPTION_02', 'ROUND', 'SUNGLASSES', 'WAYFARERS']
    list_clothe_type = ['BLAZER_SHIRT', 'BLAZER_SWEATER', 'COLLAR_SWEATER', 'GRAPHIC_SHIRT', 'HOODIE', 'OVERALL', 'SHIRT_CREW_NECK', 'SHIRT_SCOOP_NECK', 'SHIRT_V_NECK']
    list_clothe_color = ['BLACK', 'BLUE_01', 'BLUE_02', 'BLUE_03', 'GRAY_01', 'GRAY_02', 'HEATHER', 'PASTEL_BLUE', 'PASTEL_GREEN', 'PASTEL_ORANGE', 'PASTEL_RED', 'PASTEL_YELLOW', 'PINK', 'RED', 'WHITE']
    list_clothe_graphic_type = ['BAT', 'CUMBIA', 'DEER', 'DIAMOND', 'HOLA', 'PIZZA', 'RESIST', 'SELENA', 'BEAR', 'SKULL_OUTLINE', 'SKULL']

    if st.button('Random Avatar'):
        index_skin_color = randrange(0, len(list_skin_color))
        index_top_type = randrange(0, len(list_top_type))
        index_hair_color = randrange(0, len(list_hair_color))
        index_hat_color = randrange(0, len(list_hat_color))
        index_facial_hair_type = randrange(0, len(list_facial_hair_type))
        index_facial_hair_color = randrange(0, len(list_facial_hair_color))
        index_mouth_type = randrange(0, len(list_mouth_type))
        index_eye_type = randrange(0, len(list_eye_type))
        index_eyebrow_type = randrange(0, len(list_eyebrow_type))
        index_accessories_type = randrange(0, len(list_accessories_type))
        index_clothe_type = randrange(0, len(list_clothe_type))
        index_clothe_color = randrange(0, len(list_clothe_color))
        index_clothe_graphic_type = randrange(0, len(list_clothe_graphic_type))
    else:
        index_skin_color = 0
        index_top_type = 0
        index_hair_color = 0
        index_hat_color = 0
        index_facial_hair_type = 0
        index_facial_hair_color = 0
        index_mouth_type = 0
        index_eye_type = 0
        index_eyebrow_type = 0
        index_accessories_type = 0
        index_clothe_type = 0
        index_clothe_color = 0
        index_clothe_graphic_type = 0

    option_skin_color = st.sidebar.selectbox('Skin color',
                                             list_skin_color,
                                             index=index_skin_color)

    st.sidebar.subheader('Head top')
    option_top_type = st.sidebar.selectbox('Head top',
                                           list_top_type,
                                           index=index_top_type)
    option_hair_color = st.sidebar.selectbox('Hair color',
                                             list_hair_color,
                                             index=index_hair_color)
    option_hat_color = st.sidebar.selectbox('Hat color',
                                            list_hat_color,
                                            index=index_hat_color)

    st.sidebar.subheader('Face')
    option_facial_hair_type = st.sidebar.selectbox('Facial hair type',
                                                   list_facial_hair_type,
                                                   index=index_facial_hair_type)
    option_facial_hair_color = st.sidebar.selectbox('Facial hair color',
                                                    list_facial_hair_color,
                                                    index=index_facial_hair_color)
    option_mouth_type = st.sidebar.selectbox('Mouth type',
                                             list_mouth_type,
                                             index=index_mouth_type)
    option_eye_type = st.sidebar.selectbox('Eye type',
                                           list_eye_type,
                                           index=index_eye_type)
    option_eyebrow_type = st.sidebar.selectbox('Eyebrow type',
                                               list_eyebrow_type,
                                               index=index_eyebrow_type)

    st.sidebar.subheader('Clothe and accessories')
    option_accessories_type = st.sidebar.selectbox('Accessories type',
                                                   list_accessories_type,
                                                   index=index_accessories_type)
    option_clothe_type = st.sidebar.selectbox('Clothe type',
                                              list_clothe_type,
                                              index=index_clothe_type)
    option_clothe_color = st.sidebar.selectbox('Clothe Color',
                                               list_clothe_color,
                                               index=index_clothe_color)
    option_clothe_graphic_type = st.sidebar.selectbox('Clothe graphic type',
                                                      list_clothe_graphic_type,
                                                      index=index_clothe_graphic_type)

    # Creating the Avatar
    # options provided in https://github.com/kebu/py-avataaars/blob/master/py_avataaars/__init__.py
    avatar = pa.PyAvataaar(
        # style=pa.AvatarStyle.CIRCLE,
        style=eval('pa.AvatarStyle.%s' % option_style),
        skin_color=eval('pa.SkinColor.%s' % option_skin_color),
        top_type=eval('pa.TopType.SHORT_HAIR_SHORT_FLAT.%s' % option_top_type),
        hair_color=eval('pa.HairColor.%s' % option_hair_color),
        hat_color=eval('pa.ClotheColor.%s' % option_hat_color),
        facial_hair_type=eval('pa.FacialHairType.%s' % option_facial_hair_type),
        facial_hair_color=eval('pa.FacialHairColor.%s' % option_facial_hair_color),
        mouth_type=eval('pa.MouthType.%s' % option_mouth_type),
        eye_type=eval('pa.EyesType.%s' % option_eye_type),
        eyebrow_type=eval('pa.EyebrowType.%s' % option_eyebrow_type),
        nose_type=pa.NoseType.DEFAULT,
        accessories_type=eval('pa.AccessoriesType.%s' % option_accessories_type),
        clothe_type=eval('pa.ClotheType.%s' % option_clothe_type),
        clothe_color=eval('pa.ClotheColor.%s' % option_clothe_color),
        clothe_graphic_type=eval('pa.ClotheGraphicType.%s' % option_clothe_graphic_type)
    )

    # Custom function for encoding and downloading avatar image
    def imagedownload(filename):
        image_file = open(filename, 'rb')
        b64 = base64.b64encode(image_file.read()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
        return href

    st.subheader('**Rendered Avatar**')
    rendered_avatar = avatar.render_png_file('avatar.png')
    image = Image.open('avatar.png')
    st.image(image)
    st.markdown(imagedownload('avatar.png'), unsafe_allow_html=True)

def youtube_translator():
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

def website_chat():
    import validators
    from langchain_community.document_loaders import UnstructuredURLLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from google import genai
    from dotenv import load_dotenv
    import os

    # Load environment variables
    load_dotenv()
    genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # ---- Helper Functions ----

    def load_website_content(url):
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
        return content

    def get_text_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return splitter.split_text(text)

    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_web_index")

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
        If the answer is not in the provided context, just say, "Answer is not available in the context." 
        Don't provide a wrong answer.

        Context:\n{context}\n
        Question:\n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def generate_response(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_web_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return result["output_text"]

    # ---- Streamlit UI ----

    def main():
        st.title("Chat with Website 🌐 using Gemini + RAG")

        # Session State for Chat History
        if "message" not in st.session_state:
            st.session_state["message"] = [
                {"role": "assistant", "content": "Hello! Enter a website URL, and ask anything based on its content!"}
            ]

        # Display chat history
        for msg in st.session_state["message"]:
            st.chat_message(msg["role"]).write(msg["content"])

        # Input: Website URL
        website_url = st.text_input("Enter Website URL:")

        if website_url and not validators.url(website_url):
            st.warning("Please enter a valid website URL.")

        if st.button("Process Website"):
            if website_url and validators.url(website_url):
                with st.spinner("Fetching and processing website content..."):
                    try:
                        content = load_website_content(website_url)
                        chunks = get_text_chunks(content)
                        get_vector_store(chunks)
                        st.success("Website content processed and indexed successfully!")
                    except Exception as e:
                        st.error(f"Error loading website: {e}")
            else:
                st.warning("Please enter a valid URL.")

        # Chat input
        if prompt := st.chat_input("Ask a question based on the website content..."):
            st.session_state.message.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = generate_response(prompt)
                    except Exception as e:
                        response = f"Error: {e}"
                st.session_state.message.append({"role": "assistant", "content": response})
                st.write(response)

    if __name__ == "__main__":
        main()          


# Main app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app",
                                    ["Home Page","Thumbnail Generator", "Video to GIF Converter", "Website Summarizer", "Chat With Website", "Chat With YouTube", 
                                     "YouTube Content Moderator", "YouTube Video Summarizer","YouTube Trend Analyzer", "YouTube Translator", "YouTube To Blog Generator", "Blog Generator",
                                     "Image Transformation Toolset", "Avatar Generator"])
    if app_mode == "Home Page":
        front()
    elif app_mode == "Thumbnail Generator":
        thumbnail_generator()
    elif app_mode == "Video to GIF Converter":
        video_to_gif()
    elif app_mode == "Website Summarizer":
        website_summarizer()
    elif app_mode == "Chat With YouTube":
        chat_with_youtube()
    elif app_mode == "Chat With Website":
        website_chat()        
    elif app_mode == "YouTube Content Moderator":
        youtube_content_moderator()
    elif app_mode == "YouTube Video Summarizer":
        youtube_video_summarizer()
    elif app_mode == "YouTube Trend Analyzer":
        youtube_trend_analyzer()
    elif app_mode == "YouTube Translator":
        youtube_translator()    
    elif app_mode == "YouTube To Blog Generator":
        youtube_to_blog()      
    elif app_mode == "Blog Generator":
        blog_generator()
    elif app_mode == "Image Transformation Toolset":
        image()    
    elif app_mode == "Avatar Generator":
        avatar_generator()

if __name__ == "__main__":
    main()
