import streamlit as st

# Set page config once at the beginning of the script
st.set_page_config(page_title="CreatiFlow", layout="wide",page_icon="🎨")


def front():
    import streamlit as st

    # Title and description
    st.title("🎨 CreatiFlow - Your Creative Companion")
    st.write("Welcome to **CreatiFlow**! Unleash your creativity with these powerful tools designed for content creators, marketers, and digital enthusiasts.")

    # App descriptions
    st.subheader("Available Features")
    st.write("""
    1. **Thumbnail Generator**: Create stunning YouTube thumbnails with customizable colors, text, and images.
    2. **Video to GIF Converter**: Convert your videos into high-quality GIFs with adjustable speed and resolution.
    3. **Website Summarizer**: Summarize any website content into concise, easy-to-read points.
    4. **YouTube Content Moderator**: Analyze YouTube video transcripts for content moderation and policy violations.
    5. **YouTube Video Summarizer**: Get detailed notes and summaries from YouTube video transcripts.
    6. **Blog Generator**: Craft engaging and SEO-friendly blog posts with AI assistance.
    7. **Avatar Generator**: Design unique and personalized avatars for your projects or profiles.
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
            color1 = st.color_picker('Choose the first color', st.session_state.color1, key='color1')
            color2 = st.color_picker('Choose the second color', st.session_state.color2, key='color2')
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

def youtube_content_moderator():
    from dotenv import load_dotenv
    import os
    import google.generativeai as genai
    from youtube_transcript_api import YouTubeTranscriptApi

    load_dotenv()  # Load all the environment variables
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = """You are YouTube video content moderator. You will be taking the transcript text
    and moderating the entire video content as per YouTube content policies and providing the content moderation recommendation
    or the places where the content in video is violating the policies in points
    within 250 words. Please provide the same for the text given here:  """

    # Getting the transcript data from YouTube videos
    def extract_transcript_details(youtube_video_url):
        try:
            video_id = youtube_video_url.split("=")[1]
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]

            return transcript

        except Exception as e:
            raise e

    # Getting the summary based on Prompt from Google Gemini Pro
    def generate_gemini_content(transcript_text, prompt):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    st.title("YouTube Content Moderator")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get AI Recommendation"):
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Moderation Recommendation:")
            st.write(summary)

def youtube_video_summarizer():
    from dotenv import load_dotenv
    import os
    import google.generativeai as genai
    from youtube_transcript_api import YouTubeTranscriptApi

    load_dotenv()  # Load all the environment variables
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = """You are YouTube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points
    within 250 words. Please provide the summary of the text given here:  """

    # Getting the transcript data from YouTube videos
    def extract_transcript_details(youtube_video_url):
        try:
            video_id = youtube_video_url.split("=")[1]
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]

            return transcript

        except Exception as e:
            raise e

    # Getting the summary based on Prompt from Google Gemini Pro
    def generate_gemini_content(transcript_text, prompt):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    st.title("YouTube Transcript to Detailed Notes Converter")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes"):
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)

def blog_generator():
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
            f"Generate a comprehensive, engaging blog post relevant to the given title \"{blog_title}\" and keywords \"{keywords}\". Make sure to incorporate these keywords in the blog post. The blog should be approximately {num_words} words."
        ]

        # Submit button
        submit_button = st.button("Generate Blog")

    if submit_button:
        response = model.generate_content(prompt_parts)
        st.write(response.text)

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

# Main app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app",
                                    ["Home Page","Thumbnail Generator", "Video to GIF Converter", "Website Summarizer",
                                     "YouTube Content Moderator", "YouTube Video Summarizer", "Blog Generator",
                                     "Avatar Generator"])
    if app_mode == "Home Page":
        front()
    elif app_mode == "Thumbnail Generator":
        thumbnail_generator()
    elif app_mode == "Video to GIF Converter":
        video_to_gif()
    elif app_mode == "Website Summarizer":
        website_summarizer()
    elif app_mode == "YouTube Content Moderator":
        youtube_content_moderator()
    elif app_mode == "YouTube Video Summarizer":
        youtube_video_summarizer()
    elif app_mode == "Blog Generator":
        blog_generator()
    elif app_mode == "Avatar Generator":
        avatar_generator()

if __name__ == "__main__":
    main()