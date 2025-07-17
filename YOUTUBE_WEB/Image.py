import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from cartooner import cartoonize
from io import BytesIO

st.set_page_config(page_title="Image Transformer Suite", layout="wide")
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
    GOOGLE_API_KEY = "GOOGLE_API_KEY" # Replace with your own if needed

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
    GOOGLE_API_KEY = "AIzaSyCWl0CpJtyV348SJ9DwOpRLcEKmMaW8ZMw"  # Replace this with your key

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











