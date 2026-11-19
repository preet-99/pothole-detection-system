import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
import requests

try:
    model = load_model("pothole_detection_cnn.keras")
except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e).__name__}")


st.title("Road Pothole Detection System")
st.header("Check your road image has pothole or not")


def predictions(image: Image.Image) -> Image.Image:
    img = image.convert("RGB").resize((128, 128))
    img = np.array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    return result

file_name: str | None = None
source_type = st.radio(
    "Select Image Source", ["File Upload", "Phone / Device", "Image Link(URL)"], horizontal=True
)

image: Image.Image | None = None
if source_type == "File Upload":
    uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
       file_name = uploaded_file.name
       image = Image.open(uploaded_file)

elif source_type == "Phone / Device":
    camera_photo = st.camera_input("Click photo with your camera")
    if camera_photo is not None:
       file_name = camera_photo.name
       image = Image.open(camera_photo)

elif source_type == "Image Link(URL)":
    url = st.text_input("Paste your link here")
    if url:
        try:
            response = requests.get(url.strip(), timeout=10)
            response.raise_for_status()
            file_name = url.strip().split("/")[-1].split("?")[0] or "url_image.jpg"
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Image is not load using Link: {e}")


if image is not None:
    st.image(image, caption="Uploaded Image", width="stretch")
    st.write("File Name", file_name)

    if st.button("Find", type="primary"):
        result = predictions(image)
        if result < 0.5:
            st.write(f"Normal")
        else:
            st.write(f"Pathole")
# else:
#   st.info("Please upload image first")
