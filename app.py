import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

MODEL_PATH = "cats_vs_dogs_model.keras"
IMAGE_SIZE = (128, 128)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and the trained CNN model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)

    # The model was trained with image_dataset_from_directory,
    # so pixel values are kept in the normal 0-255 image range.
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array, verbose=0)[0][0])

    # Keras binary classification convention:
    # sigmoid < 0.5 -> first class (cats), >= 0.5 -> second class (dogs)
    if prediction < 0.5:
        label = "Cat 🐱"
        confidence = (1 - prediction) * 100
    else:
        label = "Dog 🐶"
        confidence = prediction * 100

    st.subheader(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence:.2f}%")

    if confidence < 60:
        st.warning("The model is not very confident about this prediction.")
    else:
        st.success(f"The model predicts this image is a {label.split()[0].lower()}.")

st.markdown("---")
st.caption("Powered by TensorFlow + Streamlit")
