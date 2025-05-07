import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Load your trained model
model = YOLO("./best.pt")  # <-- update path if needed

st.title("📷 Object Detection App")
st.write("Upload an image to detect multiple objects using your custom YOLOv8 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction when button is clicked
    if st.button("🔍 Run Detection"):
        with st.spinner("Detecting..."):
            # Save image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                results = model(tmp_file.name)

            # Plot the detection results
            result_image = results[0].plot()  # returns a numpy array

            # Display result
            st.image(result_image, caption="Detection Results", use_column_width=True)

            # Optional: clean up temp file
            os.unlink(tmp_file.name)
