import streamlit as st
import cv2
from PIL import Image, ImageDraw
import numpy as np

# Function to perform basic image operations
def perform_basic_operations(image, operation, params):
    if operation == "Resize":
        new_width, new_height = params
        return cv2.resize(image, (new_width, new_height))
    elif operation == "Rotate":
        angle = params
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (cols, rows))
    elif operation == "Flip":
        return cv2.flip(image, 1)

# Streamlit app
def main():
    st.title("🎨 Image Editing App")

    # Image upload and display
    uploaded_image = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        original_image = np.array(image)

        # Display original image
        col1, col2 = st.columns(2)
        col1.image(original_image, caption="Original Image", use_column_width=True)

        # Basic image operations
        col2.subheader("🔧 Basic Operations")
        selected_operation = col2.selectbox("Select Operation", ["None", "Resize", "Rotate", "Flip"])

        if selected_operation != "None":
            if selected_operation == "Resize":
                new_width = col2.slider("New Width", 1, 1000, 500)
                new_height = col2.slider("New Height", 1, 1000, 500)
                edited_image = perform_basic_operations(original_image, selected_operation, (new_width, new_height))
            elif selected_operation == "Rotate":
                angle = col2.slider("Rotation Angle", -180, 180, 0)
                edited_image = perform_basic_operations(original_image, selected_operation, angle)
            elif selected_operation == "Flip":
                edited_image = perform_basic_operations(original_image, selected_operation, None)

            col2.image(edited_image, caption=f"{selected_operation} Image", use_column_width=True)

        # Color manipulation
        st.sidebar.subheader("🌈 Color Manipulation")
        brightness_factor = st.sidebar.slider("Brightness", 0.0, 3.0, 1.0)
        brightened_image = cv2.convertScaleAbs(original_image, alpha=brightness_factor, beta=0)

        # Filter effects
        st.sidebar.subheader("🎨 Filter Effects")
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Object detection (example using face detection)
        st.sidebar.subheader("👀 Object Detection")
        if st.sidebar.button("Detect Faces"):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Drawing feature
        st.sidebar.subheader("✏️ Drawing")
        drawing_checkbox = st.sidebar.checkbox("Enable Drawing")
        drawing_data = Image.fromarray(original_image.copy())

        if drawing_checkbox:
            st.sidebar.info("Click and drag to draw on the image.")
            draw = ImageDraw.Draw(drawing_data)
            st.image(drawing_data, use_column_width=True)
            st.info("Drawing enabled. Click and drag on the image.")

        # Display images horizontally
        col1, col2, col3 = st.columns(3)
        col1.image(brightened_image, caption="Brightened Image", use_column_width=True)
        col2.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
        col3.image(original_image, caption="Faces Detected", use_column_width=True)

        # Save and download
        if st.button("💾 Save and Download"):
            cv2.imwrite("edited_image.jpg", original_image)
            st.success("Image saved successfully!")
            st.download_button("Download Edited Image", "edited_image.jpg")

if __name__ == "__main__":
    main()
