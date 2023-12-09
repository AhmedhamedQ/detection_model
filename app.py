import streamlit as st
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
from io import BytesIO
import cv2
from datetime import datetime

# Disable eager execution for compatibility with TensorFlow 1.x
tf.disable_eager_execution()

model = load_model('VGG16.keras')

classes = {0: 'there are animals', 1: 'no animals there are'}

def rescale(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return tf.image.resize(image, [224, 224])

def decode_image(uploaded_file):
    content = uploaded_file.read()
    img = tf.image.decode_jpeg(content, channels=3)
    img = rescale(img)

    # Run the TensorFlow session to convert the symbolic tensor to a numpy array
    with tf.compat.v1.Session() as sess:
        img = sess.run(img)

    return np.expand_dims(img, axis=0)

# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    # Resize the frame to match the expected input size of the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Replace the following line with your actual preprocessing steps
    # For simplicity, let's assume the frame is already preprocessed
    return resized_frame

# Streamlit app
st.set_page_config(page_title='Animals classify' , page_icon='🐼')
st.title("Image and Video Classification App")

# File uploader for both photo and video
uploaded_files = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"] , accept_multiple_files=True)
if st.button('Pridect'):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            if uploaded_file.type.startswith('image/'):
                # Photo processing
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
                # Predict the class for the uploaded image
                with st.spinner('Please wait until ended classifying image...'):
                    decoded_image = decode_image(uploaded_file)
                    prediction = np.argmax(model.predict(decoded_image), axis=1)[0]
                    st.write(f'Prediction: {classes[prediction]}')
        
            elif uploaded_file.type.startswith('video/'):
                with st.spinner('Please wait until ended classifying video...'):
                    # Save the video locally
                    video_path = "uploaded_video.mp4"
                    with open(video_path, "wb") as video_file:
                        video_file.write(uploaded_file.read())
            
                    # Video processing
                    st.video(uploaded_file)
            
                    # Create a VideoCapture object
                    vidcap = cv2.VideoCapture(video_path)
            
                    # Lists to store timestamps and classes
                    timestamps = []
                    frame_classes = []
            
                    success, frame = vidcap.read()
                    count = 0
            
                    # Loop through video frames
                    while success:
                        # Preprocess the frame
                        processed_frame = preprocess_frame(frame)
            
                        # Get the timestamp of the frame
                        timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC)
                        timestamps.append(timestamp)
            
                        # Predict the class using your model
                        prediction = np.argmax(model.predict(np.expand_dims(processed_frame, axis=0)), axis=1)[0]
                        frame_classes.append(prediction)
                        
                        # Read the next frame
                        success, frame = vidcap.read()
                        count += 1
                    if 0 in frame_classes:
                            st.write(f'Prediction for video: {classes[0]}')
                    else :
                        st.write(f"Prediction for video: {classes[1]}")
                    
            
            else:
                st.warning("Please upload a valid image (jpg, jpeg, png) or video file (mp4, avi, mkv).")
