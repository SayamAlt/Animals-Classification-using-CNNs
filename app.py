import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

st.title("Animal Classification")

# Load the saved model
model = tf.keras.models.load_model("animal_classifier.keras")

# Define the class labels
classes = ['cavallo',
 'pecora',
 'elefante',
 'gatto',
 'scoiattolo',
 'gallina',
 'ragno',
 'mucca',
 'cane',
 'farfalla']

label_mapping = {i: label for i, label in enumerate(classes)}

# Upload an image file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

predict_button = st.button("Predict")

if predict_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make a prediction
        predicted_probabilities = model.predict(image_array)
        predicted_class = label_mapping[np.argmax(predicted_probabilities, axis=1)[0]]
        
        # Display the predicted class
        st.write(f"Predicted class: {predicted_class}")
    else:
        st.info("Please upload an image file.")