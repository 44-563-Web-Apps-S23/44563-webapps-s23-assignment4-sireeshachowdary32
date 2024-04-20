# emotion_model.py

import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

# Load the pre-trained VGG16 model
def load_model():
    model = VGG16(weights='imagenet', include_top=True)
    return model

# Preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Predict the emotion in the image
def predict_emotion(model, img_path):
    x = preprocess_image(img_path)
    preds = model.predict(x)
    return preds

# Get the emotion label
def get_emotion_label(preds, emotion_dict):
    emotion_id = np.argmax(preds)
    if emotion_id in emotion_dict:
        emotion_label = emotion_dict[emotion_id]
    else:
        emotion_label = "Happy"
    return emotion_label
