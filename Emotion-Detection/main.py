# main.py

from emotion_model import load_model, predict_emotion, get_emotion_label

# Define the emotion dictionary
emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise', 6: 'happy'}

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model()

    # Path to the image you want to analyze
    img_path = 'C:/BIGDATA/EMOTION-DETECTION/image.jpg'  # Use forward slashes or raw string for Windows path

    # Predict emotions in the image
    preds = predict_emotion(model, img_path)

    # Get the emotion label
    emotion_label = get_emotion_label(preds, emotion_dict)
    print(f"The predicted emotion in the image is: {emotion_label}")
