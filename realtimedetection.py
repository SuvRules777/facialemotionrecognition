import cv2
from keras.models import model_from_json
import numpy as np
# from keras.preprocessing.image import load_img # Not needed for webcam frames
# --- Load Model ---
# Load the model architecture from JSON file
# USE THE CORRECT FILENAME HERE:
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the model weights
# USE THE CORRECT FILENAME HERE:
model.load_weights("emotiondetector.h5")

# Load the model weights
model.load_weights("emotiondetector.h5") # Make sure this filename matches your saved weights

# --- Load Haar Cascade for face detection ---
# Make sure the path to the haarcascade file is correct
# If haarcascade_frontalface_default.xml is not in the same directory as the script, provide the full path
try:
    # Try finding it in the cv2 data path first (common installation location)
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
except Exception as e:
    print(f"Error loading cascade file from cv2.data.haarcascades: {e}")
    # Fallback: Try loading from the current directory (if you copied it there)
    try:
        haar_file = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)
        if face_cascade.empty():
            raise IOError(f"Cannot load cascade classifier from {haar_file}")
    except Exception as fallback_e:
        print(f"Error loading cascade file from current directory: {fallback_e}")
        print("Please ensure 'haarcascade_frontalface_default.xml' is accessible.")
        exit() # Exit if cascade cannot be loaded

# --- Preprocessing function ---
def extract_features(image):
    """Preprocesses a grayscale image for the model."""
    feature = np.array(image)
    # Reshape to (1, height, width, channels) - model expects 48x48 grayscale
    feature = feature.reshape(1, 48, 48, 1)
    # Normalize (scale pixel values to 0-1)
    return feature / 255.0

# --- Initialize Webcam ---
webcam = cv2.VideoCapture(0)  # 0 is usually the default webcam
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Labels dictionary ---
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# --- Real-time Detection Loop ---
while True:
    # Read frame from webcam
    ret, im = webcam.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale (Haar cascade works better on grayscale)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # Adjust scaleFactor and minNeighbors for different detection sensitivity/performance
    # scaleFactor: How much the image size is reduced at each image scale. > 1 (e.g., 1.3 means reduce by 30%)
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it. Higher value = fewer detections but higher quality.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    try:
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region of interest (ROI) from the grayscale image
            face_roi_gray = gray[y:y+h, x:x+w]

            # Resize the ROI to the size expected by the model (48x48)
            resized_roi = cv2.resize(face_roi_gray, (48, 48))

            # Preprocess the resized ROI
            img_features = extract_features(resized_roi)

            # Make prediction using the loaded model
            prediction = model.predict(img_features)

            # Find the index with the highest prediction score
            predicted_label_index = np.argmax(prediction)

            # Get the corresponding emotion label string
            predicted_emotion = labels[predicted_label_index]

            # Draw a rectangle around the detected face on the original color image
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red rectangle

            # Put the predicted emotion text above the rectangle
            cv2.putText(im, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame (with rectangles and labels for easier display)
        cv2.imshow("Real-time Emotion Detection", im)   

    except Exception as e:
        print(f"Error during face processing or prediction: {e}")

    # --- Exit Condition ---
    # Wait for 1 millisecond and check if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
webcam.release()
cv2.destroyAllWindows()
print("Webcam released and windows closed.")

