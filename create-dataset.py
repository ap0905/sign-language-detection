import os
import pickle
import mediapipe as mp
import cv2

class MyHands:
    def __init__(self, max_hands, detection_con, track_con):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Initialize Hands object with custom parameters
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

# Define the directory containing the data
DATA_DIR = './data'

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lists to store processed data and labels
data = []
labels = []

# Initialize Hands object with custom parameters
my_hands = MyHands(max_hands=2, detection_con=0.3, track_con=0.5)

# Iterate through directories and images in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Ignore non-directory files (e.g., .DS_Store)
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        # Lists to store hand landmarks
        x_ = []
        y_ = []

        # Check if the file is an image
        if not img_path.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))

        # Check if image is read successfully
        if img is None:
            print(f"Error reading image: {os.path.join(DATA_DIR, dir_, img_path)}")
            continue

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the Hands model
        results = my_hands.hands.process(img_rgb)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y

                    x_.append(x)
                    y_.append(y)

            # Normalize hand landmarks and append to data_aux list
            for i in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[i]
                for j in range(len(hand_landmarks.landmark)):
                    x_normalized = hand_landmarks.landmark[j].x - min(x_)
                    y_normalized = hand_landmarks.landmark[j].y - min(y_)
                    data_aux.extend([x_normalized, y_normalized])

            # Append data_aux to data list and label to labels list
            data.append(data_aux)
            labels.append(dir_)

# Save the processed data and labels
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
