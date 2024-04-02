import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def analyze_video(video_path=r"C:\Users\Spm\Desktop\py\ddd.mp4"):
    """
    Analyzes a video of pistol shooting to check body movement, shoulder position, right hand speed, and wrist movement using MediaPipe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        None
    """
    # Initialize MediaPipe Hands and Pose models
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # Initialize VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    # Initialize MediaPipe Hands and Pose instances
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Loop through each frame in the video
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results_hands = hands.process(rgb_frame)

        # Detect poses
        results_pose = pose.process(rgb_frame)

        # Draw hand landmarks and pose landmarks on the frame
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results_pose.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    

def train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Define the RNN model architecture
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    loss, mae = model.evaluate(X_test, y_test)

    print("Test Loss:", loss)
    print("Test MAE:", mae)

    # Make predictions
    predictions = model.predict(X_test)

def analyze_shooting_technique(frame):
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.h5')  # Replace 'trained_model.h5' with the path to your trained model

    # Function to preprocess input data
    def preprocess_data(frame):
        # Preprocess frame (resize, normalize, etc.)
        # Example: 
        preprocessed_frame = cv2.resize(frame, (input_width, input_height))
        preprocessed_frame = preprocessed_frame / 255.0  # Normalize pixel values
        return preprocessed_frame

    # Preprocess input data
    preprocessed_frame = preprocess_data(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Perform inference
    prediction = model.predict(preprocessed_frame)

    # Process prediction (e.g., analyze shooting technique)
    # Example:
    if prediction >= 0.5:
        return "Good form"
    else:
        return "Needs improvement"

def display_recommendations(frame, recommendation):
    # Display recommendation text on the frame
    cv2.putText(frame, recommendation, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Recommendations', frame)

def capture_shooter_movements():
    # Simulate capturing shooter's movements from sensors or cameras
    # This function would typically capture data streams in real-time
    return {"shoulder_position": "Good", "hand_position": "Needs improvement", "wrist_position": "Good"}

def generate_recommendations(movements):
    # Generate recommendations based on shooter's movements
    recommendations = []
    if movements["hand_position"] == "Needs improvement":
        recommendations.append("Adjust hand position for better stability")
    # Add more recommendations based on other movements
    return recommendations

def provide_feedback(recommendations):
    # Provide feedback to the shooter based on recommendations
    print("Recommendations for improvement:")
    for recommendation in recommendations:
        print("-", recommendation)

def monitor_performance():
    # Continuously monitor the system's performance and make necessary adjustments
    while True:
        shooter_movements = capture_shooter_movements()
        recommendations = generate_recommendations(shooter_movements)
        provide_feedback(recommendations)
        time.sleep(5)  # Adjust the interval as needed
