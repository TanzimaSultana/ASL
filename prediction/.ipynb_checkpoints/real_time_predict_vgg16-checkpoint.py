import os
import cv2
import numpy as np
from tensorflow import keras
from collections import deque, Counter

# Load the trained VGG16 model
model_path = os.path.abspath("../model/vgg16.keras")
model = keras.models.load_model(model_path)

# Define label mapping
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
reverse_map = {i: label for i, label in enumerate(labels)}

# Preprocessing for VGG16 — RGB, resized, normalized
def preprocess_roi(roi):
    roi_resized = cv2.resize(roi, (96, 96))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = roi_normalized.reshape(1, 96, 96, 3)
    return roi_input, roi_resized  # return resized for optional display

# Initialize deque for smoothing
recent_preds = deque(maxlen=5)
last_saved_label = None

# Optional save directory (if you want to capture images later)
save_dir = "./real_time_captured_images/vgg16/"
os.makedirs(save_dir, exist_ok=True)
frame_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit the webcam stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x, y, w, h = 100, 100, 450, 450
    roi = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    roi_input, roi_resized = preprocess_roi(roi)

    prediction = model.predict(roi_input, verbose=0)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    recent_preds.append(pred_class)
    most_common = Counter(recent_preds).most_common(1)[0][0]
    pred_label = reverse_map[most_common]

    cv2.putText(frame, f"Predicted: {pred_label} ({confidence:.2f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Alphabet Recognition (VGG16)", frame)
    # cv2.imshow("ROI", roi_resized)

    # Example: Save only if prediction changes
    if pred_label != last_saved_label:
         save_path = os.path.join(save_dir, f"{pred_label}_{frame_counter:06d}.png")
         cv2.imwrite(save_path, cv2.resize(roi_resized, (256, 256)))
         frame_counter += 1
         last_saved_label = pred_label

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam stream ended.")
