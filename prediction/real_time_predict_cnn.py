import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter
from tensorflow import keras

# Load the trained model
model_path = os.path.abspath("../model/cnn.keras")
model = keras.models.load_model(model_path)

# Define label mapping
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
reverse_map = {i: label for i, label in enumerate(labels)}

def preprocess_roi(roi):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    roi_resized = cv2.resize(roi_blur, (64, 64))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = roi_normalized.reshape(1, 64, 64, 1)
    return roi_input, roi_resized

recent_preds = deque(maxlen=5)
last_saved_label = None

save_dir = "./real_time_captured_images/cnn/"
os.makedirs(save_dir, exist_ok=True)
frame_counter = 0

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

    if pred_label != last_saved_label:
        # Upscale the ROI image for saving (e.g., 256×256)
        roi_save = cv2.resize(roi_resized, (256, 256), interpolation=cv2.INTER_NEAREST)
        save_path = os.path.join(save_dir, f"{pred_label}_{frame_counter:06d}.png")
        cv2.imwrite(save_path, roi_save)
        frame_counter += 1
        last_saved_label = pred_label

    cv2.putText(frame, f"Predicted: {pred_label} ({confidence:.2f})",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Alphabet Recognition (CNN)", frame)
    cv2.imshow("ROI", roi_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam stream ended.")