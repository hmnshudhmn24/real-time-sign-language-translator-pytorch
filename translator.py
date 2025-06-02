
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from collections import deque
from tkinter import *
from threading import Thread

# Define the CNN Model
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = SignLanguageCNN()
model_path = "sign_language_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Preprocess image
def preprocess_hand_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img

# Predict character
def predict(img):
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        return chr(predicted.item() + 65)

# GUI Setup
class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("800x600")
        self.root.configure(bg='white')

        self.label = Label(root, text="Predicted Sign:", font=("Helvetica", 20), bg='white')
        self.label.pack(pady=20)

        self.output = Label(root, text="", font=("Helvetica", 40), fg="blue", bg='white')
        self.output.pack(pady=20)

        self.start_button = Button(root, text="Start Camera", command=self.start_camera, font=("Helvetica", 16), bg='green', fg='white')
        self.start_button.pack(pady=20)

        self.stop_button = Button(root, text="Stop Camera", command=self.stop_camera, font=("Helvetica", 16), bg='red', fg='white')
        self.stop_button.pack(pady=10)

        self.stop_flag = False
        self.queue = deque(maxlen=15)

    def start_camera(self):
        self.stop_flag = False
        Thread(target=self.run_camera).start()

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 10
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 10
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 10
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 10

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size == 0:
                        continue

                    img_tensor = preprocess_hand_img(hand_img)
                    char = predict(img_tensor)
                    self.queue.append(char)

                    most_common = max(set(self.queue), key=self.queue.count)
                    self.output.config(text=most_common)

            cv2.imshow("Webcam - Sign Language", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop_camera(self):
        self.stop_flag = True

# Run the App
if __name__ == "__main__":
    root = Tk()
    app = TranslatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_camera)
    root.mainloop()
