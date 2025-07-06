import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = 64
dataset_path = 'dataset'

# Prepare dataset
X = []
y = []

# ✅ Only folders (no .DS_Store or files)
class_names = sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])

print(f"✅ Classes found: {class_names}")

for idx, gesture in enumerate(class_names):
    gesture_folder = os.path.join(dataset_path, gesture)
    for img_name in os.listdir(gesture_folder):
        img_path = os.path.join(gesture_folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)

X = np.array(X) / 255.0
y = np.array(y)
y = to_categorical(y, num_classes=len(class_names))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("🛠 Training Gesture Model...")
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('gesture_model.h5')
print("✅ Gesture Model Saved as gesture_model.h5")
