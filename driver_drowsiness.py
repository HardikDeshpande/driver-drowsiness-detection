import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Parameters
IMG_SIZE = 64
dataset_path = 'OACE'  # Use your new dataset

# Prepare dataset
X = []
y = []

categories = ['close', 'open']

for idx, category in enumerate(categories):
    category_folder = os.path.join(dataset_path, category)
    for img_name in os.listdir(category_folder):
        img_path = os.path.join(category_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)

X = np.array(X) / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
print("🛠️ Training Drowsiness Detection Model...")
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('drowsiness_model.h5')
print("✅ New drowsiness_model.h5 saved successfully!")
