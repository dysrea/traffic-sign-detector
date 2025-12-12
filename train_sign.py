import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- SETTINGS ---
path = 'Images'
imgDimensions = (64, 64, 3) # px size and 3 channels (RGB)
epochsVal = 15
batchSizeVal = 50

# Load Data
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Importing Classes...")

# Loader
for x in myList:
    if not os.path.isdir(path + "/" + x): continue
    myPicList = os.listdir(path + "/" + x)
    for y in myPicList:
        try:
            curImg = cv2.imread(path + "/" + x + "/" + y)
            curImg = cv2.resize(curImg, (imgDimensions[0], imgDimensions[1]))
            images.append(curImg)
            classNo.append(int(x))
        except:
            pass
    print(f"{x}", end=" ")
print("\nDone.")

images = np.array(images)
classNo = np.array(classNo)

# Split & pre process 
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0
X_validation = X_validation / 255.0

noOfClasses = max(classNo) + 1
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Data Augmentation 
dataGen = ImageDataGenerator(
    width_shift_range=0.1,   # Shift image left/right
    height_shift_range=0.1,  # Shift image up/down
    zoom_range=0.2,          # Zoom in/out
    shear_range=0.1,         # Tilt the sign
    rotation_range=10        # Rotate slightly
)
dataGen.fit(X_train)

# Build Model
model = Sequential()

# Layer 1
model.add(Conv2D(60, (5, 5), input_shape=imgDimensions, activation='relu'))
model.add(Conv2D(60, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization()) # Helps stabilize training

# Layer 2
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization()) 
model.add(Dropout(0.5))

# Classification
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(noOfClasses, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN
print("Starting Robust Training...")
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                    epochs=epochsVal,
                    validation_data=(X_validation, y_validation),
                    shuffle=1)

model.save("traffic_model.keras")
print("Model Saved!")