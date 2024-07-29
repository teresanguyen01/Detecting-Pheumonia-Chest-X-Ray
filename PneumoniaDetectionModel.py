import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.image as mpimg

warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model, image_dataset_from_directory, load_img, img_to_array

path = 'chest_xray/chest_xray/train'
classes = os.listdir(path)
print(classes)

PNEUMONIA_dir = os.path.join(path, classes[1])
NORMAL_dir = os.path.join(path, classes[2])

pneumonia_names = os.listdir(PNEUMONIA_dir)
normal_names = os.listdir(NORMAL_dir)

print('There are ', len(pneumonia_names), 'images of pneumonia infected in training dataset')
print('There are ', len(normal_names), 'normal images in training dataset')

fig = plt.gcf()
fig.set_size_inches(16, 8)

# Select the starting index for the images to display
pic_index = 210

# Create lists of the file paths for the 8 images to display
pneumonia_images = [os.path.join(PNEUMONIA_dir, fname) for fname in pneumonia_names[pic_index-8:pic_index]]
# Loop through the image paths and display each image in a subplot
for i, img_path in enumerate(pneumonia_images):
    sp = plt.subplot(2, 4, i+1)
    sp.axis('Off')
    # Read in the image using Matplotlib's imread() function
    img = mpimg.imread(img_path)
    plt.imshow(img)

# Display the plot with the 16 images in a 4x4 grid
plt.show()

# Set the figure size
fig = plt.gcf()
fig.set_size_inches(16, 8)

# Create lists of the file paths for the 8 images to display
normal_images = [os.path.join(NORMAL_dir, fname) for fname in normal_names[pic_index-8:pic_index]]
# Loop through the image paths and display each image in a subplot
for i, img_path in enumerate(normal_images):
    sp = plt.subplot(2, 4, i+1)
    sp.axis('Off')
    # Read in the image using Matplotlib's imread() function
    img = mpimg.imread(img_path)
    plt.imshow(img)

# Display the plot with the 16 images in a 4x4 grid
plt.show()

Train = image_dataset_from_directory(
    directory='chest_xray/chest_xray/train',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256)
)
Test = image_dataset_from_directory(
    directory='chest_xray/chest_xray/test',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256)
)
Validation = image_dataset_from_directory(
    directory='chest_xray/chest_xray/val',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256)
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    layers.BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    layers.BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    layers.BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    layers.BatchNormalization(),
    Dense(2, activation='sigmoid')
])

model.summary()

# Plot the keras model
plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    Train,
    epochs=10,
    validation_data=Validation
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

loss, accuracy = model.evaluate(Test)
print('The accuracy of the model on test dataset is', np.round(accuracy*100))

# Load the image from the directory with the target size of (256, 256)
test_image = load_img(
    "chest_xray/chest_xray/test/NORMAL/IM-0010-0001.jpeg",
    target_size=(256, 256)
)

# Display the loaded image
plt.imshow(test_image)

# Convert the loaded image into a NumPy array and expand its dimensions to match the expected input shape of the model
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Use the trained model to make a prediction on the input image
result = model.predict(test_image)

# Extract the probability of the input image belonging to each class from the prediction result
class_probabilities = result[0]

# Determine the class with the highest probability and print its label
if class_probabilities[0] > class_probabilities[1]:
    print("Normal")
else:
    print("Pneumonia")

# Load another image from the directory with the target size of (256, 256)
test_image = load_img(
    "chest_xray/chest_xray/test/PNEUMONIA/person85_bacteria_417.jpeg",
    target_size=(256, 256)
)

# Display the loaded image
plt.imshow(test_image)

# Convert the loaded image into a NumPy array and expand its dimensions to match the expected input shape of the model
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Use the trained model to make a prediction on the input image
result = model.predict(test_image)

# Extract the probability of the input image belonging to each class from the prediction result
class_probabilities = result[0]

# Determine the class with the highest probability and print its label
if class_probabilities[0] > class_probabilities[1]:
    print("Normal")
else:
    print("Pneumonia")

