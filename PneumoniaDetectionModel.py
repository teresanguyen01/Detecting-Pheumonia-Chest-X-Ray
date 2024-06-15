"""
This file contains the code for the Pneumonia Detection Model.
The model is built using the VGG16 pre-trained model and fine-tuned on the Chest X-ray dataset.
The model is trained and evaluated on the dataset and the performance metrics are displayed.
"""

# Importing the required libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, f1_score, accuracy_score
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

# Define paths
train_path = '/Users/teresanguyen/Documents/DetectingPheumoniaX-Ray/chest_xray/train'
test_path = '/Users/teresanguyen/Documents/DetectingPheumoniaX-Ray/chest_xray/test'
val_path = '/Users/teresanguyen/Documents/DetectingPheumoniaX-Ray/chest_xray/val'
weights_path = '/Users/teresanguyen/Documents/DetectingPheumoniaX-Ray/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Function to load and preprocess images
def get_data(data_dir):
    data = []
    labels = []
    for label in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(data_dir, label)
        class_num = 0 if label == 'NORMAL' else 1
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is not None:  # Ensure the image is loaded properly
                    resized_arr = cv2.resize(img_arr, (224, 224))  # Reshape images to preferred size
                    data.append(resized_arr)
                    labels.append(class_num)
                else:
                    print(f"Failed to load image: {os.path.join(path, img)}")
            except Exception as e:
                print(f"Error loading image: {os.path.join(path, img)} - {e}")
    return np.array(data), np.array(labels)

# Load and preprocess the data
X_train, y_train = get_data(train_path)
X_train, y_train = shuffle(X_train, y_train, random_state=42)  # Shuffle the data
X_train = X_train[:624] 
X_train = X_train.reshape(-1, 224, 224, 1)
X_train_rgb = np.repeat(X_train, 3, axis=3)
y_train = y_train[:624]

X_val, y_val = get_data(val_path)
X_val = np.repeat(X_val, 3, axis=-1)
X_test, y_test = get_data(test_path)
X_test = np.repeat(X_test, 3, axis=-1)

# Normalize pixel values
X_val = X_val.reshape(-1, 224, 224, 1) / 255.0
X_test = X_test.reshape(-1, 224, 224, 1) / 255.0
X_train_rgb = X_train_rgb / 255.0

# Check the shape of the data
print(f'Train set shape: {X_train_rgb.shape}')
print(f'Validation set shape: {X_val.shape}')
print(f'Test set shape: {X_test.shape}')

# Build the model using VGG16
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights=None)
base_model.load_weights(weights_path)  # Load weights from the local file

x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification

# Combine the base model and custom layers
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Train the model
history = model.fit(datagen.flow(X_train_rgb, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('pneumonia_detection_model.h5')

# Plot accuracy and loss curves
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.show()

plot_history(history)

# Fine-tune the model
for layer in base_model.layers:
    layer.trainable = True

# Re-compile the model with a lower learning rate
model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training
history_fine = model.fit(datagen.flow(X_train_rgb, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Evaluate the fine-tuned model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy after fine-tuning: {accuracy * 100:.2f}%')

# Save the fine-tuned model
model.save('pneumonia_detection_model_fine_tuned.h5')

# Predictions
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Display the confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display the classification report
print(classification_report(y_test, predictions))

# Display the ROC curve
fpr, tpr, _ = roc_curve(y_test, predictions)
auc = roc_auc_score(y_test, predictions)

plt.plot(fpr, tpr, label=f'AUC: {auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Display the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, predictions)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Calculate and display additional metrics
f1 = f1_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'ROC AUC Score: {roc_auc}')

