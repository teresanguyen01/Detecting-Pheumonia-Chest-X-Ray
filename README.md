# Chest X-Ray Pneumonia Detection

This project involves building a convolutional neural network (CNN) model to classify chest X-ray images as either normal or pneumonia-infected. The model is trained using TensorFlow and Keras, and includes functionalities for displaying sample images, training the model, and predicting classes for new images uploaded by the user.

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install the required libraries:**
    ```bash
    pip install tensorflow pandas matplotlib numpy
    ```

3. **Ensure the dataset is structured as follows:**
    ```
    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/
    ```

## Usage

1. **Run the script:**
    ```bash
    python chest_xray_pneumonia_detection.py
    ```

2. **View sample images:**
    The script will display a few sample images from the training dataset.

3. **Train the model:**
    The CNN model will be trained using the training dataset. Training progress, including loss and accuracy, will be displayed in real-time.

4. **Evaluate the model:**
    After training, the model's accuracy on the test dataset will be displayed.

5. **Predict new images:**
    The script allows you to input the path of an image for classification. To do this:
    - Enter the full path to the image you want to predict.
    - If the path is valid, the image will be displayed along with the prediction result (Normal or Pneumonia).
    - Type "exit" to stop the prediction loop and end the program.

## Example

1. **Sample Output:**

    ```
    ['NORMAL', 'PNEUMONIA']
    There are  3875 images of pneumonia infected in training dataset
    There are  1341 normal images in training dataset
    ...
    The accuracy of the model on test dataset is 96.0
    Enter the path of the image to predict (type 'exit' to stop): path/to/your/image.jpeg
    Prediction: Normal
    ```

2. **Prediction Example:**

    ```
    Enter the path of the image to predict (type 'exit' to stop): chest_xray/chest_xray/test/NORMAL/IM-0010-0001.jpeg
    ```

    The script will display the image and print whether it is classified as "Normal" or "Pneumonia".

## Notes

- Ensure the dataset paths are correct and the images are properly organized in the respective directories.
- The script expects the images to be in JPEG format.
- Adjust the `pic_index` variable if you want to display different sample images from the training dataset.

## Datasets and Libraries

- The dataset used in this project is publicly available and can be found at [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- TensorFlow and Keras libraries are used for building and training the CNN model.
