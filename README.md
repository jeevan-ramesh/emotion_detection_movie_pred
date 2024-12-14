# emotion_detection_movie_pred
# emotion-detection-movie-recomendation
# Emotion Detection Using Deep Learning

This project focuses on classifying emotions using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The dataset consists of labeled images organized into training and testing directories. The project leverages GPU acceleration for efficient computations and includes a robust methodology for data preprocessing, model training, evaluation, and deployment.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset Structure](#dataset-structure)
5. [Model Architecture](#model-architecture)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)

---

## **Project Overview**
The goal of this project is to develop a machine learning model capable of accurately detecting emotions from image inputs. Emotions like happiness, sadness, anger, etc., are classified using a deep learning pipeline that emphasizes efficiency and accuracy.

---

## **Features**
- **Data Preprocessing:** Image normalization and augmentation.
- **Model Training:** CNN-based architecture optimized for emotion classification.
- **Evaluation:** Performance metrics including accuracy and confusion matrix.
- **Deployment Ready:** Model saved for integration into applications.

---

## **Technologies Used**
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow/Keras
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** OpenCV
- **GPU Acceleration:** Enabled via TensorFlow

---

## **Dataset Structure**
The dataset is organized into training and testing directories, with subdirectories for each emotion class:
```
/emotion_dataset/
  /train/
    /happy/
    /sad/
    /angry/
  /test/
    /happy/
    /sad/
    /angry/
```
- Images are labeled and stored in corresponding class directories.

---

## **Model Architecture**
    model = Sequential()

    # Convolutional and pooling layers

    model.add(Conv2D(32, (3, 3), strides=1, activation="relu",
                     padding="same", input_shape=(48, 48, 1)))

    model.add(Conv2D(64, (3, 3), strides=1, activation="relu",
                     padding="same"))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    # Convolutional and pooling layers

    model.add(Conv2D(128, (3, 3), strides=1, activation="relu",
                     padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    model.add(Conv2D(256, (3, 3), strides=1, activation="relu",
                     padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    # Flatten and dense layer

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    # Flatten and dense layer

    model.add(Dense(512, activation="relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    # Final layer

    model.add(Dense(7, activation="softmax"))

    # Compiler

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

---


---

## **Results**
- **Accuracy:** Achieved 78% accuracy on the test dataset.
- **Accuracy:** Achieved 96% accuracy on the train dataset

---

## **Future Improvements**
- Integrate real-time emotion detection using webcam input.
- Expand dataset to include more diverse emotion classes.
- Perform hyperparameter tuning for further optimization.

