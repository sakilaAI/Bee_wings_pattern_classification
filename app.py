import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  
import os
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# App title
st.title("🐝 Bee Wing Pattern Classification")
st.write("A CNN-based deep learning model for classifying bee wing patterns.")

# Load and preprocess datasets
@st.cache_resource
def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    training_set = train_datagen.flow_from_directory(
        'training_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_set = test_datagen.flow_from_directory(
        'test_set', target_size=(64, 64), batch_size=32, class_mode='categorical', shuffle=False)

    return training_set, test_set

training_set, test_set = load_data()
class_labels = list(training_set.class_indices.keys())

# Build CNN model
@st.cache_resource
def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(96, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(len(class_labels), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train model
with st.spinner("Training model..."):
    history = model.fit(
        training_set,
        validation_data=test_set,
        epochs=10  # You can increase this, but keep it low for fast demo
    )

# Accuracy and Loss plots
st.subheader("📈 Model Performance")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title("Accuracy")
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title("Loss")
ax2.legend()
st.pyplot(fig)

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_set.classes
cm = confusion_matrix(y_true, y_pred)

fig_cm = plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig_cm)

# Classification Report
st.subheader("📃 Classification Report")
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Upload & predict a new image
st.subheader("📷 Test a New Bee Wing Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(64, 64))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"Predicted Class: **{predicted_class}**")
