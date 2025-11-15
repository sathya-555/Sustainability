# 🌿 **Tomato Leaf Disease Detection Using Deep Learning**

This project is developed as part of the **AICTE Virtual Internship – Sustainable Development Theme**.
The aim is to classify tomato leaves as **Healthy** or **Diseased** using a deep learning model.



## 📌 **Project Overview**

Tomato plants often suffer from fungal and bacterial diseases which impact crop yield.
Early detection helps farmers take timely protective actions.

This project uses a **Convolutional Neural Network (CNN)** / **MobileNetV2** model to classify tomato leaf images into:

* ✅ Healthy
* ❌ Diseased

The model is trained using a small custom dataset and tested on new images.

---

## 🎯 **Learning Objectives**

* Understand image classification using deep learning
* Learn dataset preprocessing & augmentation
* Train a CNN model using TensorFlow/Keras
* Evaluate model accuracy and make predictions
* Save the trained model for deployment

---

## 🛠 **Tools & Technologies Used**

| Category                 | Tools                                              |
| ------------------------ | -------------------------------------------------- |
| **Programming Language** | Python                                             |
| **Frameworks**           | TensorFlow, Keras                                  |
| **Libraries**            | NumPy, Matplotlib                                  |
| **IDE / Platform**       | Google Colab                                       |
| **Version Control**      | GitHub                                             |
| **Dataset**              | Custom dataset (Healthy vs Diseased tomato leaves) |

---

## 🧪 **Dataset Details**

A small manually created dataset:

* **Healthy** leaf images
* **Diseased** leaf images

Images are resized to **128 × 128** and normalized during training.

Folder structure:

```
dataset/
 ├── healthy/
 ├── diseased/
```

---

## 🔍 **Methodology**

### 1️⃣ Data Collection

Collected tomato leaf images from Google and arranged into folders.

### 2️⃣ Data Preprocessing

* Resized images
* Normalized pixel values
* Applied augmentation (rotation, flip, zoom)

### 3️⃣ Model Building

A simple CNN / MobileNetV2 model was created:

* Conv2D layers
* MaxPooling
* Flatten
* Dense layers
* Softmax output

### 4️⃣ Model Training

* Train–validation split (80/20)
* Optimizer: Adam
* Loss: Categorical Crossentropy
* Metrics: Accuracy

### 5️⃣ Evaluation

Plotted training & validation accuracy and loss.

### 6️⃣ Prediction

Given a test image, the model predicts:

```
Leaf Status: Healthy / Diseased
```

The leaf image is also displayed.

---

## 🧾 **How to Run the Project**

### ✔ Step 1 — Upload Dataset

Upload `dataset/healthy` and `dataset/diseased` folders in Colab.

### ✔ Step 2 — Run Training Code

Execute the notebook cells to train the model.

### ✔ Step 3 — Save Model

```
model.save("improved_tomato_model.h5")
```

### ✔ Step 4 — Test Prediction

Upload a test leaf image and run the prediction cell.

---

## 📸 Sample Output

* Displays the input leaf image
* Shows predicted label
* Shows confidence percentage

Example:

```
Predicted Class: Healthy
```

---

## 🧩 **Problem Statement**

Manual detection of plant diseases is slow, subjective, and requires expert knowledge.
There is a need for an automated AI-based system to classify tomato leaf diseases accurately.

---

## 💡 **Proposed Solution**

A deep learning–based CNN model is developed that:

* Processes tomato leaf images
* Classifies them as **Healthy** or **Diseased**
* Can be extended to multiple plant species
* Helps farmers with early disease identification

---

## 🏁 **Conclusion**

* The model successfully classifies tomato leaves using deep learning.
* Helps promote **sustainable agriculture** by reducing crop loss.
* Can be enhanced by using larger datasets and adding more disease classes.

---

