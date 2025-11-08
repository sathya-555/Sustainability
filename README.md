# Week1
Plant Disease Detection using CNN (Sustainability Theme)
# 🌿 Plant Disease Detection using CNN (AICTE Internship – Sustainability Theme)

## 🧩 Problem Statement
To detect tomato plant leaf diseases using Artificial Intelligence and Convolutional Neural Networks (CNN).  
This project aims to support **Sustainable Agriculture** by helping farmers identify diseases early and reduce pesticide use.

---

## 🎯 Objective
- Predict whether a plant leaf is **Healthy** or **Diseased** using a CNN model.  
- Support sustainable farming by minimizing chemical usage.  
- Build an AI system that helps farmers take preventive actions.

---

## 📊 Dataset Details
- **Dataset Name:** [PlantVillage Dataset (Tomato subset)](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Type:** Image dataset (Tomato leaf images)  
- **Classes:** Healthy + multiple tomato leaf disease types  
- **Source:** Kaggle  
- **Data Format:** `.jpg` image files  

---

## 🧠 Model Information
- **Model Type:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  
- **Environment:** Google Colab  
- **Model Output:** Predicts the disease category of the tomato leaf  
- **Saved Model File:** `plant_disease_model.h5`  

**Model Architecture (Basic CNN):**
1. Convolutional + ReLU layers  
2. MaxPooling layers  
3. Dense + Dropout layers  
4. Softmax output layer  

---

## ⚙️ Steps Followed
1️⃣ Finalized the problem statement under the **Sustainability theme**  
2️⃣ Collected the dataset from **Kaggle (PlantVillage)**  
3️⃣ Preprocessed the dataset and created training/validation splits  
4️⃣ Built and trained a CNN model using TensorFlow/Keras  
5️⃣ Evaluated the model (Accuracy, Precision, Recall, F1-score)  
6️⃣ Saved the trained model as `.h5` file for future use  
7️⃣ Tested the model on new leaf images  

---

## 📈 Evaluation Metrics
| Metric | Description |
|---------|--------------|
| **Accuracy** | Percentage of correct predictions |
| **Precision** | How many predicted positives are actually true |
| **Recall** | How many actual positives are correctly predicted |
| **F1 Score** | Harmonic mean of Precision and Recall |

🧪 Example results after training (for 10–12 epochs):  
✅ Accuracy: ~90%  
✅ F1-score: ~0.88 (depends on dataset size and training time)

---

## 💾 Files in this Repository
| File Name | Description |
|------------|-------------|
| `plant_disease_tomato_thanglish.ipynb` | Main project notebook (Google Colab–ready) |
| `plant_disease_model.h5` | Trained CNN model (generated after running notebook) |
| `README.md` | Project description and documentation |

---

## 🛠️ Tools and Technologies Used
- 🐍 Python  
- 💻 Google Colab  
- 🧠 TensorFlow / Keras  
- 📦 Kaggle API  
- 📊 Matplotlib, NumPy, Pandas  

---

## 🌱 Sustainability Impact
This project promotes sustainable agriculture by:  
- Helping farmers detect plant diseases early.  
- Reducing unnecessary pesticide usage.  
- Increasing crop yield and protecting the environment.  

It supports the **UN Sustainable Development Goals (SDG 2 – Zero Hunger)** and **SDG 12 – Responsible Consumption and Production**.

---

## 🚀 Future Improvements
- Use **Transfer Learning** (MobileNet, EfficientNet) to improve accuracy.  
- Develop a **Streamlit web UI** for easy image upload and detection.  
- Deploy the model as a web or mobile app for farmers.

## 🗓️ Week 2 Progress
- Implemented CNN model using TensorFlow/Keras
- Trained sample dataset to demonstrate model working
- Model achieved ~85–90% accuracy
- Saved model as `plant_disease_model.h5`
- Will replace sample data with PlantVillage dataset in Week 3
