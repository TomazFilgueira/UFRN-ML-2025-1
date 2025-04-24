
# ❤️ Heart Disease Prediction Project

This project focuses on predicting heart disease using machine learning models. It includes data cleaning, exploratory data analysis (EDA), feature importance analysis, model selection, parameter tuning, and deployment via a web service. The solution is designed for effective containerization and deployment.

---

## 🗂️ Table of Contents
1. [📌 Project Overview](#-project-overview)
2. [📁 Directory Structure](#-directory-structure)
3. [❓ Problem Description](#-problem-description)


---

## 📌 Project Overview

Heart disease remains one of the leading causes of death globally. This project uses machine learning techniques to predict the likelihood of heart disease based on patient data. 

Key features include:
- 🧹 Data preparation and cleaning.  
- 🔍 Exploratory Data Analysis (EDA) to uncover patterns and relationships.  
- 🧠 Model training, evaluation, and parameter optimization.  
- 🌐 Deployment via Flask and containerization using Docker for scalable web service hosting. 
- ☁️ Cloud deployment using AWS Elastic Beanstalk.  

---

## 📁 Directory Structure

```plaintext
Heart-Disease-App/
│
├── data/                          # Contains the dataset
├── train.py                       # Script for training and saving the model
├── predict.py                     # Web service for serving the model
└── README.md                      # Project description and instructions
```

---

## ❓ Problem Description

**Cardiovascular diseases** are a major global health challenge. This project aims to use machine learning to:
- ⚠️ Identify individuals at risk of heart disease.  
- 🩺 Assist healthcare professionals in making more informed decisions.  
- 🌍 Provide an easily deployable service for real-world applications.  

### Heart Disease Prediction Dataset 📊
This dataset was cloned from Maxim-eyengue github. He accquired the data from Kaggle community below.

[The dataset](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset/data) combines five publicly available heart disease datasets, with a total of 2181  records:

<ul>
    <li> 📝 Heart Attack Analysis & Prediction Dataset: 304 records from Rahman, 2021</li>
    <li> 📝 Heart Disease Dataset: 1,026 records from Lapp, 2019</li>
    <li> 📝 Heart Attack Prediction (Dataset 3): 295 records from Damarla, 2020</li>
    <li> 📝 Heart Attack Prediction (Dataset 4): 271 records from Anand, 2018</li>
    <li> 📝 Heart CSV Dataset: 290 records from Nandal, 2022</li>
</ul>

Merging these data sets provides a more robust foundation for training machine learning models aimed at early detection and prevention of heart disease. 

The [resulting dataset](/data/raw_merged_heart_dataset.csv) contains anonymized patient records with various features, such as age, cholesterol levels, and blood pressure, which are crucial for predicting heart attack and stroke risks, covering both medical and demographic factors.

### Features Description:
<ul>
    <li><strong>age</strong>: age of the patient 
        [years: Numeric]</li>
    <li><strong>sex</strong>: gender of the patient 
        [1: Male, 0: Female]</li>
    <li><strong>cp</strong>: chest pain type 
        [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]</li>
    <li><strong>trestbps</strong>: resting blood pressure 
        [mm Hg: Numeric]</li>
    <li><strong>chol</strong>: serum cholesterol level 
        [mg/dl: Numeric]</li>
    <li><strong>fbs</strong>: fasting blood sugar 
        [1: if fasting blood sugar > 120 mg/dl, 0: otherwise]</li>
    <li><strong>restecg</strong>: resting electrocardiographic results 
        [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria]</li>
    <li><strong>thalach</strong>: maximum heart rate achieved 
        [Numeric value between 60 and 202]</li>
    <li><strong>exang</strong>: exercise-induced angina 
        [1: Yes, 0: No]</li>
    <li><strong>oldpeak</strong>: ST depression induced by exercise relative to rest 
        [Numeric value measured in depression]</li>
    <li><strong>slope</strong>: slope of the peak exercise ST segment 
        [0: Upsloping, 1: Flat, 2: Downsloping]</li>
    <li><strong>ca</strong>: number (0-3) of major vessels (arteries, veins, and capillaries) colored by fluoroscopy 
        [0, 1, 2, 3] </li>
    <li><strong>thal</strong>: Thalassemia types 
        [1: Normal, 2: Fixed defect, 3: Reversible defect]</li>
    <li><strong>target</strong>: outcome variable for heart attack risk 
        [1: disease or more chance of heart attack, 0: normal or less chance of heart attack]</li>
</ul>

