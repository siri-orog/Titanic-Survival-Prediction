# Titanic Survival Prediction

Titanic Survival Prediction is a machine learning–based classification project that predicts whether a passenger survived the Titanic disaster based on personal, socio-economic, and travel-related features.

This project demonstrates an end-to-end machine learning pipeline including data preprocessing, feature engineering, model training, evaluation, and deployment readiness.

---

## Table of Contents

1. Project Overview  
2. Purpose of the Project  
3. Project Development Workflow  
4. Tools & Technologies Used  
5. Dataset Description  
6. Data Preprocessing  
7. Machine Learning Algorithms Used  
8. Model Evaluation  
9. Challenges Faced  
10. How to Run the Project  
11. How to Update the Project  
12. Future Improvements  

---

## Project Overview

The Titanic disaster dataset is one of the most popular real-world datasets used to understand **binary classification problems**.  
This project predicts passenger survival (`Survived = 0 or 1`) using machine learning techniques based on features such as age, gender, class, and fare.

---

## Purpose of the Project

The main objectives of this project are:

- To understand real-world classification problems
- To learn data preprocessing and feature engineering
- To compare multiple machine learning models
- To evaluate models using standard metrics
- To build a clean and reproducible ML project structure

---

## Project Development Workflow

The project was developed following these steps:

1. Understanding the problem statement  
2. Dataset exploration and analysis  
3. Handling missing and inconsistent data  
4. Feature encoding and scaling  
5. Model selection and training  
6. Performance evaluation  
7. Preparing the project for deployment  

---

## Tools & Technologies Used

- **Programming Language:** Python  
- **Libraries:**  
  - Pandas – data handling  
  - NumPy – numerical computations  
  - Matplotlib & Seaborn – data visualization  
  - Scikit-learn – machine learning models and evaluation  
- **IDE:** VS Code  
- **Version Control:** Git & GitHub  

---

## Dataset Description

- **Dataset Name:** Titanic Dataset  
- **Source:** Kaggle (Public Dataset)  
- **File Used:** `train.csv`  

### Key Features:
- `Pclass` – Passenger class  
- `Sex` – Gender  
- `Age` – Age of passenger  
- `Fare` – Ticket fare  
- `SibSp` – Siblings/spouses aboard  
- `Parch` – Parents/children aboard  
- `Embarked` – Port of embarkation  

**Target Variable:**  
- `Survived` (0 = Did not survive, 1 = Survived)

---

## Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values (Age, Embarked)
- Encoding categorical variables (Sex, Embarked)
- Removing irrelevant features (PassengerId, Name)
- Feature scaling where required
- Splitting data into training and testing sets

---

## Machine Learning Algorithms Used

### 1. Logistic Regression
- Used as a baseline classification model
- Simple and interpretable

### 2. Decision Tree Classifier
- Captures non-linear relationships
- Easy to visualize and explain

### 3. Random Forest Classifier
- Ensemble method for improved accuracy
- Reduces overfitting compared to decision trees

### Why these algorithms?
- They are well-suited for binary classification
- Provide a balance between performance and interpretability

---

## Model Evaluation

Models were evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)

Random Forest achieved the best overall performance.

---

## Challenges Faced

- Handling missing data without losing important information
- Encoding categorical variables correctly
- Avoiding overfitting
- Selecting the best-performing model
- Maintaining a clean and reproducible project structure

---

## How to Run the Project

1. Clone the repository
2. Install dependencies
3. Run the application as streamlit run app.py
git clone https://github.com/siri-orog/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction

---

##Future Improvements

- Hyperparameter tuning for better accuracy
- Feature engineering for improved predictions
- Model deployment using Flask or FastAPI
- Web-based user interface
- Integration with real-time user inputs
