# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project uses machine learning to predict the presence of heart disease based on patient health characteristics. It compares two classification algorithms — **Logistic Regression** and **Random Forest Classifier** — to determine which performs better on the test dataset.

The project demonstrates a complete beginner-friendly machine learning workflow, from loading and preprocessing data to training models, evaluating performance, visualizing results, and making predictions on new patient data.

## 🎯 Objectives

- Predict the presence of heart disease using patient health data.
- Compare Logistic Regression and Random Forest classification models.
- Apply feature scaling using `StandardScaler`.
- Evaluate models using accuracy, classification reports, and confusion matrices.
- Visualize classification performance using a heatmap.
- Generate predictions for new patient records.

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualization
- **Seaborn** — Confusion matrix heatmap
- **Scikit-learn** — Preprocessing, model training, and evaluation

## 📂 Dataset

The project uses a heart disease dataset stored as `heart.csv`.

The dataset contains patient characteristics such as:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex category encoded numerically |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels, as encoded in the dataset |
| `thal` | Thalassemia-related category, as encoded in the dataset |
| `target` | Target label indicating heart disease status |

**Target variable:** `target`

The interpretation of the target labels and categorical encodings depends on the source dataset.

## ⚙️ Project Workflow

1. **Data Loading:** Load the dataset using Pandas.
2. **Data Inspection:** Display sample records and check for missing values.
3. **Feature Scaling:** Standardize input features using `StandardScaler`.
4. **Data Splitting:** Split the dataset into training and testing sets using an 80:20 ratio.
5. **Model Training:** Train Logistic Regression and Random Forest classifiers.
6. **Model Evaluation:** Compare accuracy scores and generate classification reports.
7. **Visualization:** Plot a confusion matrix heatmap using Seaborn.
8. **Prediction:** Use the selected model to predict the outcome for a new patient record.

## 🤖 Machine Learning Models

### 1. Logistic Regression

A classification algorithm used to estimate the probability of a binary outcome. It provides a baseline for comparing model performance.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to make predictions. It can capture nonlinear relationships between patient characteristics and the target.

The model with the higher test accuracy is selected for the final evaluation and prediction.

## 📊 Model Evaluation

The project uses the following evaluation metrics:

- **Accuracy:** Proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases identified correctly.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Summary of correct and incorrect classifications.

Accuracy alone is insufficient for evaluating a medical classification model. Recall, precision, and false-negative rates are particularly important when assessing potential heart disease detection systems.

## 🚀 Getting Started

### Prerequisites

Install Python and the required libraries.

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Heart-disease-prediction.git
```

### 2. Navigate to the project directory

```bash
cd heart-disease-prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Add the dataset

Place `heart.csv` in the project directory, or update the file path in the Python script to match your dataset location.

### 5. Run the project

```bash
python heart_disease.py
```

The script prints the model accuracies, classification report, and confusion matrix, displays a heatmap, and generates a prediction for the supplied patient example.

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── heart_disease.py
├── heart.csv
├── README.md
└── requirements.txt
```

## 🔮 Example Prediction

The script creates a sample patient record and uses the selected model to produce one of two outputs:

- `At risk of Heart Disease`
- `No Heart Disease`

The output is a model prediction, not a confirmed medical diagnosis.

## 🔧 Potential Improvements

- Perform exploratory data analysis and visualize feature distributions.
- Handle missing values explicitly where necessary.
- Use cross-validation and `GridSearchCV` for hyperparameter tuning.
- Compare additional models such as Support Vector Machines and XGBoost.
- Evaluate recall, precision, ROC-AUC, and false-negative rates.
- Prevent data leakage by fitting preprocessing steps only on the training data.
- Build a simple prediction interface using Flask or Streamlit.
- Save the trained model and preprocessing pipeline for deployment.

## ⚠️ Disclaimer

This project is intended for educational and machine learning practice purposes only. It is not a clinically validated diagnostic tool and must not be used to diagnose heart disease or replace professional medical advice.

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer | Aspiring AI Engineer

GitHub: [@Vedline2547](https://github.com/Vedline2547)

LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

---

⭐ If you find this project useful, consider giving the repository a star!
