# Customer Churn Prediction

## Author
**PIYUSH KUMAR YADAV**


# Project Overview
Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom company based on customer information such as contract type, monthly charges, internet service, tenure, payment method, and other features.

The project uses multiple Machine Learning algorithms and compares their performance. A Streamlit web application is developed to make predictions using the trained model.

# Objectives
- Predict customer churn.
- Compare different Machine Learning models.
- Analyze important features affecting churn.
- Deploy the best model using Streamlit.

# Dataset Information
Dataset Name: Telecom Customer Churn Dataset

Number of Records: 7043

Target Variable:

- Churn
  - Yes
  - No

Features Include:
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges


# Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

# Machine Learning Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Random Forest was selected as the final model.

# Exploratory Data Analysis

The project includes:

- Missing Value Analysis
- Duplicate Removal
- Churn Distribution
- Gender Distribution
- Contract Analysis
- Monthly Charges Distribution
- Tenure Distribution
- Confusion Matrix
- ROC Curve
- Feature Importance

# Project Structure

```
Customer_Churn_Prediction/
│
├── app.py
├── main.py
├── dataset.csv
├── requirements.txt
├── README.md
│
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
```

---

# Installation

Clone the repository:

```bash
git clone <repository_link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Project

Train the model:

```bash
python main.py
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```


# Model Performance
Replace the values below with the results from your project.

| Model | Accuracy |
|--------|----------|
| Logistic Regression | XX.XX% |
| Decision Tree | XX.XX% |
| Random Forest | XX.XX% |

# Future Improvements
- Hyperparameter tuning
- Cross-validation
- XGBoost implementation
- Feature selection
- Cloud deployment
- Real-time prediction API

# Screenshots
Add screenshots of:
- Home Page
- Prediction Page
- Churn Prediction Result
- Non-Churn Prediction Result

# Conclusion

The project successfully predicts customer churn using Machine Learning techniques. Among all tested algorithms, Random Forest provided the best performance and was deployed using Streamlit to create an interactive web application.

# Author
PIYUSH KUMAR YADAV
Machine Learning Engineer (Aspiring)