# 🎓 Admission Predictor

## About

**Admission Predictor** is a machine learning web application that estimates the probability of university admission based on academic and research credentials. Built with Python and Streamlit, it features both an interactive front-end and a modular backend for model training, evaluation, and visualization. The tool supports fast experimentation and clear interpretability using regression models and feature importance diagnostics.

## Features

- **🔘 Interactive Web App**  
  Launch `AdmissionApp.py` to access a Streamlit-based interface where users can upload data, generate predictions, and view visualizations.

- **🧠 Model Training & Evaluation**  
  `AdmissionClassifier.py` trains and evaluates models using the data in `admissions_data.csv`. It supports regression models such as Random Forest and Linear Regression.

- **📊 Visual Analytics**  
  Key diagnostic plots are generated during training:
  - `Actual vs Residual.png`
  - `Training&ValidationLoss.png`
  - `Training&ValidationMAE.png`
  - `Feature Importances.png`

- **🧹 Data Preprocessing**  
  Built-in preprocessing using `Pandas` and `NumPy`, including feature scaling and automatic train/test splitting.

- **📈 Evaluation Metrics**  
  The model is evaluated using:
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - R² Score  
  - Loss and MAE visualizations across training epochs

- **🧾 Explainability**  
  Feature importance visualizations provide transparency into which input variables most affect predictions.

## File Overview

| File                         | Description                                           |
|------------------------------|-------------------------------------------------------|
| `AdmissionApp.py`           | Streamlit front-end for interactive predictions and visualizations |
| `AdmissionClassifier.py`    | Core backend script for training and evaluation       |
| `admissions_data.csv`       | Dataset containing historical applicant records       |
| `Actual vs Residual.png`    | Plot showing residual error for trained model         |
| `Feature Importances.png`   | Bar chart ranking most influential features           |
| `Training&ValidationLoss.png` | Training vs validation loss over epochs             |
| `Training&ValidationMAE.png` | Training vs validation MAE over epochs              |
| `README.md`                 | Project documentation                                 |
| `LICENSE`                   | Project license (MIT)                                 |

## Technologies Used

- **Languages & Libraries**  
  Python, Pandas, NumPy, scikit-learn, OpenCV, Matplotlib, Seaborn

- **Frameworks**  
  Streamlit (for frontend UI)

- **Models**  
  Linear Regression, Random Forest (extensible to more)

- **Evaluation Metrics**  
  MAE, MSE, R² Score

## Future Improvements

- Add support for neural networks (e.g., Keras)
- Enable user-based retraining in-browser
- Integrate essay/text features via NLP
- Deploy with authentication for student accounts


## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/neelmajmudar/AdmissionPredictior.git
cd AdmissionPredictior

pip install "Everything given in AdmissionClassifier and App.py

streamlit run AdmissionApp.py

