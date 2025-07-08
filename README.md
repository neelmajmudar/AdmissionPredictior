# 🎓 Admission Predictor

## About
Admission Predictor is a machine learning-driven solution that estimates the likelihood of university admissions based on applicant profiles. Leveraging a blend of academic, standardized test, and extracurricular data, this tool empowers students, counselors, and institutions to gain insights into application outcomes. It supports CSV batch processing, enables model retraining, and provides performance evaluation to facilitate data-informed decision-making.

## Features
- **Wide-ranging input**  
  Handles applicant data including GRE, TOEFL, University Rating, SOP strength, LOR strength, CGPA, and research experience.

- **Prediction outputs**  
  Provides smooth probability scores (0–1) for admission chances and bins them into percentile categories for easy interpretation.

- **Model flexibility**  
  Includes options for linear regression, random forest, and neural networks. Easily switchable based on experimental needs.

- **Complete toolchain**  
  From preprocessing and training to prediction and evaluation—it supports the full ML workflow through  `AdmissionClassifier.py`.

- **Visualization support**  
  Generates model diagnostics (e.g. ROC, predicted vs. actual plots) for deeper insights into strengths and weaknesses.

- **Configurable and reproducible**  
  Utilizes config files or CLI flags for setting hyperparameters, data paths, or model types. Ensures reproducibility with fixed seeds and clear documentation.

## Project overview
Admission Predictor is organized to support seamless experimentation:

- **Data ingestion**  
  Clean and scale datasets via `data/prepare_data.py`. Supports splitting into training, validation, and test sets.

- **Model training**  
  The `train.py` script ingests preprocessed data to train a selected model (linear regression, random forest, etc.). After training, models are serialized for reuse.

- **Batch prediction**  
  Use `predict.py` to apply trained models to new applicant data in CSV form. Includes CLI options for thresholds, batching, and output customization.

- **Performance evaluation**  
  The `evaluate.py` script compares predictions to ground truth using metrics like Mean Squared Error (MSE), R², classification accuracy, and ROC/AUC. Optional visual plots help assess model validity.

- **Extensibility**  
  Easily plug in new models (e.g. XGBoost, TensorFlow/Keras nets) or add engineered features (like work experience, essays). The modular codebase ensures contributors can enhance each component independently.

- **Usage flow**  
  1. `prepare_data.py` → 2. `train.py` → 3. `predict.py` → 4. `evaluate.py` (optional retraining) → repeat.

---

## Quickstart

```bash
git clone https://github.com/neelmajmudar/AdmissionPredictior.git
cd AdmissionPredictior
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Prepare data
python data/prepare_data.py --input data/raw_admissions.csv --output data/processed.pkl

# Train model
python train.py --data data/processed.pkl --model models/admit_rf.pkl

# Predict on new applicants
python predict.py --input new_students.csv --model models/admit_rf.pkl --output results/predictions.csv

# Evaluate on a test set
python evaluate.py --model models/admit_rf.pkl --test data/test.csv --plots --metrics
