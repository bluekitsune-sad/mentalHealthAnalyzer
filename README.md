# Mental Health Counseling Prediction

A Flask-based machine learning application that predicts whether a student likely needs counseling based on survey-like inputs (e.g., anxiety, sleep quality, CGPA, stress level). The project includes a full pipeline: data ingestion, preprocessing, model training with stratified CV, an inference pipeline, a web UI, and test utilities.

## Features

- Automated target creation: `IsCounselingNeeded` derived from multiple indicators
- Data ingestion with stratified train/test split
- Robust preprocessing with `OneHotEncoder(handle_unknown="ignore")` and scaling
- Binary classification models (best-by-F1 via grid search with stratified CV)
- Prediction pipeline using serialized artifacts (`artifacts/model.pkl`, `artifacts/preprocessor.pkl`)
- Flask web app with a form to collect inputs and display prediction
- Test script (`test.py`) that validates the pipeline and reports performance

## Project Structure

```
mentalHealth/
  app.py
  test.py
  artifacts/
    data.csv
    train.csv
    test.csv
    preprocessor.pkl
    model.pkl
  src/
    components/
      data_ingestion.py
      data_transformation.py
      model_trainer.py
    pipeline/
      predict_pipeline.py
    utils/
      filesManager.py
      features.py
    logger.py
    exception.py
  templates/
    index.html
    home.html
  notebook/
    ... (explorations)
```

## Setup

1. Python environment
   - Python 3.8+
   - Optionally create a venv and activate it
2. Install dependencies

```bash
pip install -r requirements.txt
```

## How It Works

1. Data Ingestion (`src/components/data_ingestion.py`)

   - Loads `notebook/data/mentalhealth_dataset.csv`
   - Creates target column `IsCounselingNeeded` from rule-based conditions
   - Drops non-feature columns like `Timestamp`, `Course`, `YearOfStudy` if present
   - Performs stratified `train_test_split` on `IsCounselingNeeded`
   - Writes `artifacts/data.csv`, `train.csv`, and `test.csv`

2. Data Transformation (`src/components/data_transformation.py`)

   - Auto-detects numerical/categorical columns (excluding target)
   - Pipelines: impute + scale numerics; impute + one-hot + scale categoricals
   - Uses `OneHotEncoder(handle_unknown="ignore")` for robust inference
   - Serializes `artifacts/preprocessor.pkl`

3. Model Training (`src/components/model_trainer.py`)

   - Trains multiple classifiers (RF, DT, GB, Logistic, XGBoost, CatBoost, AdaBoost)
   - Hyperparameter tuning with `GridSearchCV` using `StratifiedKFold(n_splits=2)`
   - Selects best model by F1 score on validation folds
   - Saves best model to `artifacts/model.pkl`

4. Prediction Pipeline (`src/pipeline/predict_pipeline.py`)

   - Loads artifacts, preprocesses inputs, and returns model predictions
   - `CustomData` class normalizes inputs (e.g., gender capitalization) and outputs a DataFrame

5. Web App (`app.py` and `templates/`)

   - `GET /predictdata` shows a form for inputs
   - `POST /predictdata` runs the prediction and renders a label:
     - `Counseling Needed` or `No Counseling Needed`

6. Tests (`test.py`)
   - Validates artifacts, pipeline, sample predictions, multiple scenarios
   - Evaluates performance on `artifacts/test.csv` and prints metrics (Accuracy, F1, Confusion Matrix)

## Run the Pipeline

1. Generate artifacts (ingestion → transform → train):

```bash
py -m src.components.data_ingestion
```

2. Run tests:

```bash
python test.py
```

3. Run the Flask app:

```bash
python app.py
```

- Navigate to `http://localhost:5000/predictdata`

## API Usage (Programmatic)

Example of using the prediction pipeline directly:

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    gender="Male", age=21, cgpa=3.4, depression=0, anxiety=1, panic_attack=0,
    specialist_treatment=0, symptom_frequency_last7days=2, has_mental_health_support=1,
    sleep_quality=4, study_stress_level=2, study_hours_per_week=12, academic_engagement=4
)
X = data.get_data_as_data_frame()
preds = PredictPipeline().predict(X)
label = "Counseling Needed" if int(preds[0]) == 1 else "No Counseling Needed"
print(label)
```

## Notes on Class Imbalance

- The target creation rule may produce highly imbalanced labels (mostly 1s)
- We mitigate via stratified split and class-weighted models, but consider:
  - Collecting more balanced data
  - Threshold tuning via `predict_proba`
  - Sampling techniques (undersample/SMOTE) if required

## Troubleshooting

- OneHotEncoder unknown category errors: already mitigated via `handle_unknown="ignore"`
- If you change the schema, re-run ingestion and training to regenerate artifacts
- Ensure `catboost` and `xgboost` are installed if you keep those models in the grid

## Deployment

- Artifacts required at runtime:
  - `artifacts/preprocessor.pkl`
  - `artifacts/model.pkl`
- Expose a WSGI callable and serve via a production server (gunicorn/uvicorn + WSGI/ASGI bridge)

Important: For deployment on many platforms, name your Flask entry file `application.py` (instead of `app.py`) so the platform auto-detects the app object. Create `application.py` with the same content as `app.py` or import the app there and run it.

<!-- IsCounselingNeeded
1    0.997
0    0.003
Name: proportion, dtype: float64 -->
