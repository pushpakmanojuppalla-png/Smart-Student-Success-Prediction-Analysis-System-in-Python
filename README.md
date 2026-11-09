# Smart Student Success Prediction & Analysis System

An end-to-end Python machine-learning project that predicts how well a student will
perform, flags students who are **At-Risk**, explains what drives the prediction and
suggests concrete interventions.

It includes:

- a reproducible **synthetic dataset generator** (swap in real data any time)
- a **scikit-learn pipeline** (imputation, scaling, one-hot encoding)
- two models: a **classifier** (At-Risk / Average / High) and a **regressor** (final score 0-100)
- automatic **model comparison** with cross-validation
- **EDA charts**, permutation-based **feature importance** and an auto-written insights report
- rule-based, explainable **recommendations** per student
- a **Streamlit dashboard** (overview, single prediction, batch CSV prediction, model metrics)
- a **CLI** and a **pytest** suite

## Project structure

```
smart_student_success/
├── main.py                  # CLI: generate | train | analyze | predict | all
├── requirements.txt
├── conftest.py              # lets pytest import `src`
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── src/
│   ├── config.py            # paths, feature lists, thresholds
│   ├── data_generator.py    # synthetic data
│   ├── preprocessing.py     # load, validate, split, preprocessing pipeline
│   ├── train.py             # model comparison, training, evaluation, saving
│   ├── analysis.py          # EDA figures, feature importance, insights.md
│   ├── predict.py           # StudentPredictor (single + batch)
│   └── recommend.py         # rule-based recommendations
├── tests/
│   └── test_pipeline.py
├── data/                    # students.csv (generated)
├── models/                  # classifier.joblib, regressor.joblib (generated)
└── reports/                 # metrics.json, insights.md, CSV tables, figures/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py all                 # generate data -> train -> analyze -> demo prediction
streamlit run app/dashboard.py     # open the dashboard
python -m pytest                   # run the tests
```

Individual steps:

```bash
python main.py generate --n 3000        # new synthetic dataset
python main.py train                    # add --fast for a quicker run
python main.py analyze                  # figures + feature importance + insights
python main.py predict                  # demo student
python main.py predict --csv new_students.csv --out predictions.csv
```

## Features used

| Column | Type | Description |
|---|---|---|
| age | numeric | 17-24 |
| study_hours_per_week | numeric | weekly self-study hours |
| attendance_pct | numeric | class attendance % |
| previous_gpa | numeric | prior GPA on a 4.0 scale |
| assignment_completion_pct | numeric | assignments submitted % |
| internal_marks | numeric | internal assessment marks (0-100) |
| participation_score | numeric | class participation (1-10) |
| sleep_hours | numeric | average nightly sleep |
| stress_level | numeric | self-reported stress (1-10) |
| extracurricular_activities | numeric | number of activities |
| absences | numeric | days absent |
| gender, parental_education, family_support, internet_access | categorical | background factors |

Targets: `final_score` (regression) and `performance_level`
(`At-Risk` < 55, `Average` 55-75, `High` >= 75; thresholds live in `src/config.py`).

## Method

1. **Preprocess**: median imputation + standard scaling for numeric columns, most-frequent
   imputation + one-hot encoding for categorical columns. The preprocessor is part of the
   saved pipeline, so inference needs no extra steps and tolerates missing inputs.
2. **Split**: stratified 80/20 train/test on performance level.
3. **Select**: 5-fold cross-validation over Logistic Regression, Random Forest and Gradient
   Boosting (classifier, macro-F1) and Ridge, Random Forest and Gradient Boosting
   (regressor, RMSE). The best candidate is refit on the full training split.
4. **Evaluate** on the held-out test split: accuracy, macro-F1, confusion matrix, MAE,
   RMSE, R².
5. **Explain**: permutation importance on the test set shows which inputs matter most.
6. **Act**: `src/recommend.py` turns a student's weak spots into specific suggestions.

## Using real data

The bundled dataset is **synthetic**, generated from the formula in
`src/data_generator.py`, so the reported accuracy reflects how learnable that formula is,
not real-world performance. To use your own data, save a CSV at `data/students.csv` with
the columns listed above plus `final_score` and `performance_level`, then run
`python main.py train` and `python main.py analyze`. If your column names differ (for
example the UCI "Student Performance" dataset), edit the lists in `src/config.py`.

## Ideas for extension

- add a time dimension (weekly snapshots) and predict dropout early in the term
- add SHAP explanations per student
- calibrate probabilities and tune the At-Risk decision threshold for recall
- check fairness across gender / background groups before any real deployment
- store predictions in a database and add authentication to the dashboard
- containerise with Docker

## Responsible use

Predictions should support, never replace, human judgement. Use them to offer help early,
not to label or penalise students, and audit any model trained on real student data for
bias.
