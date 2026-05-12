"""
Train the stress assessment model from mental_health_risk_dataset.csv.

Usage:
    python train_model.py
    python train_model.py "C:/path/to/mental_health_risk_dataset.csv"
"""

import sys
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── locate dataset ────────────────────────────────────────────────────────────
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'mental_health_risk_dataset.csv')

if not os.path.exists(CSV_PATH):
    print(f'ERROR: Dataset not found at:\n  {CSV_PATH}')
    print('\nUsage: python train_model.py "path/to/mental_health_risk_dataset.csv"')
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f'Loaded {len(df):,} rows from {CSV_PATH}')

# ── select features that match Assessment.jsx questions ───────────────────────
FEATURES = [
    'age',
    'gender',
    'marital_status',
    'employment_status',
    'sleep_hours',
    'education_level',
    'physical_activity_hours_per_week',
    'screen_time_hours_per_day',
    'social_support_score',
    'work_stress_level',
    'anxiety_score',
    'depression_score',
    'panic_attack_history',
    'financial_stress_level',
    'family_history_mental_illness',
]
TARGET = 'mental_health_risk'

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    print(f'ERROR: Missing columns: {missing}')
    sys.exit(1)

X = df[FEATURES].copy()
y = df[TARGET]

# ── categorical / numerical split ─────────────────────────────────────────────
CATEGORICAL = ['gender', 'marital_status', 'employment_status', 'education_level']
NUMERICAL   = ['age', 'sleep_hours', 'physical_activity_hours_per_week',
               'screen_time_hours_per_day', 'social_support_score', 'work_stress_level']

for col in NUMERICAL:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(df[col].median())

print('Class distribution:', y.value_counts().sort_index().to_dict())
print('Categorical values:')
for col in CATEGORICAL:
    print(f'  {col}: {sorted(X[col].unique())}')

# ── build pipeline ────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )),
])

# ── train / evaluate ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nTraining on {len(X_train):,} samples, evaluating on {len(X_test):,} …')
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

# ── save ──────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assessment_model.pkl')
with open(OUT, 'wb') as f:
    pickle.dump(pipeline, f)
print(f'\nModel saved -> {OUT}')
