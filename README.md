# MindBuddy Python Stress Model

This service exposes a FastAPI endpoint that computes stress score and category from assessment answers.

## Run

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

## Endpoint

POST /predict-stress

Request JSON should include:
- age
- gender
- maritalStatus
- employmentStatus
- sleepDuration
- educationLevel
- physicalActivity
- screenTime
- socialSupport
- workStress

Response:
- stressScore
- stressCategory
