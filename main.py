import os
import pickle
from enum import Enum

import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Annotated
from pydantic import BaseModel, Field

# ── load trained model ────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assessment_model.pkl')

with open(_MODEL_PATH, 'rb') as f:
    _model = pickle.load(f)
print(f'[MindBuddy] Loaded ML model from {_MODEL_PATH}')


# ── value maps (frontend option -> dataset value) ─────────────────────────────
_GENDER_MAP     = {'male': 'Male', 'female': 'Female', 'other': 'Other'}
_MARITAL_MAP    = {'married': 'Married', 'divorced': 'Divorced', 'other': 'Single'}
_EMPLOYMENT_MAP = {'employed': 'Employed', 'self-employed': 'Self-Employed', 'other': 'Unemployed'}
_EDUCATION_MAP  = {'high-school': 'High School', 'bachelor': 'Bachelor', 'other': 'Master'}
_SLEEP_MAP      = {'1-5': 3.0, '5-10': 7.5, '10-15': 12.5}
_ACTIVITY_MAP   = {'0-3': 1.5, '3-6': 4.5, '6-9': 7.5, '9+': 11.0}
_SCREEN_MAP     = {'0-2': 1.0, '2-4': 3.0, '4-6': 5.0, '6-8': 7.0, '8+': 10.0}

# category midpoints used to derive a continuous 0-100 score from probabilities
_MIDPOINTS = [17.0, 49.5, 82.5]  # low / moderate / high


# ── pydantic input schema ─────────────────────────────────────────────────────
class Gender(str, Enum):
    male = 'male'
    female = 'female'
    other = 'other'


class MaritalStatus(str, Enum):
    married = 'married'
    divorced = 'divorced'
    other = 'other'


class EmploymentStatus(str, Enum):
    employed = 'employed'
    self_employed = 'self-employed'
    other = 'other'


class SleepDuration(str, Enum):
    one_to_five = '1-5'
    five_to_ten = '5-10'
    ten_to_fifteen = '10-15'


class EducationLevel(str, Enum):
    high_school = 'high-school'
    bachelor = 'bachelor'
    other = 'other'


class PhysicalActivity(str, Enum):
    zero_to_three = '0-3'
    three_to_six = '3-6'
    six_to_nine = '6-9'
    nine_plus = '9+'


class ScreenTime(str, Enum):
    zero_to_two = '0-2'
    two_to_four = '2-4'
    four_to_six = '4-6'
    six_to_eight = '6-8'
    eight_plus = '8+'


class StressInput(BaseModel):
    age: int
    gender: Gender
    maritalStatus: MaritalStatus
    employmentStatus: EmploymentStatus
    sleepDuration: SleepDuration
    educationLevel: EducationLevel
    physicalActivity: PhysicalActivity
    screenTime: ScreenTime
    socialSupport: Annotated[int, Field(ge=1, le=10)]
    workStress: Annotated[int, Field(ge=1, le=10)]
    anxietyScore: Annotated[int, Field(ge=1, le=10)]
    depressionScore: Annotated[int, Field(ge=1, le=10)]
    financialStress: Annotated[int, Field(ge=1, le=10)]
    panicAttackHistory: Annotated[int, Field(ge=0, le=1)]
    familyHistoryMentalIllness: Annotated[int, Field(ge=0, le=1)]


class StressResult(BaseModel):
    stressScore: int
    stressCategory: str


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title='MindBuddy Stress Model',
    description='Predicts stress score and category from assessment answers.',
    version='2.0.0',
)


@app.get('/')
async def root():
    return {'status': 'ok', 'service': 'mindbuddy-stress-model'}


# ── endpoint ──────────────────────────────────────────────────────────────────
@app.post('/predict-stress', response_model=StressResult)
async def predict_stress(payload: StressInput):
    try:
        row = pd.DataFrame([{
            'age':                              int(payload.age),
            'gender':                           _GENDER_MAP[payload.gender.value],
            'marital_status':                   _MARITAL_MAP[payload.maritalStatus.value],
            'employment_status':                _EMPLOYMENT_MAP[payload.employmentStatus.value],
            'sleep_hours':                      _SLEEP_MAP[payload.sleepDuration.value],
            'education_level':                  _EDUCATION_MAP[payload.educationLevel.value],
            'physical_activity_hours_per_week': _ACTIVITY_MAP[payload.physicalActivity.value],
            'screen_time_hours_per_day':        _SCREEN_MAP[payload.screenTime.value],
            'social_support_score':             int(payload.socialSupport),
            'work_stress_level':                int(payload.workStress),
            'anxiety_score':                    int(payload.anxietyScore),
            'depression_score':                 int(payload.depressionScore),
            'financial_stress_level':           int(payload.financialStress),
            'panic_attack_history':             int(payload.panicAttackHistory),
            'family_history_mental_illness':    int(payload.familyHistoryMentalIllness),
        }])

        probas = _model.predict_proba(row)[0]
        score  = round(sum(p * m for p, m in zip(probas, _MIDPOINTS)))
        score  = max(0, min(100, score))

        if score < 35:
            category = 'low'
        elif score < 65:
            category = 'moderate'
        else:
            category = 'high'

        return {'stressScore': score, 'stressCategory': category}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
