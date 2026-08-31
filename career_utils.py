"""
career_utils.py
----------------
Shared feature-engineering and helper functions for the AI Career Mentor
model-training pipeline (8 datasets -> 8 trained models).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def count_items(series, sep=","):
    """Count comma-separated items in a text column (NaN/blank -> 0)."""
    return series.fillna("").apply(lambda x: len([i for i in x.split(sep) if i.strip()]))


def encode_categorical(train_series, other_series_list=None):
    """Label-encode a categorical column. Fits on train, transforms consistently
    across any other splits, mapping unseen categories to -1."""
    le = LabelEncoder()
    le.fit(train_series.astype(str))
    mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

    def transform(s):
        return s.astype(str).map(mapping).fillna(-1).astype(int)

    return transform(train_series), le, transform


ORDINAL_MAPS = {
    "education_level": {
        "High School": 0, "Diploma": 1, "Associate Degree": 1, "Bachelor's Degree": 2,
        "Master's Degree": 3, "PhD": 4, "Doctorate": 4,
    },
    "overall_rating": {"Poor": 0, "Below Average": 1, "Average": 2, "Good": 3, "Excellent": 4},
    "review_rating": {"Poor": 0, "Needs Improvement": 1, "Average": 2, "Good": 3, "Excellent": 4},
    "learning_priority": {"Low": 0, "Medium": 1, "High": 2},
    "difficulty_level": {"Easy": 0, "Moderate": 1, "Challenging": 2, "Hard": 3},
}


def ordinal_encode(series, col_name):
    m = ORDINAL_MAPS[col_name]
    return series.map(m).fillna(-1).astype(int)
