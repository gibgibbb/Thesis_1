import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, jaccard_score
)
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv('test.csv')
print("Dataset shape:", df.shape)
print("Class balance:\n", df['Ignited'].value_counts(normalize=True))
print("Missing values:\n", df.isnull().sum())
print("\nDataset head:\n", df.head())