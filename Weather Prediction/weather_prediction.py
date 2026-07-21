import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class DateYMExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        # Set required fitted attributes for sklearn compatibility
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.n_features_in_ = 1
            self.feature_names_in_ = [self.col_name]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        else:
            s = X
        s = pd.to_datetime(s, errors="raise")
        return np.c_[
            s.dt.year.to_numpy(),
            s.dt.month.to_numpy(),
            s.dt.day.to_numpy()
        ]

def build_pipeline(num_attribs, date_attribs):
    date_pipeline= Pipeline([
        ("date", DateYMExtractor(date_attribs))
    ])
    num_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("date", date_pipeline, date_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

def classification_accuracy(predictions, test_set_labels):
    count= 0
    for i in range(len(predictions)):
        if predictions[i]==test_set_labels.iloc[i]:
            count= count+1
    accuracy= float((count/len(predictions))*100)
    return accuracy

data= pd.read_excel("Weather_history.xlsx")

col= data.columns.tolist()
for i in range(6):
    if data[col[i]].dtype=="object":
        data[col[i]]=data[col[i]].astype("string")

# string_cols = data.select_dtypes(include=["string"]).columns
# for col_name in string_cols:
#     data[col_name] = data[col_name].fillna(np.nan).astype("object")

# print(data["weather"].dtype)

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

data["avg_temp"]= (data["temp_max"] + data["temp_min"])/2
n_bins = 5
data["avg_temp_bin"] = pd.qcut(data["avg_temp"], q=n_bins, duplicates="drop")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["avg_temp_bin"]):
    train_set = data.loc[train_index].drop(columns=["avg_temp_bin"])
    test_set = data.loc[test_index].drop(columns=["avg_temp_bin"])

test_set_labels= test_set["weather"].copy()
test_set_input= test_set.drop("weather", axis=1)

train_set_labels= train_set["weather"].copy()
train_set_input= train_set.drop("weather", axis=1)

label_encoder= LabelEncoder()
train_set_labels_arr= label_encoder.fit_transform(train_set_labels)
# print(train_set_labels_arr)

date_attribs= ["date"]
num_attribs= train_set_input.drop("date", axis=1).columns.tolist()

pipeline= build_pipeline(num_attribs, date_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

xgb_model= XGBClassifier(objective="multi:softprob", num_class=5, random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_labels_arr)
xgb_model = grid_search.best_estimator_

test_set_input_arr= pipeline.transform(test_set_input)
predictions_encoded= xgb_model.predict(test_set_input_arr)
predictions= label_encoder.inverse_transform(predictions_encoded)
model_accuracy= classification_accuracy(predictions, test_set_labels)
print(f"The accuracy in this classification model is {model_accuracy}%")
test_set["predicted_weather"]= predictions
print(test_set)