import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.base import TransformerMixin, BaseEstimator

MODEL_FILE= "Sales_predictions.pkl"
PIPELINE_FILE= "Pipeline.pkl"

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
    
def build_pipeline(date_attribs, cat_attribs, num_attribs):
    date_pipeline= Pipeline([
        ("date", DateYMExtractor(date_attribs))
    ])
    cat_pipeline= Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("date", date_pipeline, date_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

data= pd.read_excel("train.xlsx")
for i in range(len(data.columns.tolist())):
    if data[data.columns.tolist()[i]].dtype=="object":
        data[data.columns.tolist()[i]]= data[data.columns.tolist()[i]].astype("string")

for i in range(len(data.columns.tolist())):
    if data[data.columns.tolist()[i]].dtype!="string" and data[data.columns.tolist()[i]].dtype!="datetime64[ns]":
        data[data.columns.tolist()[i]]= data[data.columns.tolist()[i]].astype("int64")

data["Sales_bin"] = pd.qcut(data["Sales"], q=5, duplicates="drop")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Sales_bin"]):
    training_data = data.loc[train_index].drop(columns=["Sales_bin"])
    testing_data = data.loc[test_index].drop(columns=["Sales_bin"])

test_labels= testing_data["Sales"].copy()
test_input= testing_data.drop("Sales", axis=1)


if not os.path.exists(MODEL_FILE):
    train_labels= training_data["Sales"].copy()
    train_input= training_data.drop("Sales", axis=1)
   
    date_attribs= ["Date"]
    cat_attribs= ["StateHoliday"]
    num_attribs= train_input.drop(["Date", "StateHoliday"], axis=1).columns.tolist()

    pipeline= build_pipeline(date_attribs, cat_attribs, num_attribs)
    train_input_arr= pipeline.fit_transform(train_input)

    param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
    }

    xgb_model= XGBRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(train_input_arr, train_labels)
    xgb_model = grid_search.best_estimator_

    joblib.dump(xgb_model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Your xgbregressor model is ready and is being saved in the file MODEL_FILE")
else:
    model= joblib.load(MODEL_FILE)
    pipeline= joblib.load(PIPELINE_FILE)
    test_input_arr= pipeline.transform(test_input)
    predictions= model.predict(test_input_arr)
    testing_data["Predicted_Sales"]= predictions
    print(testing_data)
    model_rmse= root_mean_squared_error(test_labels, predictions)
    print(f"The error in this prediction of sales is {model_rmse}")