import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

def build_pipeline(obj_attribs, num_attribs):
    obj_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("obj", obj_pipeline, obj_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

data= pd.read_csv("Stock_data.csv")
data["Date"]= pd.to_datetime(data["Date"])
data["Year"]= data["Date"].dt.year.astype("int64")
data["Month"]= data["Date"].dt.month.astype("int64")
data["Day"]= data["Date"].dt.day.astype("int64")
data.drop("Date", axis=1, inplace=True)

n_bins=10
data["Daily_Percent_bin"]= pd.qcut(data["Daily_Change_Percent"], q=n_bins, duplicates="drop")
split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Daily_Percent_bin"]):
    train_data= data.loc[train_index].drop(columns=["Daily_Percent_bin"])
    test_data= data.loc[test_index].drop(columns=["Daily_Percent_bin"])

train_data_label= train_data["Daily_Change_Percent"].copy()
train_data_input= train_data.drop("Daily_Change_Percent", axis=1)

test_data_label= test_data["Daily_Change_Percent"].copy()
test_data_input= test_data.drop("Daily_Change_Percent", axis=1)

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

col= train_data_input.columns.tolist()
obj_attribs=[]
num_attribs=[]
for obj_col in col:
    if train_data_input[obj_col].dtype=="object":
        obj_attribs.append(obj_col)
    else:
        num_attribs.append(obj_col)

pipeline= build_pipeline(obj_attribs, num_attribs)
train_data_input_arr= pipeline.fit_transform(train_data_input)
xgb_model= XGBRegressor(random_state=42, n_jobs=-1)
grid_search= GridSearchCV(xgb_model, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_search.fit(train_data_input_arr, train_data_label)
xgb_model= grid_search.best_estimator_
print("The Model is trained!")

test_data_input_arr= pipeline.transform(test_data_input)
predictions= xgb_model.predict(test_data_input_arr)
xgb_rmse= root_mean_squared_error(test_data_label, predictions)
test_data["Predicted_Daily_Change_Precent"]= predictions
print(test_data)
print(f"the root mean squared error of this model is {xgb_rmse:.2f}")