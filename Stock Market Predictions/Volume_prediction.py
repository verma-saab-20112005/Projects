import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRFRegressor
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

def build_pipeline(obj_attribs, num_attribs):
    obj_pipeline= Pipeline([
        ("Impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipeline= Pipeline([
        ("Impute", SimpleImputer(strategy="median")),
        ("Standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("obj", obj_pipeline, obj_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

data= pd.read_csv("Stock_data.csv")
data["Date"]= pd.to_datetime(data["Date"])
data["Year"]= data["Date"].dt.year
data["Month"]= data["Date"].dt.month
data["Day"]= data["Date"].dt.day
data.drop(columns=["Date"], inplace=True)
n_bins=10
data["Volume_bin"]= pd.qcut(data["Volume"], q=n_bins, duplicates="drop")

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Volume_bin"]):
    train_set= data.loc[train_index].drop(columns= ["Volume_bin"])
    test_set= data.loc[test_index].drop(columns= ["Volume_bin"])

train_set_label= train_set["Volume"].copy()
train_set_input= train_set.drop(columns= ["Volume"])

test_set_label= test_set["Volume"].copy()
test_set_input= test_set.drop(columns= ["Volume"])

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
col= train_set_input.columns.tolist()
obj_attribs=[]
num_attribs=[]
for col_1 in col:
    if train_set_input[col_1].dtype=="object":
        obj_attribs.append(col_1)
    else:
        num_attribs.append(col_1)

pipeline= build_pipeline(obj_attribs, num_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)
xgbrf_model= XGBRFRegressor(random_state=42, n_jobs=-1)
grid_search= GridSearchCV(xgbrf_model, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label)
xgbrf_model= grid_search.best_estimator_
print("Your model is trained and is ready for the deployment!")

test_set_input_arr= pipeline.transform(test_set_input)
predictions= xgbrf_model.predict(test_set_input_arr)
test_set["Predicted_Volume"]= predictions
xgbrf_rmse= root_mean_squared_error(test_set_label, predictions)
print(test_set)
print(f"The root mean square error of this model is {xgbrf_rmse:.2f}")