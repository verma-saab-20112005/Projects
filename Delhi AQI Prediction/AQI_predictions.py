import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
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

data= pd.read_excel("Delhi_AQI_2025.xlsx")
data["date_ist"]= pd.to_datetime(data["date_ist"])
data["year"]= data['date_ist'].dt.year
data["month"]= data["date_ist"].dt.month
data["day"]= data["date_ist"].dt.day

data["time_ist"]= data["time_ist"].astype("string")
data["hours"]= data["time_ist"].str.split(":", expand=True)[0].astype("int64")
data= data.drop(["date_ist", "time_ist"], axis=1)

col= data.columns.tolist()
for d_col in col:
    if data[d_col].dtype=="int32":
        data[d_col]= data[d_col].astype("int64")
for obj_col in col:
    if data[obj_col].dtype=="object":
        data[obj_col]= data[obj_col].fillna(np.nan).astype("object")

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["aqi_index"]):
    train_set= data.loc[train_index]
    test_set= data.loc[test_index]

train_set_label= train_set["aqi_index"].copy()
train_set_input= train_set.drop("aqi_index", axis=1)

test_set_label= test_set["aqi_index"].copy()
test_set_input= test_set.drop("aqi_index", axis=1)

t_col= train_set_input.columns.tolist()
obj_attribs=[]
num_attribs=[]
for col_1 in t_col:
    if train_set_input[col_1].dtype=="object":
        obj_attribs.append(col_1)
for col_1 in t_col:
    if train_set_input[col_1].dtype!="object":
        num_attribs.append(col_1)

pipeline= build_pipeline(obj_attribs, num_attribs)
rf= RandomForestRegressor()

train_set_input_arr= pipeline.fit_transform(train_set_input)
rf.fit(train_set_input_arr, train_set_label)

test_set_input_arr= pipeline.transform(test_set_input)
predictions= rf.predict(test_set_input_arr)
test_set["predicted_aqi_index"]= predictions
rf_rmse= root_mean_squared_error(test_set_label, predictions)
print("The model is trainden on the AQI dataset!")
print(test_set)
print(f"The rmse error of this model in predicting the AQI is {rf_rmse}")