import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
# from sklearn.metrics import root_mean_squared_error

def model_accuracy(test_set_label, predictions):
    count=0
    for i in range(len(test_set_label)):
        if predictions[i]== test_set_label.iloc[i]:
            count+=1
    accuracy= float((count/len(test_set_label))*100)
    return accuracy

def build_pipeline(obj_attribs, num_attribs):
    obj_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipeline=Pipeline([
        ("Impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("obj", obj_pipeline, obj_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

data= pd.read_csv("original_dataset.csv")
col= data.columns.tolist()
obj_col=[]
for col_1 in col:
    if data[col_1].dtype=="object":
        obj_col.append(col_1)

for col_2 in obj_col:
    data[col_2]= data[col_2].fillna(np.nan).astype("object")

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["loan_paid_back"]):
    train_set= data.loc[train_index]
    test_set= data.loc[test_index]

test_set_label= test_set["loan_paid_back"].copy()
test_set_input= test_set.drop("loan_paid_back", axis=1)

train_set_label= train_set["loan_paid_back"].copy()
train_set_input= train_set.drop("loan_paid_back", axis=1)

train_col= train_set_input.columns.tolist()
obj_attribs=[]
num_attribs=[]

for col_3 in train_col:
    if train_set_input[col_3].dtype=="object":
        obj_attribs.append(col_3)

for col_4 in train_col:
    if train_set_input[col_4].dtype!="object":
        num_attribs.append(col_4)

pipeline= build_pipeline(obj_attribs, num_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

pipeline= build_pipeline(obj_attribs, num_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

xgb_model= XGBClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label)
xgb_model = grid_search.best_estimator_

test_set_input_arr= pipeline.transform(test_set_input)
predictions= xgb_model.predict(test_set_input_arr)
test_set["predicted_loan_paid_back"]= predictions

xgb_accuracy= model_accuracy(test_set_label, predictions)
print("The model is trained!")
print(test_set)
print(f"The accuracy of this model is {xgb_accuracy}%")