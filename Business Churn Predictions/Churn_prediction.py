import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def xgb_model_accuracy(predictions, test_set_label):
    count=0
    for i in range(len(test_set_label)):
        if predictions[i]==test_set_label.iloc[i]:
            count+=1
    accuracy= float((count/len(test_set_label))*100)
    return accuracy

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
        ("object", obj_pipeline, obj_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

data= pd.read_csv("customer_churn_business_dataset.csv")
col= data.columns.tolist()
# for i in range(len(col)):
#     if data[col[i]].dtype=="object":
#         data[col[i]]= data[col[i]].astype("string")

object_cols = data.select_dtypes(include=["object"]).columns
for col_name in object_cols:
    data[col_name] = data[col_name].fillna(np.nan).astype("object")

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["churn"]):
    train_set= data.loc[train_index]
    test_set= data.loc[test_index]

train_set_label= train_set["churn"].copy()
train_set_input= train_set.drop("churn", axis=1)

test_set_label= test_set["churn"].copy()
test_set_input= test_set.drop("churn", axis=1)

obj_attribs= []
num_attribs= []
col_1= train_set_input.columns.tolist()

for i in range(len(col_1)):
    if train_set_input[col_1[i]].dtype=="object":
        obj_attribs.append(col_1[i])
for i in range(len(col_1)):
    if train_set_input[col_1[i]].dtype!="object":
        num_attribs.append(col_1[i])
# print(obj_attribs)
# print(num_attribs)

pipeline= build_pipeline(obj_attribs, num_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

xgb_model= XGBClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label)
xgb_model = grid_search.best_estimator_

test_set_input_arr= pipeline.transform(test_set_input)
predictions= xgb_model.predict(test_set_input_arr)
test_set["predicted_churn"]=predictions
xgbclassifier_accuracy= xgb_model_accuracy(predictions, test_set_label)
print(f"The accuracy of this business churn model is {xgbclassifier_accuracy}%")
print(test_set)