import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning

def log_reg_accuracy(predictions, test_set_label):
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
        ("obj", obj_pipeline, obj_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)

data=pd.read_csv("ecommerce_customer_churn_dataset.csv")
col= data.columns.tolist()

object_cols = data.select_dtypes(include=["object"]).columns
for col_name in object_cols:
    data[col_name] = data[col_name].fillna(np.nan).astype("object")

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [200]
}

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Churned"]):
    train_set= data.loc[train_index]
    test_set= data.loc[test_index]

test_set_label= test_set["Churned"].copy()
test_set_input= test_set.drop("Churned", axis=1)

train_set_label= train_set["Churned"].copy()
train_set_input= train_set.drop("Churned", axis=1)

col_1= train_set_input.columns.tolist()
obj_attribs=[]
for col_2 in col_1:
    if train_set_input[col_2].dtype=="object":
        obj_attribs.append(col_2)

num_attribs=[]
for col_2 in col_1:
    if train_set_input[col_2].dtype!="object":
        num_attribs.append(col_2)

pipeline= build_pipeline(obj_attribs, num_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

log_reg= LogisticRegression(random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label)
log_reg = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Model trained on the dataset !")

test_set_input_arr= pipeline.transform(test_set_input)
predictions= log_reg.predict(test_set_input_arr)
test_set["Predicted_Churn"]= predictions
model_accuracy= log_reg_accuracy(predictions, test_set_label)
print(f"The accuracy in this logistic regression model is {model_accuracy}%")
print(test_set)