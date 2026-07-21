import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from xgboost import XGBClassifier

MODEL_FILE= "Training_model.pkl"
PIPELINE_FILE= "Pipeline.pkl"

def build_pipeline(obj_attribs, num_attribs):
    obj_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("Standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("obj", obj_pipeline, obj_attribs),
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

data= pd.read_excel("train.xlsx")
col= data.columns.tolist()
obj_attribs=[]
obj_col=[]
num_attribs=[]
for col_1 in col:
    if data[col_1].dtype=="object":
        obj_col.append(col_1)

for objs in obj_col:
    data[objs]= data[objs].fillna(np.nan).astype("object")

train_set_label= data["Placement_Status"].copy()
train_set_input= data.drop("Placement_Status", axis=1)

train_col= train_set_input.columns.tolist()
for col_1 in train_col:
    if data[col_1].dtype=="object":
        obj_attribs.append(col_1)
for col_1 in train_col:
    if data[col_1].dtype!="object":
        num_attribs.append(col_1)

pipeline= build_pipeline(obj_attribs, num_attribs)

label_encoder= LabelEncoder()
train_set_label_arr= label_encoder.fit_transform(train_set_label)
n_classes = len(np.unique(train_set_label_arr))

train_set_input_arr= pipeline.fit_transform(train_set_input)
xgb_model= XGBClassifier(objective="multi:softmax", num_class=n_classes, random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label_arr)
xgb_model = grid_search.best_estimator_

joblib.dump(xgb_model, MODEL_FILE)
joblib.dump(pipeline, PIPELINE_FILE)
print("The model is trained on the dataset!")