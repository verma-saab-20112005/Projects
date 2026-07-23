import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

def model_accuracy(predictions, test_set_label):
    count=0
    for i in range(len(test_set_label)):
        if predictions[i]==test_set_label.iloc[i]:
            count+=1
    accuracy= float((count/len(predictions))*100)
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

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

data= pd.read_csv("Performance.csv")

object_cols = data.select_dtypes(include=["object"]).columns
for col_name in object_cols:
    data[col_name] = data[col_name].fillna(np.nan).astype("object")

if data["Performance_Impact"].dtype=="object":
    data["Performance_Impact"]= data["Performance_Impact"].astype("string")

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Weekly_Gaming_Hours"]):
    train_set= data.loc[train_index]
    test_set= data.loc[test_index]

train_set_label= train_set["Performance_Impact"].copy()
train_set_input= train_set.drop("Performance_Impact", axis=1)

test_set_label= test_set["Performance_Impact"].copy()
test_set_input= test_set.drop("Performance_Impact", axis=1)

col_1= train_set_input.columns.tolist()

obj_attribs=[]
num_attribs=[]
for i in range(len(col_1)):
    if train_set_input[col_1[i]].dtype=="object":
        obj_attribs.append(col_1[i])
for i in range(len(col_1)):
    if train_set_input[col_1[i]].dtype!="object":
        num_attribs.append(col_1[i])

pipeline= build_pipeline(obj_attribs, num_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

label_encoder= LabelEncoder()
train_set_label_arr= label_encoder.fit_transform(train_set_label)

xgb_model= XGBClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label_arr)
xgb_model = grid_search.best_estimator_
print("The model is trained and is ready for predictions.")

test_set_input_arr= pipeline.transform(test_set_input)
predictions_arr= xgb_model.predict(test_set_input_arr)
predictions= label_encoder.inverse_transform(predictions_arr)
test_set["Predicted_Performance_Impact"]= predictions
accuracy= model_accuracy(predictions, test_set_label)
print(f"The accuracy of prediction of Performance Impact in this data is {accuracy}%.\n")
print(test_set)