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
def model_accuracy(predictions, test_set_label):
    count=0
    for i in range(len(predictions)):
        if predictions[i]==test_set_label.iloc[i]:
            count+=1
    accuarcy= float((count/len(predictions))*100)
    return accuarcy

data= pd.read_excel("test.xlsx")
test_set_label= data["Placement_Status"].copy()
test_set_input= data.drop("Placement_Status", axis=1)

pipeline= joblib.load(PIPELINE_FILE)
xgb_model= joblib.load(MODEL_FILE)

col= test_set_input.columns.tolist()
for col_1 in col:
    if test_set_input[col_1].dtype=="object":
        test_set_input[col_1]= test_set_input[col_1].fillna(np.nan).astype("object")
label_encoder= LabelEncoder()
test_set_label_arr= label_encoder.fit_transform(test_set_label)

test_set_input_arr= pipeline.transform(test_set_input)
predictions_encoded= xgb_model.predict(test_set_input_arr)
predictions= label_encoder.inverse_transform(predictions_encoded)
accuracy= model_accuracy(predictions, test_set_label)
data["Predicted_placement_status"]= predictions
print(data)
print(f"The accuracy of this model is {accuracy}%")