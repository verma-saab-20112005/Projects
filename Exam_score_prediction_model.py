import pandas as pd
import numpy as np
import os 
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error

MODEL_FILE= "Exam_Predicter.pkl"
PIPELINE_FILE= "Pipeline_for_predicter.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])

    cat_pipeline= Pipeline([
        ("onehot", OneHotEncoder(handle_unknown= "ignore"))
    ])

    full_pipeline= ColumnTransformer([
        ("num_attribs", num_pipeline, num_attribs),
        ("cat_attribs", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

data= pd.read_csv("Exam_Score_Prediction.csv")
data["study_hours_cat"]= pd.cut(data["study_hours"], bins= [0,1,2,3,4,5,6,7,np.inf], labels= [1,2,3,4,5,6,7,8])

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["study_hours_cat"]):
    strat_train_set= data.loc[train_index].drop("study_hours_cat", axis=1)
    strat_test_set= data.loc[test_index].drop("study_hours_cat", axis=1)
exam_test_data= strat_test_set.drop("exam_score", axis=1)
exam_test_data.to_excel("Exam_Pred_Test.xlsx")
# print(strat_test_set)

if not os.path.exists(MODEL_FILE):
    exam_score= strat_train_set["exam_score"].copy()
    exam_param= strat_train_set.drop("exam_score", axis=1)

    num_attribs= ['student_id','age', 'study_hours','class_attendance','sleep_hours']
    cat_attribs= ['gender','course','internet_access','sleep_quality','study_method','facility_rating','exam_difficulty']

    pipeline= build_pipeline(num_attribs, cat_attribs)
    exam_param_cat= pipeline.fit_transform(exam_param)

    model= RandomForestClassifier(random_state=42)
    model.fit(exam_param_cat, exam_score)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("model and pipeline both saved to pkl file")
else:
    model= joblib.load(MODEL_FILE)
    pipeline= joblib.load(PIPELINE_FILE)

    test_data_param= pd.read_excel("Exam_Pred_Test.xlsx")
    transformed_data= pipeline.transform(test_data_param)
    predictions= model.predict(transformed_data)
    test_data_param["exam_score_predicted"]= predictions
    # test_data_param["exam_score"]= strat_test_set["exam_score"]
    test_data_param.to_excel("Predicted_exam_scores.xlsx")
    randomforest_rmse= root_mean_squared_error(strat_test_set["exam_score"], predictions)
    print(f"The rmse error in predicting the marks of the students is {randomforest_rmse}")
    print(test_data_param, strat_test_set)
    # print(strat_test_set["exam_score"])
    # print(set(test_data_param["exam_score_predicted"]))