import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE= "model.pkl"
PIPELINE_FILE= "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline= Pipeline([
    ("impute", SimpleImputer(strategy= "median")),
    ("standardize", StandardScaler()),
    ])

    cat_pipeline= Pipeline([
        ("onehot", OneHotEncoder(handle_unknown= "ignore"))
    ])

    full_pipeline= ColumnTransformer([
        ("num_handling",num_pipeline, num_attribs),
        ("cat_handling", cat_pipeline, cat_attribs),
    ])
    return full_pipeline

data= pd.read_csv("Housing.csv")
# print(data)

data["income_cat"]= pd.cut(data["median_income"], bins= [0,1.5,3.0,4.5,6.0,np.inf], labels= [1,2,3,4,5])
# print(data)

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set= data.loc[train_index].drop("income_cat", axis=1)
    strat_test_set= data.loc[test_index].drop("income_cat", axis=1)
housing_test= strat_test_set.drop("median_house_value", axis=1)
housing_test.to_excel("Housing_Test_Data.xlsx", index= False)

if not os.path.exists(MODEL_FILE):

    housing_labels= strat_train_set["median_house_value"].copy()
    housing_features= strat_train_set.drop("median_house_value", axis=1)

    num_attribs= housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs= ["ocean_proximity"]

    pipeline= build_pipeline(num_attribs, cat_attribs)
    housing_prepared= pipeline.fit_transform(housing_features)
    # print(housing_prepared)

    model= RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained !")
else:
    model= joblib.load(MODEL_FILE)
    pipeline= joblib.load(PIPELINE_FILE)

    input_data= pd.read_excel("Housing_Test_Data.xlsx")
    transformed_data= pipeline.transform(input_data)
    predictions= model.predict(transformed_data)
    input_data["median_house_value"]= predictions
    input_data.to_excel("Housing_Tset_Output_Data.xlsx", index=False)
    # print("Inference saved to Housing_Tset_Output_Data.xlsx file !")
    print(input_data)
    randomforest_rmse= root_mean_squared_error(strat_test_set["median_house_value"], predictions)
    print(f"The error found in the values predicted by this model is {randomforest_rmse}")