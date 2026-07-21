import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import root_mean_squared_error

class DateYMExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        # Set required fitted attributes for sklearn compatibility
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.n_features_in_ = 1
            self.feature_names_in_ = [self.col_name]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        else:
            s = X
        s = pd.to_datetime(s, errors="raise")
        return np.c_[
            s.dt.year.to_numpy(),
            s.dt.month.to_numpy(),
            s.dt.day.to_numpy()
        ]
    
def build_pipeline(date_attribs, num_attribs, cat_attribs):
    date_pipeline= Pipeline([
        ("date", DateYMExtractor(date_attribs))
    ])
    cat_pipeline= Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipeline= Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    full_pipeline= ColumnTransformer([
        ("date", date_pipeline, date_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("num", num_pipeline, num_attribs)
    ])
    return full_pipeline

data= pd.read_excel("Weather_history.xlsx")
col= data.columns.tolist()
for i in range(6):
    if data[col[i]].dtype=="object":
        data[col[i]]=data[col[i]].astype("string")

# for i in range(6):
#     print(data[col[i]].dtype)

data["avg_temp"]= (data["temp_max"] + data["temp_min"])/2
n_bins = 5
data["avg_temp_bin"] = pd.qcut(data["avg_temp"], q=n_bins, duplicates="drop")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["avg_temp_bin"]):
    train_set = data.loc[train_index].drop(columns=["avg_temp_bin", "avg_temp"])
    test_set = data.loc[test_index].drop(columns=["avg_temp_bin", "avg_temp"])

test_set_label= test_set["temp_min"].copy()
test_set_input= test_set.drop("temp_min", axis=1)

train_set_label= train_set["temp_min"].copy()
train_set_input= train_set.drop("temp_min", axis=1)

param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate [web:7][web:16]
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

date_attribs= ["date"]
cat_attribs= ["weather"]
num_attribs= train_set_input.drop(["date", "weather"], axis=1).columns.tolist()

pipeline= build_pipeline(date_attribs, num_attribs, cat_attribs)
train_set_input_arr= pipeline.fit_transform(train_set_input)

xgb_model= XGBRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(train_set_input_arr, train_set_label)
xgb_model = grid_search.best_estimator_

test_set_input_arr= pipeline.transform(test_set_input)
predictions= xgb_model.predict(test_set_input_arr)
test_set["predicted_min_temp"]= predictions
test_set["temp_min"]= test_set_label

xgb_rmse= root_mean_squared_error(test_set_label, predictions)
print(f"The root_mean_squared error of this Minimum temperature prediction model is {xgb_rmse}")
print(test_set)