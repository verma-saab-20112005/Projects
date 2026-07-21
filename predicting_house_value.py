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

data= pd.read_csv("Housing.csv")
# print(data)

data["income_cat"]= pd.cut(data["median_income"], bins= [0,1.5,3.0,4.5,6.0,np.inf], labels= [1,2,3,4,5])
# print(data)

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set= data.loc[train_index].drop("income_cat", axis=1)
    strat_test_set= data.loc[test_index].drop("income_cat", axis=1)

# print(strat_test_set)

housing_labels= strat_train_set["median_house_value"].copy()
housing= strat_train_set.drop("median_house_value", axis=1)
# print(housing, housing_labels)

num_attribs_train= housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs_train= ["ocean_proximity"]
# print(num_attribs, cat_attribs)

num_pipeline= Pipeline([
    ("impute", SimpleImputer(strategy= "median")),
    ("standardize", StandardScaler()),
])

cat_pipeline= Pipeline([
    ("onehot", OneHotEncoder(handle_unknown= "ignore"))
])

full_pipeline= ColumnTransformer([
    ("num_handling",num_pipeline, num_attribs_train),
    ("cat_handling", cat_pipeline, cat_attribs_train),
])

housing_cat= full_pipeline.fit_transform(housing)
# print(housing_cat.size)

lin_reg= LinearRegression()
lin_reg.fit(housing_cat, housing_labels)
lin_preds= lin_reg.predict(housing_cat)
# lin_rmse= root_mean_squared_error(housing_labels, lin_preds)
lin_rmses= -cross_val_score(lin_reg, housing_cat, housing_labels, scoring= "neg_root_mean_squared_error", cv= 10)
print(f"The root mean squared error for the linear regression model is:-\n{pd.Series(lin_rmses).describe()}")

dec_reg= DecisionTreeRegressor()
dec_reg.fit(housing_cat, housing_labels)
dec_preds= dec_reg.predict(housing_cat)
# dec_rmse= root_mean_squared_error(housing_labels, dec_preds)
dec_rmses= -cross_val_score(dec_reg, housing_cat, housing_labels, scoring= "neg_root_mean_squared_error", cv=10)
print(f"The root mean squared error for the decisiontree regression model is:-\n{pd.Series(dec_rmses).describe()}")

randomforest_reg= RandomForestRegressor()
randomforest_reg.fit(housing_cat, housing_labels)
randomforest_preds= randomforest_reg.predict(housing_cat)
# randomforest_rmse= root_mean_squared_error(housing_labels, randomforest_preds)
randomforest_rmses= -cross_val_score(randomforest_reg, housing_cat, housing_labels, scoring= "neg_root_mean_squared_error", cv= 10)
print(f"The root mean squared error for the randomforest regression model is:-\n{pd.Series(randomforest_rmses).describe()}")
