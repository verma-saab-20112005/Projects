import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def model_accuracy(predictions, y_test):
    count=0
    for i in range(len(predictions)):
        if predictions[i]==y_test.iloc[i]:
            count+=1
    accuracy= float((count/(len(y_test)))*100)
    return accuracy

def build_pipeline(num_attribs):
    full_pipeline= Pipeline([
        ("Impute", SimpleImputer(strategy="median")),
        ("Standardize", StandardScaler())
    ])
    return full_pipeline

data= pd.read_csv("letter-recognition.csv")
data["letter"]= data["letter"].astype("string")

X= data.iloc[:,1:]
y= data.iloc[:,0]
x_train, x_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)

label_encoder= LabelEncoder()
y_train_arr= label_encoder.fit_transform(y_train)

num_attribs= x_train.columns.tolist()
pipeline= build_pipeline(num_attribs)

x_train_arr= pipeline.fit_transform(x_train)
knn= KNeighborsClassifier()
knn.fit(x_train_arr, y_train_arr)
print("The model is trained on the dataset!")

x_test_arr= pipeline.transform(x_test)
predictions_arr= knn.predict(x_test_arr)
predictions= label_encoder.inverse_transform(predictions_arr)
knn_accuracy= model_accuracy(predictions, y_test)
print(f"The accuracy of this model is {knn_accuracy:.2f}%")