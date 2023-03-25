import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
dataset = pd.read_csv('data.csv')
dataset.head()
 
# data preprocessing
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder() 
X[:,3] = labelEncoder_X.fit_transform(X[ : , 3])
  
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
 
     
X = X[:, 1:]
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 
 
# predicting the test set results
y_pred = regressor.predict(X_test)
  
y_test
 
y_pred
