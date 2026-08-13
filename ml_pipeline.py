import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()

X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=0 )




