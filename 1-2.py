import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("emails.csv")
print(data.head())
y = data['Prediction'].values
X = data.drop('Prediction', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
mymodel = LogisticRegression(max_iter=10000)
mymodel.fit(X_train, y_train)
print("train",mymodel.score(X_train, y_train))
print("test",mymodel.score(X_test, y_test))

selector = SelectKBest(k=4, score_func=f_classif)
selector.fit(X, y)
X = X.iloc[:,selector.get_support()]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
mymodel = LogisticRegression(max_iter=10000)
mymodel.fit(X_train, y_train)
print("train",mymodel.score(X_train, y_train))
print("test",mymodel.score(X_test, y_test))