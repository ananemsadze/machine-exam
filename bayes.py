import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

heart = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv")
# print(heart.head())
y = heart['target'].values
# X = heart.drop('target', axis=1).values
# print(heart['target'].value_counts(normalize=True))
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

## LogisticRegression
# mymodel = LogisticRegression(max_iter=100000, n_jobs=-1, C=0.001)
# mymodel.fit(X_train, y_train)
# print(mymodel.score(X_test, y_test))

## GaussianNB / Bayes Theory (Prior)
# mymodel = GaussianNB(priors=[0.55, 0.45])
# mymodel.fit(X_train, y_train)
# print(mymodel.score(X_test, y_test))

## MultinomialNB
X = heart[["age", "sex", "cp", "slope", "thal"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
mymodel = MultinomialNB()
mymodel.fit(X_train, y_train)

# print(mymodel.score(X_test, y_test))
y_predicted = mymodel.predict(X_test)
print(classification_report(y_test, y_predicted))