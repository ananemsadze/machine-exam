import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB

data = pd.read_csv("https://raw.githubusercontent.com/UCLSPP/datasets/master/data/Credit.csv")
# print(data.head())

mylabel = LabelEncoder()
# ტექსტურის რიცხვებში გადაყვანა
data["Gender"] = mylabel.fit_transform(data["Gender"])
data["Student"] = mylabel.fit_transform(data["Student"])
data["Married"] = mylabel.fit_transform(data["Married"])
data["Ethnicity"] = mylabel.fit_transform(data["Ethnicity"])
# print(data.head())

# Logistic
y = data["Student"].values
X = data.drop("Student", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

mymodel = LogisticRegression(max_iter=100000, n_jobs=-1, C=0.001)
mymodel.fit(X_train, y_train)
print("Logistic Score is ", mymodel.score(X_test, y_test))  # 0.99, რაც  თითქმის 100 პროცენტია და კარგი შედეგია

# Gaussian
mymodel = GaussianNB()
mymodel.fit(X_train, y_train)
print("Gaussian score is", mymodel.score(X_test, y_test))  # 0.9, კარგია თუმცა ლოჯისტიკურმა უკეთესი შედეგი აჩვენა

# Multi
X = data[["Cards", "Gender", "Ethnicity", "Rating"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
mymodel = MultinomialNB()
mymodel.fit(X_train, y_train)
print("Multi score is", mymodel.score(X_test, y_test))  # 0.9, იგივე შედეგი რაც gaussian

# ყველაზე კარგი შედეგი აჩვენა ლოჯისტიკურმა რეგრესიამ, რომელიც თითქმის მიახლოებული იყო 100 პროცენტს. 0.99