import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("https://raw.githubusercontent.com/kvinlazy/Dataset/master/drug200.csv")
mylabel = LabelEncoder()

data["Sex"] = mylabel.fit_transform(data["Sex"])
data["BP"] = mylabel.fit_transform(data["BP"])
data["Cholesterol"] = mylabel.fit_transform(data["Cholesterol"])
data["Drug"] = mylabel.fit_transform(data["Drug"])

y = data["Drug"].values
X = data.drop("Drug", axis=1).values
#print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.1)

mymodel = AdaBoostClassifier()
mymodel.fit(X_train, y_train)
print(mymodel.score(X_train, y_train))
print(mymodel.score(X_test, y_test))
# სატრენინგო აბრუნებს 0.8444444444444444, ხოლო სატესტო 0.8, აქედან გამომდინარე სატრენინგოს უფრო კაი შედეგი აქვს

parameters = {"learning_rate": [0.1,0.2,0.3,0.4,0.7,0.9]}
hybrid = GridSearchCV(mymodel, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_score_)  # დააბრუნა 0.8349917081260365, რაც ნორმალური შედეგია

parameters = {"learning_rate": [0.1,0.2,0.3,0.4,0.7,0.9],"n_estimators": [30,40,60]}
hybrid = GridSearchCV(mymodel, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_) # კომბინაცია {'learning_rate': 0.1, 'n_estimators': 30}
print(hybrid.best_score_) # 0.8349917081260365, იგივეა, რაც n_estimators-ის გარეშე