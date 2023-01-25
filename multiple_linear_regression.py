import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv")
# print(data.head())
# y = data["Profit"].values
# X = data["R&D Spend"].values

# plt.scatter(X, y, s=10)
# plt.show()
## მთელი მონაცემები აიღო და კონვერტაციის შემდეგ გადააქცია ორ განზომილებიან მატრიცად
# X = X.reshape(-1, 1)
# mymodel = LinearRegression()
# mymodel.fit(X, y)
# y_predicted = mymodel.predict(X)
# plt.scatter(X, y, s=15)
# plt.plot(X, y_predicted)
# plt.show()

## ტექსტის კონვერტაცია
mylabel = LabelEncoder()
data["State"] = mylabel.fit_transform(data["State"])
y = data["Profit"].values
X = data.iloc[:, [0, 2, 3]].values  # X = data[["R&D Spend", "Marketing Spend", "State"]]
mymodel = LinearRegression()
mymodel.fit(X, y)
print(mymodel.score(X, y))  # y - ის ცვლილების აღწერილობა x - ის მიმართ
