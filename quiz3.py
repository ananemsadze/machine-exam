import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
mylabel = LabelEncoder()
data["sex"] = mylabel.fit_transform(data["sex"])
data["smoker"] = mylabel.fit_transform(data["smoker"])
data["region"] = mylabel.fit_transform(data["region"])

y = data['charges'].values
X = data.drop('charges', axis=1).values

mymodel = LinearRegression()
mymodel.fit(X, y)
print(mymodel.score(X, y)) # სქორი გამოვიდა 0.7507372027994937, რაც ნორმალური მაჩვენებელია



hybrid = Pipeline(steps=[(("scaler"), StandardScaler()), ("pca", PCA(n_components=3)), ('algo', LinearRegression())])
hybrid.fit(X, y)
print(hybrid.score(X, y))
# სქორი გამოვიდა 0.33881171316333913, რაც ჩემი აზრით არც თუ ისე კარგი შედეგია, რადგან დაბალი მაჩვენებელია, შესაბამისად გამოდის რომ linear regression-მა ცალკე უფრო კარგი შედეგი აჩვენა, ვიდრე პაიპლაინის გამოყენებამ