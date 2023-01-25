import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
# robust მედიანებით აკეთებს სკალირებას
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv(
    "https://raw.githubusercontent.com/PacktPublishing/Regression-Analysis-with-R/master/Chapter03/EscapingHydrocarbons.csv",
    sep=';')

# print(data.head())
# print(data.info())
# print(data.isnull().any())

y = data['AmountEscapingHydrocarbons'].values
X = data.drop('AmountEscapingHydrocarbons', axis=1).values
# print(X.shape)  # raw, dimensions

# Lasso
mymodel = Lasso()
mymodel.fit(X, y)
print(mymodel.score(X, y))

# Pipeline
hybrid = Pipeline(steps=[(("scaler"), StandardScaler()), ("pca", PCA(n_components=3)), ('algo', Lasso())])
hybrid.fit(X, y)
print(hybrid.score(X, y))
# print(hybrid.named_steps['pca'].explained_variance_ratio_[0])
print(np.sum(hybrid.named_steps['pca'].explained_variance_ratio_[0]))
