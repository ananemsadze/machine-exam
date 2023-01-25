import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

data = pd.DataFrame(pd.read_html("https://github.com/kb22/Heart-Disease-Prediction/blob/master/dataset.csv")[0])
data.drop("Unnamed: 0", axis=1, inplace=True)
print(data.head())
y = data["target"]
X = data.drop("target", axis=1)

# selector = SelectKBest(score_func=f_classif, k=4)
# selector.fit(X, y)
# selected = selector.fit_transform(X, y)
# print(selector.get_feature_names_out())

pipe = Pipeline(steps=[('selector', SelectKBest(score_func=f_classif)), ('algo', AdaBoostClassifier())])
parameters = {"selector__k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 12, 13]}

hybrid = GridSearchCV(pipe, parameters, scoring='accuracy', cv=2, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_)
print(pipe.named_steps['selector'])
