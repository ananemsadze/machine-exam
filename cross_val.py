import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=7, n_redundant=13, flip_y=0.3, random_state=1)
mymodel = RandomForestClassifier()
parameters = {"n_estimators": [20, 50, 70, 80, 90, 100], "max_depth": [5, 6, 7, 8, 9, 10]}
hybrid = GridSearchCV(mymodel, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_, hybrid.best_score_)




# import pandas as pd
# import numpy as np
# from sklearn.datasets import make_classification, make_regression
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
#
# X, y = make_classification(n_samples=2000, n_features=20, n_informative=7, n_redundant=13, flip_y=0.3, random_state=1)
# mymodel = RandomForestClassifier()
# scores = cross_val_score(mymodel, X, y, scoring='accuracy', cv=20, n_jobs=-1)
#
# print(scores)
# print(np.mean(scores))  # საშუალო
# print(np.std(scores))  # სტანდარტული გადახრა
