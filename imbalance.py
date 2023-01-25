import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

data = pd.read_csv("creditcard.csv")
print(data.head())
# print(data.isnull().any())
# print(data['Class'].value_counts())

y = data['Class'].values
X = data.drop(['Class'],axis=1).values
print(X.shape)
print(Counter(y))

# oversampled
# over = SMOTE()
# X, y = over.fit_resample(X, y)
# print(X.shape)
# print(Counter(y))


# under-sampled
# under = RandomUnderSampler()
# X, y = under.fit_resample(X, y)
# print(X.shape)
# print(Counter(y))

# both
pipe = Pipeline(steps=[("over",SMOTE(sampling_strategy=0.2)), ("under", RandomUnderSampler(sampling_strategy=0.5))])
X, y = pipe.fit_resample(X,y)
print(X.shape)
print(Counter(y))