import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # MiniBatchKMeans ყოფს პატარ-პატარა ნაწილად
from sklearn.datasets import make_blobs

# აგენერირებს მარტო x-ს
# centers - გადაეცემა კოორდინატები წერტილის
X, _ = make_blobs(n_samples=2000, n_features=2, centers=[[1, 5], [8, 10]], shuffle=True, random_state=1)

print(X)
# რამდენი განზომილება (კლასტერის წერტილი იყო)
myKmeans = KMeans(n_clusters=2, max_iter=2000)
myKmeans.fit(X)  # X-ზე დაყრდნობით ითვლის ცენტრებს
y_predicted = myKmeans.predict(X)
# თითოეულ კლასტერს გააფერადებს სხვადასხვა ფრად
plt.scatter(X[:, 0], X[:, 1], c=y_predicted)
plt.show()

#################################################

# cluster std ერთმანეთში რევს წერტილებს
X, _ = make_blobs(n_samples=2000, n_features=2, centers=[[1, 5], [8, 10]], shuffle=True, random_state=1,cluster_std=2)
myKmeans = KMeans(n_clusters=2, max_iter=2000)
myKmeans.fit(X)
# გამოყოფს კლასტერის წერტილს
centers = myKmeans.cluster_centers_
y_predicted = myKmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_predicted)
plt.scatter(centers[:,0], centers[:,1], s=100, c='red')
plt.show()