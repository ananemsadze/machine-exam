import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

wine = pd.read_csv(
    "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv")
# print(wine.head())
wine.drop("Wine", axis=1, inplace=True)
print(wine.head())
# ყველა სვეტი გადადის სტანდარტულ ფორმატში
myscaler = StandardScaler()
wine = myscaler.fit_transform(wine)

# დადის 2 განზომილებაზე
mytsne = TSNE(n_components=2, perplexity=40, n_iter=2000)
wine_embedding = mytsne.fit_transform(wine)
plt.scatter(wine_embedding[:,0], wine_embedding[0:,1])
plt.show()