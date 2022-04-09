import pandas 
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import numpy as np

data=pandas.read_csv('Employee_Data - Sheet1.csv')
d = data.head()
print(d)

desc = data.describe()
print(desc)

inf0 = data.info()
print(data.isna().sum())
d1=data
plt.boxplot(d1['Average montly hours'])
plt.title("Box plot")
plt.show()

plt.boxplot(d1['Average daily hours'])
plt.title("Box plot")
plt.show()

sns.pairplot(data)
plt.show()

data.describe()
print(data.describe())

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(data)
label = kmeans.predict(data)
print(data)

dailyHours = data['Average daily hours']
monthlyHours = data['Average montly hours']
plt.scatter(dailyHours,monthlyHours, c=label)
plt.show()

from sklearn.metrics import silhouette_score
kmeans_score = silhouette_score(data, kmeans.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % kmeans_score)

wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,7),wcss,'-o')
plt.title('Elbow method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS Value')
plt.show()