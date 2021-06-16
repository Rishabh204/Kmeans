# Kmeans
#try importing the libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline

#import ur data and have loom at it

Iris_data=pd.read_csv("Iris.csv")
Iris_data.head(10)
Iris_data.describe(include="all")
Iris_data.dtypes

#now to find optimum no. of clustering 
#choose any method u like 
# i used elbow method [wcss] u can use [sse] too

x = Iris_data.iloc[:, [0, 1, 2, 3]].values

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#and go...soo on
