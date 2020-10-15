from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

data = pd.read_csv("Cat_stats.csv")

data_clean = data.dropna()

cluster = data_clean[['Body_length', 'Tail_length', 'Height', 'Tail_texture', 'Coat_colour']]
cluster.describe()

clustervar=cluster.copy()
clustervar['Body_length'] = preprocessing.scale(clustervar['Body_length'].astype('float64'))
clustervar['Tail_length'] = preprocessing.scale(clustervar['Tail_length'].astype('float64'))
clustervar['Height'] = preprocessing.scale(clustervar['Height'].astype('float64'))
clustervar['Tail_texture'] = preprocessing.scale(clustervar['Tail_texture'].astype('float64'))
clustervar['Coat_colour'] = preprocessing.scale(clustervar['Coat_colour'].astype('float64'))

clus_train, clus_test = train_test_split(clustervar, test_size = .3, random_state = 123)
                                                           
from scipy.spatial.distance import cdist
clusters = range(1,10)
meandist = []

for k in clusters:
    model = KMeans(n_clusters = k)
    model.fit(clus_train)
    clusassign = model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis = 1)) / clus_train.shape[0])

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

model3 = KMeans(n_clusters = 3)
model3.fit(clus_train)
clusassign = model3.predict(clus_train)

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x = plot_columns[:, 0], y = plot_columns[:, 1], c = model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

clus_train.reset_index(level = 0, inplace = True)

cluslist = list(clus_train['index'])

labels = list(model3.labels_)

newlist = dict(zip(cluslist, labels))
newlist
newclus = DataFrame.from_dict(newlist, orient = 'index')
newclus
newclus.columns = ['cluster']

newclus.reset_index(level = 0, inplace = True)
merged_train = pd.merge(clus_train, newclus, on = 'index')
merged_train.head(n = 100)
merged_train.cluster.value_counts()

clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


weight_data = data_clean['Weight']
weight_train, weight_test = train_test_split(weight_data, test_size = 0.3, random_state=123)
weight_train1 = pd.DataFrame(weight_train)
weight_train1.reset_index(level = 0, inplace = True)
merged_train_all = pd.merge(weight_train1, merged_train, on = 'index')
sub1 = merged_train_all[['Weight', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

weightmod = smf.ols(formula='Weight ~ C(cluster)', data=sub1).fit()
print (weightmod.summary())

print ('means for Weight by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for Weight by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['Weight'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
