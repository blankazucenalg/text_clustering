import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import distance

# #############################################################################
# Read data from CSV
pwd = '/home/azu/Proyectos/blanch/datajam/datasets/citi-clustering-comercios/'
#read csv
df = pd.read_csv(pwd + 'Comercios.csv')
print(df['sucio'])

def stripDash(x) :
    dashed = x.split(' - ')
    if len(dashed) > 1 and dashed[0].isdigit():
        return dashed[1] 
    else:
        return x
# remove whitespaces and set to lowercase
df['sucio'] = [stripDash(x.replace('"', '').replace('#', '').replace('.', ' ').replace('*', ' ').strip().lower()) for x in df['sucio']]
df = df.sort_values('sucio')

print(df)
# take only first 200 for testing purposes
test = df['sucio'][0:200]
print(test)

def string_metric(w1, w2) :
    # return distance.sorensen(w1, w2)
    return distance.levenshtein(w1,w2)

words = test #Replace this line with data
words = np.asarray(words) #So that indexing with a list will work
lev_similarity = -1*np.array([[string_metric(w1, w2) for w1 in words] for w2 in words])

print(lev_similarity)

affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)

print(affprop)
cluster_centers_indices = affprop.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
labels = affprop.labels_

for cluster_id in np.unique(labels):
    exemplar = words[cluster_centers_indices[cluster_id]]
    cluster = np.unique(words[np.nonzero(labels==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))


import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = lev_similarity[cluster_centers_indices[k]]
    plt.plot(lev_similarity[class_members, 0], lev_similarity[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in lev_similarity[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()