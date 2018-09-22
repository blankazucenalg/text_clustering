import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# #############################################################################
# Read data from CSV
pwd = '/home/azu/Proyectos/blanch/datajam/datasets/citi-clustering-comercios/'
#read csv
df = pd.read_csv(pwd + 'Comercios.csv')
print(df['sucio'])
# remove whitespaces and set to lowercase
df['sucio'] = [x.replace('.', ' ').replace('*', ' ').strip().lower() for x in df['sucio']]

# take only first 100 for testing purposes
test = df['sucio'][1:100]
print(test)

import sklearn.cluster
import distance

words = test #Replace this line with data
words = np.asarray(words) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

print(lev_similarity)

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))
