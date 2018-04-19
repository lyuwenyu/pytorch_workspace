from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict


x = np.random.rand(1000, 100) ## here should use meaningful features

kmeans = KMeans(n_clusters=10, random_state=0, precompute_distances=True, max_iter=50, n_init=20).fit(x) ### n cluster point


cluster = defaultdict(list)
cluster_density = dict()


for i in range(len(kmeans.labels_)):
    
    cluster[kmeans.labels_[i]].append(i)
    
    if kmeans.labels_[i] in cluster_density:
        
        cluster_density[kmeans.labels_[i]] = cluster_density[kmeans.labels_[i]] + np.linalg.norm(x[i]- kmeans.cluster_centers_[kmeans.labels_[i]])
        
    else:
        
        cluster_density[kmeans.labels_[i]] = np.linalg.norm(x[i] - kmeans.cluster_centers_[kmeans.labels_[i]])
        

for i in set(kmeans.labels_):
    
    cluster_density[i] = cluster_density[i] / len(cluster[i])

curriculum_cluster = sorted(cluster_density.items(), key=lambda x: x[1])



for i in range(len(curriculum_cluster)):

    index = []
    
    for j in range(i):
    
        index += cluster[ curriculum_cluster[j][0] ]

    data = x[index]

    ### 
    # train model here
    ###

    print(index)