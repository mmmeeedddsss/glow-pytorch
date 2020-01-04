import os

import hdbscan
import pickle
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
import shutil
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

calc_clusters = False

with open('z_path_map_stanford.pkl', 'rb') as f:
    mp = pickle.load(f)

inv_mp = {}
for k, v in mp.items():
    inv_mp[tuple(v.flatten())] = k

data = [i.flatten() for i in list(mp.values())]

if calc_clusters:
    print('starting clustering')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
    clusterer = AgglomerativeClustering(n_clusters=1024)
    clusters = clusterer.fit_predict(data)
    with open('clusters_stanford.pkl', 'wb+') as f:
        pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)
else:
    with open('clusters_stanford.pkl', 'rb') as f:
        clusters = pickle.load(f)

cc = {}
for c in clusters:
    if c not in cc:
        cc[c] = 0
    cc[c] += 1
print(cc)

"""
color_palette = sns.color_palette('Paired', 12)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusters]
cluster_member_colors = [sns.desaturate(x, 1.0) for x in cluster_colors]


print('starting TSNE')
projection = TSNE().fit_transform(data)
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.show()
"""
print('Made {} clusters from {} points'.format(max(clusters), len(data)))

for i, point in tqdm(enumerate(data)):
    filename = inv_mp[ tuple(point) ]

    if not os.path.exists('/home/mert/Downloads/clustered_stanford/{}'.format(clusters[i])):
        os.makedirs('/home/mert/Downloads/clustered_stanford/{}'.format(clusters[i]))

    shutil.copy2('/home/mert/Downloads/cars_train/{}'.format(filename),
                 '/home/mert/Downloads/clustered_stanford/{}/{}'.format(clusters[i], filename))








