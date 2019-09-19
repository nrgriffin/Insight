import pandas as pd
import numpy as np
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

%matplotlib inline

data = pd.read_csv("~/Documents/jtnp_routesAndDesc_joined_reducedv2.csv")
coords = data.as_matrix(columns=['latitude', 'longitude'])
kms_per_radian = 6371.0088
epsilon = .1 / kms_per_radian
min_sample=10
db = DBSCAN(eps=epsilon, min_samples=min_sample, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

#Find the point in each cluster that is closest to its centroid
centermost_points = []
for cluster in clusters.iteritems():
    if len(cluster[1]) >= min_sample:
        centermost_points.append(get_centermost_point(cluster[1]))
        #print(centermost_points)

lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})

rs = rep_points.apply(lambda row: data[(data['latitude']==row['lat']) & (data['longitude']==row['lon'])].iloc[0], axis=1)
rs_centroids = rs[['latitude','longitude']]
#rs_centroids

campsites = pd.read_csv("~/Documents/jtnp_campsite_coords.csv")
y = campsites['latitude']
z = campsites['longitude']
n = campsites['campsite']

fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
data_scatter = ax.scatter(data['longitude'], data['latitude'], c='k', alpha=0.9, s=3)
camp_scatter = ax.scatter(campsites['longitude'], campsites['latitude'], c='r', alpha=1, s=30)
ax.set_title('All Routes vs DBSCAN Route Clusters')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([data_scatter, rs_scatter, camp_scatter], ['Routes', 'Route Clusters', 'Campsites'], loc='upper right')
for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))
plt.show()

print('Number of clusters: {}'.format(num_clusters))
