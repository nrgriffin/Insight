from flask import Flask
from flask import request, render_template, flash, redirect, send_from_directory
from boulderapp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import numpy as np
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
import logging
import sklearn
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from collections import Counter
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import scipy
from scipy.spatial import distance
from scipy.spatial import cKDTree
from random import randint

user = 'nicholasgriffin' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)


@app.route('/')
def boulderjoshua_input():
    return render_template("input.html")

@app.route('/output')
def boulderjoshua_output():
    diffrate1 = request.args.get('person1')
    diffrate2 = request.args.get('person2')
    diffrate3 = request.args.get('person3')

    key1 = request.args.get('keyword1')
    key2 = request.args.get('keyword2')
    key3 = request.args.get('keyword3')

    rinfo = pd.read_csv("~/Documents/jtnp_routesAndDesc_joined_reducedv2_KEYWORDS.csv")
    routes = rinfo.loc[(rinfo['kwone'] != key1) & (rinfo['kwtwo'] != key1) & (rinfo['kwthree'] != key1) & (rinfo['kwone'] != key2) & (rinfo['kwtwo'] != key2) & (rinfo['kwthree'] != key2) & (rinfo['kwone'] != key3) & (rinfo['kwtwo'] != key3) & (rinfo['kwthree'] != key3)]
    coords = routes.as_matrix(columns=['latitude', 'longitude'])
    kms_per_radian = 6371.0088
    epsilon = .1 / kms_per_radian
    min_sample=10
    db = DBSCAN(eps=epsilon, min_samples=min_sample, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    #print('Number of clusters: {}'.format(num_clusters))

    # This takes the coordinates of routes from a given cluster, returns a reduced list of unique coordinates
    # (no overlap/duplicates), and creates a new pandas data frame with just the routes in this cluster.

    def delete_duplicate_pairs(*arrays):
        unique = set()
        arrays = list(arrays)
        n = range(len(arrays))
        index = 0
        for pair in zip(*arrays):
            if pair not in unique:
                unique.add(pair)
                for i in n:
                    arrays[i][index] = pair[i]
                index += 1
        return [a[:index] for a in arrays]

    # Based on centroids, this will take out clusters with less than 2 of each input category

    minimums = list()

    for i in range(len(clusters)-1):
        clust0 = clusters[i]
        ai1=clust0[:,0]
        ai2=clust0[:,1]
        ai1, ai2 = delete_duplicate_pairs(ai1, ai2)
        clustsub = pd.DataFrame()
        for j in range(len(ai1)):
            xaoi = ai1[j]
            xa1i = ai2[j]
            rsub10 = routes.loc[(routes['latitude'] == xaoi) & (routes['longitude'] == xa1i)]
            clustsub = clustsub.append(rsub10)
        dfc = pd.DataFrame(clustsub['diffcategory'].value_counts())
        dfc.columns=['catcount']
        #print(dfc)
        if len(dfc) < 3:
            minimums.append(0)
        elif dfc['catcount'][diffrate1] >= 2 and dfc['catcount'][diffrate2] >= 2 and dfc['catcount'][diffrate3] >= 2:
            minimums.append(1)
        else:
            minimums.append(0)
    minimums.append(0)

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

    rs = rep_points.apply(lambda row: routes[(routes['latitude']==row['lat']) & (routes['longitude']==row['lon'])].iloc[0], axis=1)
    rs_centroids = rs[['latitude','longitude']]

    minimums2 = pd.DataFrame(minimums)
    minimums2.columns = ['bin']
    rs_centroids2 = rs_centroids.join(minimums2)
    rs_centroids3 = rs_centroids2[rs_centroids2['bin'] > 0]
    del rs_centroids3['bin']
    rs_centroids3.index = range(len(rs_centroids3.index))

    #pull 'birth_month' from input field and store it
    dinput = request.args.get('input')
    dinput2 = dinput[:3]
    dinput3 = dinput[5:]
    dinputdate = "01"
    dinput4 = dinput2 + dinputdate + dinput3
    print(dinput4)
    forecastabbrev10 = pd.read_csv("~/Documents/campgroundpredictions_twoyears_SARIMA.csv")
    campsites = pd.read_csv("~/Documents/jtnp_campsite_coords.csv")
    campsites.columns=['variable','latitude','longitude']
    az = forecastabbrev10.loc[forecastabbrev10['date'] == dinput4]
    az2=az[['Belle','BlackRock','Cottonwood','HiddenValley','IndianCove','JumboRocks','Ryan','WhiteTank']]
    az3=pd.DataFrame(az2)
    az3.columns=['Belle','Black Rock','Cottonwood','Hidden Valley','Indian Cove','Jumbo Rocks','Ryan','White Tank']
    az31=pd.melt(az2)
    az3 = pd.merge(az31, campsites, on='variable')
    az3=az3.sort_values('value')
    az3=az3.reset_index()
    bestcamp = (az3['variable'][0])
    bcocc = (az3['value'][0])
    bcoccp = bcocc * 100
    bcoccper = ("%.2f" % bcoccp)
    bestcamp2 = (az3['variable'][1])
    bcocc2 = ("%.3f" % az3['value'][1])
    bestcamp3 = (az3['variable'][2])
    bcocc3 = ("%.3f" % az3['value'][2])
    print(bestcamp)
    print(bcocc)
    print(bestcamp2)
    print(bcocc2)
    print(bestcamp3)
    print(bcocc3)


    # Finding the Day 1 Climbing Clusters
    y10=az3['latitude'][0]
    x10=az3['longitude'][0]
    array = np.array([y10, x10])
    rs_c = rs_centroids3.as_matrix()
    points_ref = rs_c
    tree = cKDTree(points_ref)
    _, idx1 = tree.query(array, k=1)
    pref1 = points_ref[idx1]
    _, idx2 = tree.query(pref1, k=2)
    pref2 = points_ref[idx2]
    print(pref2)

    # Printing out the individual route coordinates for the cluster closest to campsite (Day 1, Location 2)
    pr4 = points_ref[idx2]
    climb1 = pr4[0,:]
    cl1lat = climb1[0]
    cl1long = climb1[1]
    centroid_row = rs_centroids3.loc[rs_centroids3['latitude'] == climb1[0]]
    cri = centroid_row.index[0]
    c1routes = clusters[cri] # c1routes = list of coordinates for closest cluster

    # Printing out the individual route coordinates for the cluster 2nd closest to campsite (Day 1, Location 1)
    climb2 = pr4[1,:]
    cl2lat = climb2[0]
    cl2long = climb2[1]
    centroid_row2 = rs_centroids3.loc[rs_centroids3['latitude'] == climb2[0]]
    cri2 = centroid_row2.index[0]
    c2routes = clusters[cri2] # c2routes = list of coordinates for 2nd closest cluster

    # Finding the Day 2 Climbing Clusters
    # Need to set the array as a random coordinate pair from rs_c
    for q in range(1):
        value = randint(0, len(rs_c)-1)
        print(value)
    pointd2 = rs_c[value]
    tree = cKDTree(points_ref)
    _, idx3 = tree.query(pointd2, k=1)
    pref5 = points_ref[idx3]
    _, idx4 = tree.query(pref5, k=2)
    pref6 = points_ref[idx4]

    # Printing out the individual route coordinates for Day 2, Location 1
    pr5 = points_ref[idx4]
    climb3 = pr5[0,:]
    cl3lat = climb3[0]
    cl3long = climb3[1]
    centroid_row3 = rs_centroids3.loc[rs_centroids3['latitude'] == climb3[0]]
    cri3 = centroid_row3.index[0]
    c3routes = clusters[cri3]

    # Printing out the individual route coordinates for Day 2, Location 2
    climb4 = pr5[1,:]
    cl4lat = climb4[0]
    cl4long = climb4[1]
    centroid_row4 = rs_centroids3.loc[rs_centroids3['latitude'] == climb4[0]]
    cri4 = centroid_row4.index[0]
    c4routes = clusters[cri4]

    # Create new data frame for each Day + Location to subset from for output
    # Day 1, Location 1
    a1=c2routes[:,0]
    a2=c2routes[:,1]
    a1, a2 = delete_duplicate_pairs(a1, a2)

    rsubA = pd.DataFrame() # Data frame with the route information for D1, L1

    for i in range(len(a1)):
        xao = a1[i]
        xa1 = a2[i]
        rsub1 = routes.loc[(routes['latitude'] == xao) & (routes['longitude'] == xa1)]
        rsubA = rsubA.append(rsub1)
    numroutes1 = len(rsubA)
    rsubAurl = (rsubA['url'])
    rsubnames = list(rsubA['name'])[0:numroutes1]

    # Day 1, Location 2
    b1=c1routes[:,0]
    b2=c1routes[:,1]
    b1, b2 = delete_duplicate_pairs(b1, b2)

    rsubB = pd.DataFrame() # Data frame with the route information for D1, L1

    for i in range(len(b1)):
        xbo = b1[i]
        xb1 = b2[i]
        rsub2 = routes.loc[(routes['latitude'] == xbo) & (routes['longitude'] == xb1)]
        rsubB = rsubB.append(rsub2)
    numroutes2 = len(rsubB)
    rsubBurl = (rsubB['url'])
    rsubnames2 = list(rsubB['name'])[0:numroutes2]

    # Day 2, Location 1
    c1=c3routes[:,0]
    c2=c3routes[:,1]
    c1, c2 = delete_duplicate_pairs(c1, c2)

    rsubC = pd.DataFrame() # Data frame with the route information for D1, L1

    for i in range(len(c1)):
        xco = c1[i]
        xc1 = c2[i]
        rsub3 = routes.loc[(routes['latitude'] == xco) & (routes['longitude'] == xc1)]
        rsubC = rsubC.append(rsub3)
    numroutes3 = len(rsubC)
    rsubCurl = (rsubC['url'])
    rsubnames3 = list(rsubC['name'])[0:numroutes3]

    # Day 2, Location 2
    d1=c4routes[:,0]
    d2=c4routes[:,1]
    d1, d2 = delete_duplicate_pairs(d1, d2)

    rsubD = pd.DataFrame() # Data frame with the route information for D1, L1

    for i in range(len(d1)):
        xdo = d1[i]
        xd1 = d2[i]
        rsub4 = routes.loc[(routes['latitude'] == xdo) & (routes['longitude'] == xd1)]
        rsubD = rsubD.append(rsub4)
    numroutes4 = len(rsubD)
    rsubDurl = (rsubD['url'])
    rsubnames4 = list(rsubD['name'])[0:numroutes4]


    #just select the Cesareans  from the birth dtabase for the month that the user inputs
    #query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
    #print (query)
    #query_results=pd.read_sql_query(query,con)
    #print (query_results)
    #births = []
    #for i in range(0,query_results.shape[0]):
    #    births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    #    the_result = ModelIt(patient,births)
    return render_template("output.html", bestcamp = bestcamp, bcoccper = bcoccper,  bestcamp2 = bestcamp2, bcocc2 = bcocc2,  bestcamp3 = bestcamp3, bcocc3 = bcocc3, cl1lat = cl1lat, cl1long = cl1long, cl2lat = cl2lat, cl2long = cl2long, cl3lat = cl3lat, cl3long = cl3long, cl4lat = cl4lat, cl4long = cl4long, rsubnames = rsubnames, rsubnames2 = rsubnames2, rsubnames3 = rsubnames3, rsubnames4 = rsubnames4, numroutes1 = numroutes1, numroutes2 = numroutes2, numroutes3 = numroutes3, numroutes4 = numroutes4, rsubAurl = rsubAurl, rsubBurl = rsubBurl, rsubCurl = rsubCurl, rsubDurl = rsubDurl, diffrate1 = diffrate1, x10 = x10, y10 = y10)
