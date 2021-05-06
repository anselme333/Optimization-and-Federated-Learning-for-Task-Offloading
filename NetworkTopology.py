# Joint Federated Learning and Optimization for Intelligent
# Task Offloading in 5G and Beyond
# Author: Anselme Ndikumana
# Python 3.6.4
########################################################################################################################
# Loading needed libraries

import pandas
import pandas as pd
from pandas import DataFrame
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
#######################################################
# Loading data set of 2020-05-25 from kaggle.com for 5g-user-prediction
# Source of data https://www.kaggle.com/liukunxin/dataset/activity

#######################################################
# Data cleansing and make a topology, run user_prediction_data_visualization.py
#######################################################
User_traffic = pandas.read_csv('dataset/dataset_updated.csv')
User_traffic = User_traffic.reset_index(drop=True)
User_traffic.drop(User_traffic.filter(regex="Unname"),axis=1, inplace=True)
User_traffic = User_traffic.drop_duplicates(subset='user_id', keep="first")
#######################################################


def network_traffic(User_traffic_data):
    # We consider  24 hours for making synthetic time series
    date = np.array('2020-09-04 12:00',  dtype=np.datetime64)
    # 60 minutes * 24 hours
    date = date + np.arange(1440)
    n = User_traffic_data.shape[0]
    date = np.resize(date, n)
    #  We make a synthetic time series for 24 hours
    User_traffic_data['Time'] = date
    print(User_traffic_data.columns)
    cols = User_traffic_data.columns.tolist()
    # Make Time as the first column
    cols = cols[-1:] + cols[:-1]
    User_traffic_data_updated = User_traffic_data[cols]

    User_traffic_data_updated.replace([np.inf, -np.inf], np.nan)
    User_traffic_data_updated.dropna(inplace=True)
    print(User_traffic_data_updated.head())
    df = DataFrame(User_traffic_data_updated, columns=['video_app_flux'])
    print(df.head())
    # We group 319 areas into 10 clusters and we consider centroid as 10 edge clouds which are connected
    # to one regional cloud
    kmeans = KMeans(n_clusters=10).fit(df)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    Cluster_label = []
    for i in range(len(df)):
        k = labels[i]
        k = int(k)
        Cluster_label.append(labels[i])
    # Assign O-RU to edge cloud(O-DU)
    User_traffic_data_updated['edge_cloud'] = Cluster_label
    return kmeans,df, centroids, User_traffic_data_updated


kmeans, df, centroids, User_traffic_data_updated = network_traffic(User_traffic)
User_traffic_data_updated = \
    User_traffic_data_updated.loc[(User_traffic_data_updated[['Time', 'video_app_flux']] != 0).all(axis=1)]

User_traffic = User_traffic_data_updated[User_traffic_data_updated['edge_cloud'] == 4]
User_traffic = User_traffic[['Time','video_app_flux']]
User_traffic[:100].plot(x='Time', y='video_app_flux',linewidth=2, legend=None)
plt.grid(which='both')
plt.ylabel('Video Traffic (Mbps)')
plt.xlabel('Time')
plt.show()

# We assume that nnet_months is the number od days users  spend on networks in all recorded month traffic
daily_traffic_column = User_traffic_data_updated['video_app_flux'] / User_traffic_data_updated['innet_months']
User_traffic_data_updated['video_app_flux_day'] = daily_traffic_column

User_traffic_data_updated.to_csv('dataset/User_traffic_data.csv')


def network_topology(user_traffic_data):
    # We have 21118 users
    total_users = user_traffic_data['user_id'].nunique()
    # We have 319 areas, we assume that each area is served by one O-RU
    total_O_RAN = user_traffic_data['area_id'].nunique()
    # We have edge cloud
    total_edge_cloud = user_traffic_data['edge_cloud'].nunique()
    total_number_regional_cloud = 1
    return total_users, total_O_RAN, total_edge_cloud, total_number_regional_cloud


total_users, total_O_RAN, total_edge_cloud, total_number_regional_cloud = network_topology(User_traffic_data_updated)
print("total_users", total_users)
print("total_O_RAN", total_O_RAN)
print("total_edge_cloud", total_edge_cloud)
print("total_number_regional_cloud", total_number_regional_cloud)