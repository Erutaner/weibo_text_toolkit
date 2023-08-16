"""
Author: Erutaner
Date: 2023.08.13
"""
import numpy as np

def compute_discrepancy(embedding_list, cosine_kmeans):
    '''
    簇内间加权余弦极化指数计算
    :param embedding_list: 句嵌入后得到的embedding_list
    :param cosine_kmeans: cosine-kmeans对象，见ml_algorithms.py
    :return: 返回离散程度的度量值，在0和1之间
    '''
    embedding_list = embedding_list / np.linalg.norm(embedding_list, axis=1, keepdims=True)
    G = cosine_kmeans.n_clusters
    labels = cosine_kmeans.labels_
    cluster_centers = cosine_kmeans.cluster_centers_
    global_center = np.mean(embedding_list, axis=0)


    numerator = 0
    for i in range(G):
        points_in_cluster = embedding_list[labels == i]
        cluster_center = cluster_centers[i]

        distances = np.linalg.norm(points_in_cluster - cluster_center, axis=1)
        numerator += np.sum(distances)


    denominator = 0
    for s in range(G):
        num_points_in_cluster = np.sum(labels == s)
        cluster_center = cluster_centers[s]

        distance = np.linalg.norm(cluster_center - global_center)
        denominator += num_points_in_cluster * distance


    discrepancy = np.exp(-numerator / denominator)

    return discrepancy

def compute_entropy(embedding_list, cosine_kmeans):
    '''
        归一化熵测度计算
        :param embedding_list: 句嵌入后得到的embedding_list
        :param cosine_kmeans: cosine-kmeans对象，见ml_algorithms.py
        :return: 返回离散程度的度量值，在0和1之间
        '''
    labels = cosine_kmeans.labels_
    G = cosine_kmeans.n_clusters


    _, counts = np.unique(labels, return_counts=True)


    p_values = counts / len(embedding_list)


    entropy = -np.sum(p_values * np.log(p_values)) / np.log(G)

    return entropy