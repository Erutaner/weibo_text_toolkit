"""
Author: Erutaner
Date: 2023.08.11
"""

from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator

class CosineKMeans(KMeans):
    '''
    使用余弦相似度为聚类判据进行向量聚类，适合观点识别任务
    原理为先将传入的矩阵中的每一个行向量都转化为单位向量，再使用普通的k-means算法
    使用方法基本与sklearn的kmeans相同，但可以提供按余弦相似度聚类后的原始向量的聚类中心，通过.original_centers调用
    如果直接.n_clusters，得到的是对单位化后的向量使用k-means进行聚类的聚类中心
    '''
    def __init__(self, n_clusters=8, *, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, verbose=0,
                 random_state=None, copy_x=True,
                 algorithm='auto'):
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         n_init=n_init,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose,
                         random_state=random_state,
                         copy_x=copy_x,
                         algorithm=algorithm)

    def _normalize_data(self, X):
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def _compute_original_cluster_centers(self, X):
        self.original_cluster_centers_ = np.array([X[self.labels_ == i].mean(axis=0)
                                                  for i in range(self.n_clusters)])

    def fit(self, X, y=None, sample_weight=None):
        X_normalized = self._normalize_data(X)
        super().fit(X_normalized, y, sample_weight)
        self._compute_original_cluster_centers(X)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        X_normalized = self._normalize_data(X)
        labels = super().fit_predict(X_normalized, y, sample_weight)
        self._compute_original_cluster_centers(X)
        return labels

    # 向外部提供原始的中心数据
    @property
    def original_centers(self):
        return self.original_cluster_centers_


def kneedle_cosine_kmeans(embedding_list,start = 1, end = 11):
    '''
    基于余弦相似度自动对句向量聚类的函数，指定聚类数的尝试范围，自动确定该范围内最佳聚类数并进行聚类
    :param embedding_list: 评论列表进行句嵌入后得到的二维矩阵
    :param start: 搜参范围，默认为1
    :param end:
    :return: 返回一个CosineKMeans对象，表现与sklearn的kmeans对象基本一致
    '''
    x = range(start,end+1)
    y = []
    for k in x:
        kmeans = CosineKMeans(n_clusters = k)
        kmeans.fit(embedding_list)
        y.append(kmeans.inertia_)
    kn = KneeLocator(x,y,curve = "convex",direction = "decreasing",interp_method = "polynomial")
    best_k = kn.knee if kn.knee is not None else 4

    # 使用k-means进行聚类
    kmeans = CosineKMeans(n_clusters=best_k)
    kmeans.fit(embedding_list)
    return kmeans