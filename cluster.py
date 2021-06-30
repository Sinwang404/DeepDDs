from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
import numpy as np

AZD6244_weight = pd.read_csv('data/result/AZD6244_weight.csv', header=None)
AZD6244_weight = np.array(AZD6244_weight.T)
print('AZD6244_weight.shape', AZD6244_weight.shape)

AZD2014_weight = pd.read_csv('data/result/AZD2014_weight.csv', header=None)
AZD2014_weight = np.array(AZD2014_weight.T)
print('AZD2014_weight.shape', AZD2014_weight.shape)

#AffinityPropagation
af = AffinityPropagation().fit(AZD6244_weight)
result = af.labels_
print(result)
