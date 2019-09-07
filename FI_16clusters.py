#!/usr/bin/python3

from random import shuffle
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
from sklearn import datasets
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RFC
import os
import glob
import math
import pandas as pd
import numpy as np
import csv
import scipy
import sys
import time
from zipfile import *
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


f = open('data_neg_entropy.csv', 'r')
data_neg_entropy = pd.read_csv(f, sep='\t', header=None)
f.close()

f = open('data_pos_entropy.csv', 'r')
data_pos_entropy = pd.read_csv(f, sep='\t', header=None)
f.close()

path_pos = '20170809-pos'
path_neg = 'DATASET3'


files_neg = glob.glob(os.path.join('20170809-0830', path_neg, '*.gz'))
files_pos = glob.glob(os.path.join('20170809-0830', path_pos, '*.gz'))


data_pos = pd.concat(
                pd.read_csv(f, sep='\t', header=None) for f in files_pos
            )


data_neg = pd.concat(
                pd.read_csv(f, sep='\t', header=None) for f in files_neg
            )


orig_stdout = sys.stdout
o = open('trainingtime.txt', 'w')
sys.stdout = o

data = pd.concat([data_neg, data_pos])
data.index = range(len(data.index))

vectors_FI = data[data.columns[4:]]
labels_FI = data[data.columns[3]]


rfc1 = RFC(
    n_estimators=50,
    max_features=10,
    max_depth=30,
    min_samples_split=3,
    criterion="entropy",
    n_jobs=-1
)


start = time.time()

rfc1.fit(vectors_FI, labels_FI)

end = time.time()
print ('PROCESSING TIME = ', end - start)


FI = rfc1.feature_importances_

'''MODIFICATION OF TESTING NEGATIVE DATASET'''


vectors_neg_entropy = data_neg_entropy[data_neg_entropy.columns[4:]]


Iteration = list(range(np.shape(vectors_neg_entropy)[1]))

for j in Iteration:
    if FI[j]==0:
        vectors_neg_entropy[:, j] *= 0
        continue
    vectors_neg_entropy[:, j] *= 1 / math.sqrt(FI[j])


'''NEGATIVE DATA CLUSTERING'''

number_of_clusters = 16

kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(vectors_neg_entropy)
#maximum = max(data_pos[3])
y = kmeans.labels_
y = y.tolist()
z = pd.DataFrame(y)

for j in range(number_of_clusters):
    z.replace(
                to_replace=j,
                value=500+j,
                inplace=True
            )

z.index = data_neg_entropy.index
data_neg_entropy[3] = z

data = pd.concat([data_neg_entropy, data_pos_entropy])


"""
END OF CLUSTERING
"""

vectors = data[data.columns[4:]]
labels = data[data.columns[3]]


clf = DecisionTreeClassifier(random_state = None, max_depth = 2, criterion = 'entropy')

clf.fit(vectors,labels)

dot_data = tree.export_graphviz(clf, out_file=None)

graph = graphviz.Source(dot_data)

graph.render("FI_16clusters")


sys.stdout = orig_stdout
o.close()

