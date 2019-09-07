#!/usr/bin/python3
import pandas as pd
import numpy as np
import STAT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler


class CrossPCATool2:

    def __init__(self, data=None, high_filter=2, subsample=False, subsample_constant=13000):
        if data is not None:
            data = pd.DataFrame(data.values)  # reset all row and column indexes

        self.PCAList = None
        self.CovList = None
        self.mi = None
        self.std = None
        self.highFilter = high_filter
        self.data = data
        self.ClfList = [] # list of classifiers
        self.classes = None

        if subsample == True:

            self.__get_CovList_and_scaler_withSubsample(subsample_constant)
        else:
            self.__get_CovList_and_scaler_noSubsample()

        self.classes.remove(0)
        self.CovList.pop(0)


        self.__get_PCAList(high_filter)

        self.__get_ClfList()

    def __get_CovList_and_scaler_noSubsample(self):

        self.CovList, self.mi, self.std, self.classes = STAT.pca4all_cov_matrices_and_scaler_noSubsample(self.data)



    def __get_CovList_and_scaler_withSubsample(self, subsample_constant):

        self.CovList, self.mi, self.std, self.classes = STAT.pca4all_cov_matrices_and_scaler_withSubsample(self.data, subsample_constant)

    def __get_PCAList(self, high_filter):

        PCAList = []
        for cov in self.CovList:

            cov_matrix = pd.DataFrame(cov)
            pca_matrix = STAT.pca_reduction_HighFilter(cov_matrix, high_filter)
            PCAList.append(pca_matrix)

        self.PCAList = PCAList



    def scale_data(self, data, label):

        number_of_samples = data.shape[0]  # first column corresponds to labels

        data = np.array(data, dtype=float)

        index = self.classes.index(label) + 1 # first element in mi, std corresponds to 0 class

        for j in range(number_of_samples):
            data[j, 1:] -= self.mi[index, :]
            std_dev = self.std[index, :]
            std_dev = np.where(std_dev == 0, 1, std_dev)
            data[j, 1:] = data[j, 1:] / std_dev

        return pd.DataFrame(data)

    def make_binary_labels(self, data, label):

        '''we keep parameter::label an other labels are set to 0'''
        pd_temp = data.loc[:, 0]
        labels = np.array(self.classes).tolist()  # classes have unique values, so we just need to remove this one specific label
        labels.remove(label)

        pd_temp = pd_temp.replace(to_replace=labels, value=0)

        data.loc[:, 0] = pd_temp

        return data



    def __get_ClfList(self):


        self.ClfList = [RFC(
            n_estimators=50,
            max_features=10,
            max_depth=30,
            min_samples_split=3,
            criterion="entropy",
            n_jobs=-1
            ) for j in range(len(self.classes))]

        for label in self.classes:

            vecs = np.array(self.data, dtype=float) # this is just to avoid using reference in = operator

            vecs = pd.DataFrame(vecs) # redefine back to dataframe

            vecs = self.make_binary_labels(vecs,label)



            data = self.scale_data(vecs,label)

            index = self.classes.index(label)

            data = np.array(data, dtype=float)

            data_temp = np.dot(data[:,1:], self.PCAList[index])

            labels = data[:,0]

            vectors = data_temp

            self.ClfList[index].fit(vectors,labels)




    def test_predictions(self,test_data):

        predictions = []

        for label in self.classes:



            data = self.scale_data(test_data, label)

            index = self.classes.index(label)

            data = np.array(data, dtype=float)

            data_temp = np.dot(data[:, 1:], self.PCAList[index])

            clf = self.ClfList[index]

            test_predictions = clf.predict_proba(data_temp)

            predictions.append(test_predictions)

        final_predictions = [[0,0] for j in range(test_data.shape[0])]

        for prediction, index in zip(predictions,range(len(self.classes))):

            prediction = prediction.tolist()

            L = [j[1] for j in prediction]

            for i,j,k in zip(L,final_predictions,range(len(L))):

                if i > j[0]:
                    final_predictions[k][0] = i
                    final_predictions[k][1] = index

        final_pred = [self.classes[j[1]] for j in final_predictions]

        return final_pred












