from collections import defaultdict

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

class EvaluationTool():

    def __init__(self, file_path, delimiter, legit=None, agg_key=None):
        self.true = None
        self.pred = None
        self.stats = None
        self.labels = None
        self.keys = None

        self.legit = legit
        self.agg_key = agg_key
        self.__read_data(file_path, delimiter)
        self.__compute_stats()

    def __read_data(self, file_path, delimiter):
        """
        Reads true/pred data from a file and saves it to dataframes.
        :param file_path: Path to the file
        :param delimiter: Symbol or a string by which the data is delimited.
        """
        columns = ['true', 'pred', 'key']
        df = pd.read_csv(file_path, delimiter, header=None, names=columns)
        self.true = df['true']
        self.pred = df['pred']
        if not df['key'].isnull().any():
            self.keys = df['key']

    def __compute_stats(self):
        """
        Computes TPs, FPs, TNs and FNs of the given data.
        """
        self.labels = sorted(set(self.true) | set(self.pred))

        matrix = confusion_matrix(self.true, self.pred, labels=self.labels)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)

        stats = defaultdict(dict)

        for i, label in zip(range(len(self.labels)), self.labels):
            stats[label]['FP'] = FP[i]
            stats[label]['FN'] = FN[i]
            stats[label]['TP'] = TP[i]
            stats[label]['TN'] = TN[i]

        self.stats = stats

    def compute_precision(self, class_label):
        """
        Computes precision for the given class label.
        :param class_label: Class label of the row.
        :return: Computed precision of the classifier for the given class.
        """
        TP = self.stats[class_label]['TP']
        FP = self.stats[class_label]['FP']

        if TP + FP == 0:
            return np.nan
        return TP / (TP + FP)

    def compute_recall(self, class_label):
        """
        Computes recall for the given class label.
        :param class_label: Class label of the row.
        :return: Computed recall of the classifier for the given class.
        """
        TP = self.stats[class_label]['TP']
        FN = self.stats[class_label]['FN']

        if TP + FN == 0:
            return np.nan
        return TP / (TP + FN)

    def get_avg_precision(self, legit=True, nan=True):
        """
        Counts the average precision.
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average precision.
        """
        labels = list(self.labels)
        if not legit:
            labels.remove(self.legit)

        precs = np.fromiter(map(
            self.compute_precision, labels), dtype=float, count=len(labels)
        )

        if not nan:
            return np.nanmean(precs)
        else:
            return np.nansum(precs) / len(labels)

    def get_avg_recall(self, legit=True, nan=True):
        """
        Counts the average recall.
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average recall.
        """
        labels = list(self.labels)
        if not legit:
            labels.remove(self.legit)

        recs = np.fromiter(map(
            self.compute_recall, labels), dtype=float, count=len(labels)
        )

        if not nan:
            return np.nanmean(recs)
        else:
            return np.nansum(recs) / len(labels)
