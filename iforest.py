# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import math
import numpy as np
import random
import pandas as pd
from sklearn.metrics import confusion_matrix

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = math.ceil(np.log2(self.sample_size))

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.trees = []
        self.feature_sample_size = math.ceil(0.7*X.shape[1])
        for i in range(self.n_trees):
            X_sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False),:]
            tree = IsolationTree(self.height_limit)
            tree.fit(X_sample, depth=0, improved=improved)
            self.trees.append(tree)
        self.e = np.zeros([X.shape[0], self.n_trees])
        return self


    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        # E = []
        # for x in X:
        #     e_i = []
        #     for T in self.trees:
        #         p = T.root
        #         e = 0
        #         while isinstance(p, inNode):
        #             if x[p.splitAtt] < p.splitValue:
        #                 e = e + 1
        #                 p = p.left
        #             else:
        #                 e = e + 1
        #                 p = p.right
        #         if p is not None:
        #             # print(p.val)
        #             e = e + self.C(p.val)
        #         e_i.append(e)
        #     E.append(np.mean(e_i))
        # E = np.array(E).reshape(X.shape[0],1)
        # return E

        # better and faster method to calculate the path length
        for i, T in enumerate(self.trees):
            p = T.root
            e_i = self.walk_update_e(X, p, self.e[:, i], (np.ones(X.shape[0]) == 1))
            self.e[:, i] = e_i
        E = self.e.mean(axis=1).reshape(X.shape[0], 1)
        # print(E)
        return E


    def C(self, x):
        if x > 2:
            c = 2 * (np.log(x - 1) + 0.5772156649) - 2 * (x - 1) / self.sample_size
        elif x == 2:
            c = 1
        else:
            c = 0
        return c


    def walk_update_e(self, X, p, e, base_condition):
        if p is None: return e
        else:
            if isinstance(p, inNode):
                if p.left is not None:
                    flag = np.logical_and(base_condition, (X[:, p.splitAtt] < p.splitValue))
                    e[flag] = e[flag] + 1
                    e =  self.walk_update_e(X, p.left, e, flag)

                # base condition count for previous condition
                if p.right is not None:
                    flag = np.logical_and(base_condition, (X[:, p.splitAtt] >= p.splitValue))
                    e[flag] = e[flag] + 1
                    e = self.walk_update_e(X, p.right, e, flag)

            else: e[base_condition] = e[base_condition] + self.C(p.val)
            return e

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        E = self.path_length(X)
        c = self.C(self.sample_size)

        s = 2 ** (-E/c)
        return s


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores >= threshold)*1

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        if isinstance(X, pd.DataFrame):
            X = X.values
        scores = self.anomaly_score(X)
        pred = self.predict_from_anomaly_scores(scores, threshold)
        return pred


class IsolationTree:
    def __init__(self, height_limit):
        self.n_nodes = 0
        self.height_limit = height_limit
        self.root = None

    def fit(self, X:np.ndarray, improved=False, depth=0):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If improve=True, choose split value according to its distribution. Randomly choose the value
        from tails instead of choosing from min to max
        """
        if ((depth >= self.height_limit) | (X.shape[0]<=1)):
            self.n_nodes += 1
            return exNode(val=X.shape[0])
        else:
            splitAtt = random.randint(0,X.shape[1]-1)
            if improved:
                # narrow the range of random pick
                X_mean = np.mean(X[:, splitAtt])
                X_median = np.median(X[:, splitAtt])
                if X_median <= X_mean:
                    low_limit = np.max([np.quantile(X[:,splitAtt], 0.8, axis=0), X_mean])
                    splitValue = random.uniform(low_limit,X[:, splitAtt].max())
                else:
                    up_limit = np.min([np.quantile(X[:,splitAtt], 0.2, axis=0), X_mean])
                    splitValue = random.uniform(X[:, splitAtt].min(), up_limit)
            else:
                splitValue = random.uniform(X[:, splitAtt].min(),X[:, splitAtt].max())
            right_X = X[X[:, splitAtt] >= splitValue,:]
            left_X = X[X[:, splitAtt] < splitValue,:]
            self.root = inNode(self.fit(left_X, depth=depth + 1), self.fit(right_X, depth=depth + 1), splitAtt, splitValue)
            self.n_nodes += 1
        return self.root


class exNode:
    def __init__(self, val):
        """
        the external node class
        :param val: np.ndarray
        """
        self.val=val

class inNode:
    def __init__(self, left, right, splitAtt, splitValue):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitValue = splitValue


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1
    while threshold:
        y_pred = (scores >= threshold)*1
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return threshold, FPR
        threshold = threshold - 0.01



