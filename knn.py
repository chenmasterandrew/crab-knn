# a simple implemention of the KNN algorithm
# usage:
# 0.) set up environment (see README.md)
# 1.) import the KNN class
# 2.) make a KNN object
# 3.) train the KNN object
# 4.) predict with or test the KNN object
# see crabtest.py or unittest.py for an example

import numpy as np
from collections import Counter

class KNN:
    # static utility method for getting the euclidean distance between two hyperpoints
    @staticmethod
    def get_euclidean_distance(p1, p2):
        return np.sqrt(np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))

    def __init__(self, k):
        self.k = k

    # "trains" the KNN class by storing training data and labels
    # params:
    # [[]] training_data: a list of hyperpoints (lists)
    # [] training_labels: a list of labels for hyperpoints corresponding to the same indices
    def train(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    # makes a prediction of prediction labels based on input prediction data and stored training data and labels
    # params:
    # [[]] prediction_data: a list of hyperpoints (lists)
    def predict(self, prediction_data):
        # construct a List of prediction labels by making a prediction for each hyperpoint or row in the prediction data
        prediction_labels = [self._predict(p_d) for p_d in prediction_data]
        return prediction_labels

    # make a prediction of the label of one hyperpoint p_d
    def _predict(self, p_d):
        # calculate the distances between p_d and every point in the training data set
        distances = [self.get_euclidean_distance(p_d, t_d) for t_d in self.training_data]

        # gets the labels of the k_nearest_labels based on distances
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.training_labels[i] for i in k_nearest_indices]

        # returns the predicted label of p_d based on the most common label among k_nearest_labels
        p_l = Counter(k_nearest_labels).most_common(1)[0][0]
        return p_l
    
    # a function which tests the accuracy of the KNN using labeled testing data
    # returns the percentage of correct predictions
    # params:
    # [[]] test_data: a list of hyperpoints (lists)
    # [] test_labels: a list of labels for hyperpoints corresponding to the same indices
    def test(self, test_data, test_labels):
        prediction_labels = self.predict(test_data)
        num_correct = 0
        for i in range(len(test_labels)):
            if prediction_labels[i] == test_labels[i]:
                num_correct += 1
        return num_correct/len(test_labels)


