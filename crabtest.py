# application of the knn implementation to crab biometric data to predict crab sex
# conclusion: correctly labels 99% of crabs in test data set when k=4

from knn import KNN
from csv import reader
import pandas as pd

# value of k for knn
K = 4

# import crabdata from crabdata.csv
crabdata = pd.read_csv(r'crabdata.csv')

# select only those columns which are useful for KNN
# see dataset at https://www.kaggle.com/inputblackboxoutput/crab-body-metrics for explanation of column contents
crabdata_cleaned = pd.DataFrame(crabdata, columns=['sex', 'FL', 'RW', 'CL', 'CW', 'BD'])

# partition crabdata into training and test sets
train_set = crabdata_cleaned.iloc[0::2] # training set will contain those rows with even row numbers
test_set = crabdata_cleaned.iloc[1::2] # testing set will contain those rows with odd row numbers

# partition training and test sets into data and labels
train_labels = train_set['sex'].tolist()
train_data = pd.DataFrame(train_set, columns=['FL', 'RW', 'CL', 'CW', 'BD']).values.tolist()
test_labels = test_set['sex'].tolist()
test_data = pd.DataFrame(train_set, columns=['FL', 'RW', 'CL', 'CW', 'BD']).values.tolist()

# create and make predictions using KNN
knn = KNN(K)
knn.train(train_data, train_labels)
predict_labels = knn.predict(test_data)
accuracy = knn.test(test_data, test_labels)
outcomes = pd.DataFrame([predict_labels, test_labels], index=['predicted labels', 'actual labels']).T

# output
print('TRAINING DATASET:')
print(train_set)
print()
print('TESTING DATASET:')
print(test_set)
print()
print('TESTING RESULTS for k='+ str(K) +':')
print(outcomes)
print()
print("% CORRECT: " + str(accuracy))

