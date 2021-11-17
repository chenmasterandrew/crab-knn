# simple unit test of KNN class on whether a set of numbers is negative
# conclusion: correctly labels numbers except for 0, which the knn was not trained on

from knn import KNN

knn = KNN(3)

# training data is a set of points of dimension 1
train_data = [[-1],[-2],[-3],[-4],[1],[2],[3],[4]]
# training labels are whether the point is negative
train_labels = [1,1,1,1,0,0,0,0]
knn.train(train_data, train_labels)

# predict whether a set of numbers is negative
test_data = [[-5],[-4],[-3],[-2],[-1],[0],[1],[2],[3],[4],[5]]
test_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
predict_labels = knn.predict(test_data)

# test how accurate the knn is
accuracy = knn.test(test_data, test_labels)

print("test data:         " + str(test_data))
print("prediction labels: " + str(predict_labels))
print("test labels:       " + str(test_labels))
print("% correct:         " + str(accuracy))

