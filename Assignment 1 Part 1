import csv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        return np.array([list(map(float, row)) for row in reader])

#Calculates how close a test point is to each training point
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((np.array(x1) - np.array(x2))** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):

        distance_array = [euclidean_distance(x, x_train) for x_train in self.X_train]
        knn_indices = np.argsort(distance_array)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in knn_indices]

        most_appeared = Counter(k_nearest_labels).most_common(1)
        return most_appeared[0][0]
    
def cross_validate(X, y, k, folds=10):
    fold_size = len(X) // folds
    accuracy_array = []

    for i in range(folds):
        #Validation Set
        X_val = X[i * fold_size : (i+1) * fold_size]
        y_val = y[i * fold_size : (i+1) * fold_size]
        #Training Set
        X_train = np.concatenate([X[:i * fold_size], X[(i+1) * fold_size:]]) # Uses all data NOT in validation set
        y_train = np.concatenate([y[:i * fold_size], y[(i+1) * fold_size:]])

        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        accuracy_array.append(accuracy)

    return np.mean(accuracy_array)
    
train_inputs = load_csv('train_inputs.csv')
train_labels = load_csv('train_labels.csv').flatten()

k_values = range(1, 31)
average_accuracies = [cross_validate(train_inputs, train_labels, k=k, folds=10) for k in k_values]
best_k = k_values[np.argmax(average_accuracies)]
best_accuracy = max(average_accuracies)

print(f"Best number of neighbors: {best_k}")
print(f"Cross-validation accuracy: {best_accuracy:.2f}")
#test_inputs = load_csv('test_inputs.csv')
#test_labels = load_csv('test_labels.csv').flatten()

knn = KNN(k=3)
knn.fit(train_inputs, train_labels)

plt.plot(k_values, average_accuracies, marker = 'o')
plt.title('Average Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors')
plt.ylabel('Average Accuracy')
plt.xticks(k_values)
plt.grid()
plt.show()
plt.figure(figsize=(12,6))
