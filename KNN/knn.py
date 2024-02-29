import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k_values):
        self.k_values = k_values
       
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for k in self.k_values:
            predictions.append([self._predict(x, k) for x in X])
        return predictions
    
    def _predict(self, x, k):
        #Compute distances between x and all other data in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        #Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:k]
        
        #Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        accuracies = []
        for k, preds in zip(self.k_values, predictions):
            accuracy = np.mean(preds == y.flatten()) * 100
            accuracies.append(accuracy)
        return accuracies
    
#Load Dataset
data = pd.read_csv("KNN/iris.csv", header=None)

#Shuffle dataset
data_shuffled = shuffle(data)

k_values = list(range(1, 52, 2))

train_accuracies = []
test_accuracies = []

accurancies_train = []
accurancies_test = []

stdevs_train = []
stdevs_test = []

for _ in range(20):
    #Split the dataset into training and testing datasets
    train_data, test_data = train_test_split(data_shuffled, test_size=0.2, random_state=42)

    #Features: all columns except last one (training data)
    X_train = train_data.iloc[:, :-1].values
    #Target: only the last column
    y_train = train_data.iloc[:, -1].values
    #Features:  all columns except last one (testing data)
    X_test = test_data.iloc[:, :-1].values
    #Only the last column
    y_test = test_data.iloc[:, -1].values

    #Normalize the data
    #scalar = MinMaxScaler()
    #X_train = scalar.fit_transform(X_train)
    #X_test = scalar.transform(X_test)

    knn = KNN(k_values)
    knn.fit(X_train, y_train)
    train_accuracies.append(knn.accuracy(X_train, y_train))
    test_accuracies.append(knn.accuracy(X_test, y_test))

#Convert lists to numpy arrays
train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)

#Calculate the average accuracies for each k value
train_accuracies_avg = np.mean(train_accuracies, axis=0)
test_accuracies_avg = np.mean(test_accuracies, axis=0)

train_accuracies_std = np.std(train_accuracies, axis=0)
test_accuracies_std = np.std(test_accuracies, axis=0)

stdevs_train.append(train_accuracies_std)
stdevs_test.append(test_accuracies_std)

train_stdevs_arr = np.array(stdevs_train)
test_stdevs_arr = np.array(stdevs_test)
    
#Plot training accuracies
plt.figure(figsize=(10, 5))
plt.plot(k_values, train_accuracies_avg, marker='o', linestyle='-',color='b')
plt.title('Training Accuracies')
plt.xlabel('k value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.yticks(train_accuracies_avg)
plt.show()

#Plot testing accuracies
plt.figure(figsize=(10, 5))
plt.plot(k_values, test_accuracies_avg, marker='o', linestyle='-', color='r')
plt.title('Testing Accuracies')
plt.xlabel('k value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.yticks(test_accuracies_avg)
plt.show()

#Plot training accuracies with error bars
plt.figure(figsize=(10, 5))
plt.errorbar(k_values, train_accuracies_avg, xerr=train_stdevs_arr, fmt='o-', color='b', ecolor='green', capsize=15)
plt.title('Training Accuracies with error bar')
plt.xlabel('k value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.show()

#Plot testing accuracies with error bars
plt.figure(figsize=(10, 5))
plt.errorbar(k_values, test_accuracies_avg, xerr=test_stdevs_arr, fmt='o-', color='b', ecolor='green', capsize=15)
plt.title('Testing Accuracies with error bar')
plt.xlabel('k value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.show()






    












