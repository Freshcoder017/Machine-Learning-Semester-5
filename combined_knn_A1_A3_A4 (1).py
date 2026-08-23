# ======================================================================
# COMBINED KNN IMPLEMENTATIONS - A1, A3, A4
# ======================================================================
# A1: AI-generated implementation
# A3: scikit-learn implementation
# A4: manually implemented KNN
# The implementations are separated so their approaches can be compared.
# Only overlapping function names in A1 and A4 have been renamed.
# No algorithmic logic has been changed.
# ======================================================================


# ======================================================================
# A1 - AI MODULE GENERATED KNN CODE
# ======================================================================
# This section preserves the A1 code structure and logic.
# Function names in this section use the "_AI" suffix where required
# to distinguish them from overlapping manual implementations.
# ======================================================================

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# =========================
# Load Dataset
# =========================

df = pd.read_csv("eeg_features.csv")

X = df.drop(columns=["label", "subject"]).copy()
y = df["label"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=7,
    stratify=y
)

# Convert DataFrames/Series to NumPy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# =========================
# Distance Functions
# =========================

def euclidean_AI(v1, v2):
    distance = 0

    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2

    return math.sqrt(distance)


def manhattan_AI(v1, v2):
    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i])

    return distance


def minkowski_AI(v1, v2, p):

    if p == 1:
        return manhattan_AI(v1, v2)

    elif p == 2:
        return euclidean_AI(v1, v2)

    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p

    return distance ** (1 / p)


# =========================
# Calculate Distances
# =========================

def calculateDistances_AI(testVector, X_train, y_train, metric, p=2):

    distances = []

    for i in range(len(X_train)):

        trainVector = X_train[i]

        if metric == "euclidean":
            distance = euclidean_AI(
                testVector,
                trainVector
            )

        elif metric == "manhattan":
            distance = manhattan_AI(
                testVector,
                trainVector
            )

        elif metric == "minkowski":
            distance = minkowski_AI(
                testVector,
                trainVector,
                p
            )

        else:
            raise ValueError(
                "Invalid metric. Choose euclidean, manhattan, or minkowski."
            )

        distances.append((distance, y_train[i]))

    return distances


# =========================
# Sorting Algorithms
# =========================

def insertionSort_AI(data):

    for i in range(1, len(data)):

        key = data[i]
        j = i - 1

        while j >= 0 and data[j][0] > key[0]:

            data[j + 1] = data[j]
            j -= 1

        data[j + 1] = key

    return data


def bubbleSort_AI(data):

    n = len(data)

    for i in range(n):

        swapped = False

        for j in range(0, n - i - 1):

            if data[j][0] > data[j + 1][0]:

                data[j], data[j + 1] = \
                    data[j + 1], data[j]

                swapped = True

        if not swapped:
            break

    return data


def selectionSort_AI(data):

    n = len(data)

    for i in range(n):

        minIndex = i

        for j in range(i + 1, n):

            if data[j][0] < data[minIndex][0]:
                minIndex = j

        data[i], data[minIndex] = \
            data[minIndex], data[i]

    return data


def sortDistances_AI(data, sortMethod):

    if sortMethod == "insertion":
        return insertionSort_AI(data)

    elif sortMethod == "bubble":
        return bubbleSort_AI(data)

    elif sortMethod == "selection":
        return selectionSort_AI(data)

    else:
        raise ValueError(
            "Invalid sorting method. "
            "Choose insertion, bubble, or selection."
        )


# =========================
# KNN
# =========================

def knn_AI(testVector, X_train, y_train, k, metric, p, sortMethod):

    # Calculate distances
    distances = calculateDistances_AI(
        testVector,
        X_train,
        y_train,
        metric,
        p
    )

    # Sort distances
    sortedDistances = sortDistances_AI(
        distances,
        sortMethod
    )

    # Check K
    if k < 1 or k > len(sortedDistances):

        raise ValueError(
            "K must be between 1 and "
            "the number of training samples."
        )

    # Select K nearest neighbors
    nearestNeighbors = sortedDistances[:k]

    # Count votes
    votes = {}

    for distance, label in nearestNeighbors:

        if label not in votes:
            votes[label] = 1
        else:
            votes[label] += 1

    # Maximum votes
    maxVotes = max(votes.values())

    # Labels having maximum votes
    tiedLabels = [
        label
        for label, count in votes.items()
        if count == maxVotes
    ]

    # No tie
    if len(tiedLabels) == 1:

        predictedLabel = tiedLabels[0]

    # Tie breaker
    else:

        # Since nearestNeighbors is already sorted,
        # the first tied label is the closest one.
        for distance, label in nearestNeighbors:

            if label in tiedLabels:

                predictedLabel = label
                break

    return predictedLabel


# =========================
# KNN Main Program
# =========================

metric = "euclidean"
p = 2
sortMethod = "insertion"

# Store performance for every K
AI_results = []

# K from 1 to number of test samples
for k in range(1, len(X_test) + 1):

    predictions = []

    # Predict every test sample
    for testVector in X_test:

        predictedLabel = knn_AI(
            testVector,
            X_train,
            y_train,
            k,
            metric,
            p,
            sortMethod
        )

        predictions.append(predictedLabel)

    # =========================
    # Calculate TP, TN, FP, FN
    # =========================

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Assuming:
    # 1 = Schizophrenia
    # 0 = Healthy
    #
    # If your labels are strings, change these accordingly.

    for i in range(len(y_test)):

        actual = y_test[i]
        predicted = predictions[i]

        if actual == 1 and predicted == 1:
            TP += 1

        elif actual == 0 and predicted == 0:
            TN += 1

        elif actual == 0 and predicted == 1:
            FP += 1

        elif actual == 1 and predicted == 0:
            FN += 1

    # =========================
    # Calculate Metrics
    # =========================

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall != 0:
        f1Score = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1Score = 0

    # Store results
    AI_results.append(
        (
            k,
            accuracy,
            precision,
            recall,
            f1Score
        )
    )



# ======================================================================
# A3 - SCIKIT-LEARN KNN CODE
# ======================================================================
# This section contains the scikit-learn implementation as provided.
# It is kept separate from the AI-generated and manual implementations.
# ======================================================================

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
df=pd.read_csv("eeg_features.csv")  
X = df.drop(columns=["label","subject"]).copy()
y = df["label"].copy()
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)  
#random_state=7  same split every time you run program
# stratify=y --> healthy/schizophrenia class proportions are maintained in both sets.

#### A4 ######

A3_results = []

for k in range(1, 10):

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)

    ####### A5 #######
    accuracy = neigh.score(X_test, y_test)

    ####### A6 ########
    prediction = neigh.predict(X_test)

    # =========================
    # Calculate TP, TN, FP, FN
    # =========================

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_test)):

        actual = y_test.iloc[i]
        predicted = prediction[i]

        if actual == 1 and predicted == 1:
            TP += 1

        elif actual == 0 and predicted == 0:
            TN += 1

        elif actual == 0 and predicted == 1:
            FP += 1

        elif actual == 1 and predicted == 0:
            FN += 1

    # =========================
    # Calculate Metrics
    # =========================

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall != 0:
        f1Score = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1Score = 0

    # Store results
    A3_results.append(
        (
            k,
            accuracy,
            precision,
            recall,
            f1Score
        )
    )




# ======================================================================
# A4 - MANUAL KNN CODE
# ======================================================================
# This section preserves the manually implemented KNN code and logic.
# Function names in this section use the "_own" suffix where required
# to distinguish them from overlapping implementations.
# ======================================================================

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
df=pd.read_csv("eeg_features.csv")  
X = df.drop(columns=["label","subject"]).copy()
y = df["label"].copy()
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)  
#random_state=7  same split every time you run program
# stratify=y --> healthy/schizophrenia class proportions are maintained in both sets.
#### for this dataset imputation is not required as it is already cleaned
#### but for lab i am writing imputation function
def missingValues(df):
    return df.isnull().sum()
'''
def outliers(X):
    outliercols = []
    for col in numdf:
        Q1 = numdf[col].quantile(0.25)
        Q3 = numdf[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = numdf[(numdf[col] < lower) | (numdf[col] > upper)]
        if len(outliers) > 0:
            outliercols.append(col)
    return numdf,outliercols

def imputation(): # no mode coz no categorical data
    numdf,outlierColumns = outliers()
    for col in numdf:
        if col in outlierColumns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())

'''
################ imputation done #########################
def euclidean_own(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i]) ** 2
    return math.sqrt(dist)

def manhattan_own(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])
    return dist

def minkowski_dist_own(v1, v2, p):
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i]) ** p
    dist=dist ** (1 / p)
    return dist

# calculating vector for the n-dimensions;
def all_dist_own(testvector, X_train, y_train, metric, p=2):
    distances=[]
    for trainvec, j in zip(X_train,y_train):
        if metric == "euclidean_own":
            dist= euclidean_own(trainvec, testvector)
        elif metric == "manhattan_own":
            dist= manhattan_own(trainvec, testvector)
        elif metric == "minkowski":
            dist= minkowski_dist_own(trainvec, testvector, p )
        distances.append((dist,j))
    return distances
# now got a distance array for all vectors 
########## distance metric done ###############
def bubbleSort_own(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j]>arr[j + 1]:
                temp=arr[j]
                arr[j]=arr[j+1]
                arr[j+1]=temp
    return arr

def selectionSort_own(arr, size):
    for ind in range(size-1):
        min_index=ind
        for j in range(ind+1, size):
            if arr[j]<arr[min_index]:
                min_index=j

        arr[ind], arr[min_index] = arr[min_index], arr[ind]
    return arr

def insertionSort_own(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        # Compare key with each element on the left of it until an element smaller than it is found       
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j =j-1
        arr[j+1] = key
    return arr

def sorting_metric_own(metric,dist_rec):
    if metric=="selectionSort_own":
        return selectionSort_own(dist_rec, len(dist_rec))
    elif metric=="insertionSort_own":
        return insertionSort_own(dist_rec)
    elif metric=="bubbleSort_own":
        return bubbleSort_own(dist_rec)
    else :
        return 0
############# sorting modules ##############
def k_nearest_own(k,arr):
    k_arr=[]
    for i in range(k):
        x=arr[i]
        k_arr.append(x)
    return k_arr 
# k nearest neighbors based on distance

def tie_breaker_own(knn):
    # 1- if same number of votes for a class if k is even
    class0= 0
    class1= 0
    class0_dist= 0
    class1_dist= 0
    for dist, label in knn:
        if label == 0:
            class0+=1
            class0_dist+=dist
        else:
            class1+=1
            class1_dist+=dist
    if class0>class1:
        return 0
    elif class1>class0:
        return 1
    else:
        # choose class whose neighbors have smaller total distance
        if class0_dist < class1_dist:
            return 0
        else:
            return 1

def accuracy_own(preds, y_test):
    return (preds == y_test).mean()


#print(missingValues(df))
numdf=df.select_dtypes(include=np.number)
'''
print(outliers(X))
print(X)
''' 

#step1- print actual y_test val
#step2-calculate distance array along with the class
#step3- sort the array
#step4- from that find the nearest k neighbors


manual_results = []

for k in range(1, 10):

    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(X_test)):

        testvector = X_test.iloc[i].values

        dist_rec = all_dist_own(
            testvector,
            X_train.values,
            y_train.values,
            "euclidean_own",
            p=2
        )

        sorting = sorting_metric_own("selectionSort_own", dist_rec)

        knn = k_nearest_own(k, sorting)

        pred = tie_breaker_own(knn)

        actual = y_test.iloc[i]

        if pred == actual:
            correct += 1

        if actual == 1 and pred == 1:
            TP += 1

        elif actual == 0 and pred == 0:
            TN += 1

        elif actual == 0 and pred == 1:
            FP += 1

        elif actual == 1 and pred == 0:
            FN += 1

    # =========================
    # Calculate Metrics
    # =========================

    total = TP + TN + FP + FN

    accuracy_value = (TP + TN) / total

    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall != 0:
        f1Score = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1Score = 0

    manual_results.append(
        (
            k,
            accuracy_value,
            precision,
            recall,
            f1Score
        )
    )

# ======================================================================
# FINAL COMPARISON TABLES
# ======================================================================

# =========================
# Accuracy Comparison
# =========================

print("\n\nAccuracy Comparison")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'A1 - AI':<20}"
    f"{'A3 - Scikit':<20}"
    f"{'A4 - Manual':<20}"
)

print("-" * 70)

for i in range(len(AI_results)):

    print(
        f"{AI_results[i][0]:<5}"
        f"{AI_results[i][1]:<20.4f}"
        f"{A3_results[i][1]:<20.4f}"
        f"{manual_results[i][1]:<20.4f}"
    )


# =========================
# Precision Comparison
# =========================

print("\n\nPrecision Comparison")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'A1 - AI':<20}"
    f"{'A3 - Scikit':<20}"
    f"{'A4 - Manual':<20}"
)

print("-" * 70)

for i in range(len(AI_results)):

    print(
        f"{AI_results[i][0]:<5}"
        f"{AI_results[i][2]:<20.4f}"
        f"{A3_results[i][2]:<20.4f}"
        f"{manual_results[i][2]:<20.4f}"
    )


# =========================
# Recall Comparison
# =========================

print("\n\nRecall Comparison")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'A1 - AI':<20}"
    f"{'A3 - Scikit':<20}"
    f"{'A4 - Manual':<20}"
)

print("-" * 70)

for i in range(len(AI_results)):

    print(
        f"{AI_results[i][0]:<5}"
        f"{AI_results[i][3]:<20.4f}"
        f"{A3_results[i][3]:<20.4f}"
        f"{manual_results[i][3]:<20.4f}"
    )


# =========================
# F1 Score Comparison
# =========================

print("\n\nF1 Score Comparison")
print("-" * 70)

print(
    f"{'K':<5}"
    f"{'A1 - AI':<20}"
    f"{'A3 - Scikit':<20}"
    f"{'A4 - Manual':<20}"
)

print("-" * 70)

for i in range(len(AI_results)):

    print(
        f"{AI_results[i][0]:<5}"
        f"{AI_results[i][4]:<20.4f}"
        f"{A3_results[i][4]:<20.4f}"
        f"{manual_results[i][4]:<20.4f}"
    )

# ======================================================================
# PERFORMANCE COMPARISON GRAPHS
# ======================================================================

# Extract K values
k_values = [result[0] for result in AI_results]

# Extract metrics from stored results
AI_accuracy = [result[1] for result in AI_results]
A3_accuracy = [result[1] for result in A3_results]
manual_accuracy = [result[1] for result in manual_results]

AI_precision = [result[2] for result in AI_results]
A3_precision = [result[2] for result in A3_results]
manual_precision = [result[2] for result in manual_results]

AI_recall = [result[3] for result in AI_results]
A3_recall = [result[3] for result in A3_results]
manual_recall = [result[3] for result in manual_results]

AI_f1 = [result[4] for result in AI_results]
A3_f1 = [result[4] for result in A3_results]
manual_f1 = [result[4] for result in manual_results]


# ======================================================================
# Accuracy Graph
# ======================================================================

plt.figure(figsize=(8, 5))

plt.plot(k_values, AI_accuracy, marker='o', label="A1 - AI")
plt.plot(k_values, A3_accuracy, marker='o', label="A3 - Scikit")
plt.plot(k_values, manual_accuracy, marker='o', label="A4 - Manual")

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.show()


# ======================================================================
# Precision Graph
# ======================================================================

plt.figure(figsize=(8, 5))

plt.plot(k_values, AI_precision, marker='o', label="A1 - AI")
plt.plot(k_values, A3_precision, marker='o', label="A3 - Scikit")
plt.plot(k_values, manual_precision, marker='o', label="A4 - Manual")

plt.xlabel("K")
plt.ylabel("Precision")
plt.title("Precision Comparison")
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.show()


# ======================================================================
# Recall Graph
# ======================================================================

plt.figure(figsize=(8, 5))

plt.plot(k_values, AI_recall, marker='o', label="A1 - AI")
plt.plot(k_values, A3_recall, marker='o', label="A3 - Scikit")
plt.plot(k_values, manual_recall, marker='o', label="A4 - Manual")

plt.xlabel("K")
plt.ylabel("Recall")
plt.title("Recall Comparison")
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.show()


# ======================================================================
# F1 Score Graph
# ======================================================================

plt.figure(figsize=(8, 5))

plt.plot(k_values, AI_f1, marker='o', label="A1 - AI")
plt.plot(k_values, A3_f1, marker='o', label="A3 - Scikit")
plt.plot(k_values, manual_f1, marker='o', label="A4 - Manual")

plt.xlabel("K")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison")
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.show()


