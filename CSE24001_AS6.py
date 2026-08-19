import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# --------------------------------------------------
# 1. ENCODING
# --------------------------------------------------
def encode(data):
    data = data.copy()

    # Subject is an identifier, NOT a feature
    # Label is already 0/1, so nothing needs to be done
    # All remaining columns are numerical

    return data


# --------------------------------------------------
# 2. DATA IMPUTATION
# --------------------------------------------------
def data_imputation(data):
    data = data.copy()

    # Impute only the feature columns
    feature_columns = data.columns[2:]

    for col in feature_columns:
        data[col] = data[col].fillna(data[col].mean())

    return data


# --------------------------------------------------
# 3. EUCLIDEAN DISTANCE
# --------------------------------------------------
def distance(v1, v2):

    total = 0

    for i in range(len(v1)):
        total += (v1[i] - v2[i]) ** 2

    return total ** 0.5


# --------------------------------------------------
# 4. BUBBLE SORT
# --------------------------------------------------
def bubble_sort(arr):

    arr = arr.copy()

    n = len(arr)

    for i in range(n):

        for j in range(0, n - i - 1):

            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


# --------------------------------------------------
# 5. SELECTION SORT
# --------------------------------------------------
def selection_sort(arr):

    arr = arr.copy()

    n = len(arr)

    for i in range(n):

        min_index = i

        for j in range(i + 1, n):

            if arr[j][0] < arr[min_index][0]:
                min_index = j

        arr[i], arr[min_index] = arr[min_index], arr[i]

    return arr


# --------------------------------------------------
# 6. MERGE SORT
# --------------------------------------------------
def merge(left, right):

    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):

        if left[i][0] <= right[j][0]:
            result.append(left[i])
            i += 1

        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def merge_sort(arr):

    if len(arr) <= 1:
        return arr.copy()

    mid = len(arr) // 2

    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


# --------------------------------------------------
# 7. KNN
# --------------------------------------------------
def knn(data, test_index, k, sorting_algorithm="merge"):

    # -----------------------------------------------
    # FEATURES
    # -----------------------------------------------
    # Column 0 = subject
    # Column 1 = label
    # Columns 2 onwards = features

    X = data.iloc[:, 2:]

    # Label is column 1
    y = data.iloc[:, 1]

    # Test point
    test_point = X.iloc[test_index].values

    distances = []

    # -----------------------------------------------
    # CALCULATE DISTANCES
    # -----------------------------------------------

    for i in range(len(X)):

        # Don't compare the test point with itself
        if i == test_index:
            continue

        d = distance(
            X.iloc[i].values,
            test_point
        )

        # Store:
        # distance, label, index
        distances.append(
            (d, y.iloc[i], i)
        )

    # -----------------------------------------------
    # SORT
    # -----------------------------------------------

    if sorting_algorithm == "bubble":

        distances = bubble_sort(distances)

    elif sorting_algorithm == "selection":

        distances = selection_sort(distances)

    else:

        distances = merge_sort(distances)

    # -----------------------------------------------
    # SELECT K NEAREST NEIGHBOURS
    # -----------------------------------------------

    nearest = distances[:k]

    print("\nK Nearest Neighbours:")

    for d, label, index in nearest:

        print(
            "Subject:", data.iloc[index, 0],
            "Distance:", d,
            "Class:", label
        )

    # -----------------------------------------------
    # MAJORITY VOTING
    # -----------------------------------------------

    healthy_votes = 0
    schizophrenia_votes = 0

    for d, label, index in nearest:

        if label == 0:
            healthy_votes += 1

        else:
            schizophrenia_votes += 1

    print("\nVotes:")
    print("Healthy:", healthy_votes)
    print("Schizophrenia:", schizophrenia_votes)

    # -----------------------------------------------
    # MAJORITY VOTE
    # -----------------------------------------------

    if healthy_votes > schizophrenia_votes:

        prediction = 0

    elif schizophrenia_votes > healthy_votes:

        prediction = 1

    # -----------------------------------------------
    # TIE BREAKING
    # -----------------------------------------------

    else:

        print("\nTie detected!")

        healthy_distance = 0
        schizophrenia_distance = 0

        for d, label, index in nearest:

            if label == 0:
                healthy_distance += d

            else:
                schizophrenia_distance += d

        print(
            "Healthy total distance:",
            healthy_distance
        )

        print(
            "Schizophrenia total distance:",
            schizophrenia_distance
        )

        # Smaller total distance wins
        if healthy_distance < schizophrenia_distance:

            prediction = 0

        else:

            prediction = 1

    # -----------------------------------------------
    # RESULT
    # -----------------------------------------------

    if prediction == 0:
        print("\nPrediction: HEALTHY")

    else:
        print("\nPrediction: SCHIZOPHRENIA")

    return prediction
# --------------------------------------------------
# 7. WEIGHTED KNN
# --------------------------------------------------
def wknn(data, test_index, k, sorting_algorithm="merge"):

    # Column 0 = subject
    # Column 1 = label
    # Columns 2 onwards = features

    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]

    # Test point
    test_point = X.iloc[test_index].values

    distances = []

    # -----------------------------------------------
    # CALCULATE DISTANCES
    # -----------------------------------------------

    for i in range(len(X)):

        # Don't compare test point with itself
        if i == test_index:
            continue

        d = distance(
            X.iloc[i].values,
            test_point
        )

        # distance, label, index
        distances.append(
            (d, y.iloc[i], i)
        )

    # -----------------------------------------------
    # SORT
    # -----------------------------------------------

    if sorting_algorithm == "bubble":
        distances = bubble_sort(distances)

    elif sorting_algorithm == "selection":
        distances = selection_sort(distances)

    else:
        distances = merge_sort(distances)

    # -----------------------------------------------
    # SELECT K NEAREST NEIGHBOURS
    # -----------------------------------------------

    nearest = distances[:k]

    print("\nK Nearest Neighbours:")

    for d, label, index in nearest:

        print(
            "Subject:", data.iloc[index, 0],
            "Distance:", d,
            "Class:", label
        )

    # -----------------------------------------------
    # WEIGHTED VOTING
    # -----------------------------------------------

    healthy_weight = 0
    schizophrenia_weight = 0

    print("\nWeighted Votes:")

    for d, label, index in nearest:

        # Weight = 1 / distance
        weight = 1 / d

        print(
            "Subject:", data.iloc[index, 0],
            "Distance:", d,
            "Weight:", weight,
            "Class:", label
        )

        if label == 0:
            healthy_weight += weight

        else:
            schizophrenia_weight += weight

    print("\nTotal Weighted Votes:")
    print("Healthy:", healthy_weight)
    print("Schizophrenia:", schizophrenia_weight)

    # -----------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------

    if healthy_weight > schizophrenia_weight:

        prediction = 0

    elif schizophrenia_weight > healthy_weight:

        prediction = 1

    # -----------------------------------------------
    # TIE BREAKING
    # -----------------------------------------------

    else:

        print("\nWeighted vote tie detected!")

        # Since weighted votes are already based
        # on distance, choose the class of the
        # closest neighbour.

        closest_distance = nearest[0][0]
        closest_label = nearest[0][1]

        prediction = closest_label

        print(
            "Closest neighbour distance:",
            closest_distance
        )

    # -----------------------------------------------
    # RESULT
    # -----------------------------------------------

    if prediction == 0:

        print("\nPrediction: HEALTHY")

    else:

        print("\nPrediction: SCHIZOPHRENIA")

    return prediction



def A3(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

from sklearn.neighbors import KNeighborsClassifier

def A4(X_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    return neigh
def A5(neigh,X_test, y_test):
    score = neigh.score(X_test, y_test)
    return score
def A6(neigh, Xtest):
    y_pred = neigh.predict(Xtest)
    return y_pred
def fitit(xtrain, ytrain):
    """
    Store the training data.
    No actual training is required for KNN.
    """
    cxtrain = xtrain.copy()
    cytrain = ytrain.copy()

    return cxtrain, cytrain


def predictit(xtest, cxtrain, cytrain, k):
    """
    Predict the class of each test vector using
    custom KNN with distance-weighted voting.
    """

    predictions = []

    # Convert xtest to numpy array to iterate over rows instead of column names
    xtest_values = xtest.values if isinstance(xtest, pd.DataFrame) else xtest

    for test_vector in xtest_values:

        distances = []

        # Calculate distance from test vector
        # to every training vector
        for i in range(len(cxtrain)):

            d = distance(
                test_vector,
                cxtrain.iloc[i].values
            )

            distances.append(
                (d, cytrain.iloc[i])
            )

        # Sort according to distance
        distances = merge_sort(distances)

        # Take k nearest neighbours
        neighbours = distances[:k]

        healthy_weight = 0
        schizophrenia_weight = 0

        # Weighted voting
        for d, label in neighbours:

            # Handle identical points
            if d == 0:
                weight = float('inf')
            else:
                weight = 1 / d

            if label == 0:
                healthy_weight += weight
            else:
                schizophrenia_weight += weight

        # Majority weighted vote
        if healthy_weight > schizophrenia_weight:
            prediction = 0

        elif schizophrenia_weight > healthy_weight:
            prediction = 1

        else:
            # Tie → closest neighbour wins
            prediction = neighbours[0][1]

        predictions.append(prediction)

    return predictions

def myscore(xtest, ytest, xtrain, ytrain, k):
    """
    Calculate accuracy of the custom KNN classifier.
    """

    # Store training data
    cxtrain, cytrain = fitit(xtrain, ytrain)

    # Predict test data
    predictions = predictit(
        xtest,
        cxtrain,
        cytrain,
        k
    )

    # Count correct predictions
    correct = 0

    for i in range(len(ytest)):

        if predictions[i] == ytest.iloc[i]:
            correct += 1

    # Accuracy
    score = correct / len(ytest)

    return score
#----------------------------------------------------------------------------------------------
# Main function
#A1
'''
data = pd.read_csv("eeg_features.csv")

data = encode(data)

data = data_imputation(data)

prediction = knn(
    data,
    test_index=0,
    k=5,
    sorting_algorithm="merge"

)
'''
#A2
'''
data = pd.read_csv("eeg_features.csv")

data = encode(data)

data = data_imputation(data)

prediction = wknn(
    data,
    test_index=0,
    k=5,
    sorting_algorithm="merge"

)
'''
#A4
# A4-A7
'''
dataset = pd.read_csv("eeg_features.csv")
dataset = encode(dataset)
dataset = data_imputation(dataset)
X = dataset.drop(columns=['subject', 'label'])
Y = dataset['label']
xt, xtst, yt, ytst = A3(X, Y)
knnmodel = A4(xt, yt)
score = A5(knnmodel, xtst, ytst)

print("Accuracy:", score)
print("Accuracy (%):", score * 100)
y_pred = A6(knnmodel, xtst)

print("Predicted labels:")
print(y_pred)
print("Without SKLEARN:")
cxtrain, cytrain = fitit(xt, yt)
predictions = predictit(
    xtst,
    cxtrain,
    cytrain,
    3
)
score = myscore(
    xtst,
    ytst,
    xt,
    yt,
    3
)

print("Custom KNN Accuracy:", score)
print("Accuracy (%):", score * 100)
print(predictions)
'''
#A8
'''
# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

dataset = pd.read_csv("eeg_features.csv")

dataset = encode(dataset)
dataset = data_imputation(dataset)

# Remove subject and label from feature set
X = dataset.drop(columns=['subject', 'label'])
Y = dataset['label']

# Train-test split
xt, xtst, yt, ytst = A3(X, Y)

# Lists to store accuracy for each k
k_values = range(1, 11)

custom_scores = []
sklearn_scores = []

# Compare for different values of k
for k in k_values:

    # ----------------------------------------------
    # CUSTOM KNN
    # ----------------------------------------------

    custom_score = myscore(
        xtst,
        ytst,
        xt,
        yt,
        k
    )

    custom_scores.append(custom_score)

    # ----------------------------------------------
    # SKLEARN KNN
    # ----------------------------------------------

    neigh = KNeighborsClassifier(
        n_neighbors=k
    )

    neigh.fit(xt, yt)

    sklearn_score = neigh.score(
        xtst,
        ytst
    )

    sklearn_scores.append(sklearn_score)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\nComparison of Custom KNN and Scikit-learn KNN")
print("-----------------------------------------------")
print("k\tCustom\t\tScikit-learn")

for i in range(len(k_values)):

    print(
        k_values[i],
        "\t",
        custom_scores[i],
        "\t\t",
        sklearn_scores[i]
    )


# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    custom_scores,
    marker='o',
    label="Custom KNN"
)

plt.plot(
    k_values,
    sklearn_scores,
    marker='s',
    label="Scikit-learn KNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Custom KNN vs Scikit-learn KNN")

plt.xticks(list(k_values))
plt.legend()
plt.grid(True)

plt.show()
'''
# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

dataset = pd.read_csv("eeg_features.csv")

dataset = encode(dataset)
dataset = data_imputation(dataset)

# Remove subject and label from features
X = dataset.drop(columns=['subject', 'label'])
Y = dataset['label']

# Train-test split
xt, xtst, yt, ytst = A3(X, Y)

# Range of k values
k_values = range(1, 11)

custom_scores = []
weighted_scores = []
sklearn_scores = []


# --------------------------------------------------
# COMPARE FOR DIFFERENT VALUES OF K
# --------------------------------------------------

for k in k_values:

    # ----------------------------------------------
    # 1. CUSTOM NORMAL KNN
    # ----------------------------------------------

    custom_score = myscore(
        xtst,
        ytst,
        xt,
        yt,
        k
    )

    custom_scores.append(custom_score)


    # ----------------------------------------------
    # 2. CUSTOM WEIGHTED KNN
    # ----------------------------------------------

    weighted_predictions = []

    for test_vector in xtst.values:

        distances = []

        # Calculate distances to training data
        for i in range(len(xt)):

            d = distance(
                test_vector,
                xt.iloc[i].values
            )

            distances.append(
                (d, yt.iloc[i])
            )

        # Sort distances
        distances = merge_sort(distances)

        # Take k nearest neighbours
        neighbours = distances[:k]

        healthy_weight = 0
        schizophrenia_weight = 0

        # Weighted voting
        for d, label in neighbours:

            if d == 0:
                weight = float('inf')
            else:
                weight = 1 / d

            if label == 0:
                healthy_weight += weight
            else:
                schizophrenia_weight += weight

        # Decide class
        if healthy_weight > schizophrenia_weight:
            prediction = 0

        elif schizophrenia_weight > healthy_weight:
            prediction = 1

        else:
            # Tie → closest neighbour
            prediction = neighbours[0][1]

        weighted_predictions.append(prediction)


    # Calculate weighted KNN accuracy
    correct = 0

    for i in range(len(ytst)):

        if weighted_predictions[i] == ytst.iloc[i]:
            correct += 1

    weighted_score = correct / len(ytst)

    weighted_scores.append(weighted_score)


    # ----------------------------------------------
    # 3. SCIKIT-LEARN KNN
    # ----------------------------------------------

    neigh = KNeighborsClassifier(
        n_neighbors=k
    )

    neigh.fit(xt, yt)

    sklearn_score = neigh.score(
        xtst,
        ytst
    )

    sklearn_scores.append(sklearn_score)


# --------------------------------------------------
# PRINT COMPARISON
# --------------------------------------------------

print("\nKNN COMPARISON")
print("-----------------------------------------------")
print("k\tCustom\tWeighted\tScikit-learn")

for i in range(len(k_values)):

    print(
        k_values[i],
        "\t",
        custom_scores[i],
        "\t",
        weighted_scores[i],
        "\t\t",
        sklearn_scores[i]
    )


# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    k_values,
    custom_scores,
    marker='o',
    label="Custom KNN"
)

plt.plot(
    k_values,
    weighted_scores,
    marker='s',
    label="Custom Weighted KNN"
)

plt.plot(
    k_values,
    sklearn_scores,
    marker='^',
    label="Scikit-learn KNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Comparison of KNN Implementations")

plt.xticks(list(k_values))
plt.legend()
plt.grid(True)

plt.show()