import pandas as pd 
import numpy as np 
import time
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
# SELF IMPLEMENTED 
def minkymink(vec1,vec2,p): 
    d=0 
    for i in range(len(vec1)): 
        d+=abs(vec1[i]-vec2[i])**p 
    d=d**(1/p) 
    return d 
def myknn(target,data,k,p): 
    mydict={} 
    for i in range(len(data)): 
        dist=minkymink(target,data[i],p) # this is now a distance value from the target point to all other points 
        mydict[i]=dist 
    # Sort distances while keeping their original indices
    vals=sorted(mydict.items(), key=lambda x:x[1])
    datapts=[]
    for i in vals[:k]:
        datapts.append(i[0])
    return datapts

def membership(clusters,chosen): 
    #its a list of lists 
    #chosen is a list of points like [20,100],[40,50] 
    memb=[] 
    #print(chosen) 
    for i in chosen: 
        print("checkign for ",i) 
        for j in range(len(clusters)): 
            if i in clusters[j]: 
                memb.append(j) # this gotta be the jth cluster so thats basically the name lol 
    #print(memb) 
    return memb 
 
 
def superover(members,target,neighs): 
    scores={} 
    #members have classes like [2, 1, 2, 1, 2] and neighs are those points[p1,p2,p3...] 
    for i in members: 
        if i not in scores: 
            scores[i]=0 
    for i in range(len(members)): 
        dis=minkymink(target,neighs[i],2) 
        if dis==0: 
            #print("This belongs to class ",members[i]) 
            return members[i] 
            #break 
        #now i update it in dictionary 
        scores[members[i]]+=(1/dis) 
    #print(scores) #{2: 7.278152721522868, 1: 1.084652289093281} 
    clus=-1 
    val=-1 
    for i in scores: 
        if scores[i]>val: 
            val=scores[i] 
            clus=i 
    #print(clus) 
    #print("Belongs to ",clus) 
    return clus 

#USING SKLEARN  
def A3(data,x,y): 
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3) 
    #print("xtrain:\n",xtrain,"\nxtest:\n",xtest,"\nytrain:\n",ytrain,"\nytest:\n",ytest) 
    #print(len(ytest)) 
    return xtrain,xtest,ytrain,ytest  
 

def A456(data,x,y,k,testvec=None): 
    neigh=KNeighborsClassifier(n_neighbors=k) 
    xtrain,xtest,ytrain,ytest=A3(data,x,y) 
    print(neigh.fit(xtrain,ytrain)) 
    print(neigh.score(xtest,ytest)) 
    try: 
        testvec=np.array(testvec) 
        pred=neigh.predict(testvec.reshape(1,-1)) 
        print(pred) 
    #print(X.columns.tolist()) 
        #return neigh.score(xtest,ytest) 
    except: 
        pass 
    return neigh.score(xtest,ytest) 
def predictit(xtest,cxtrain,cytrain,k):
    #xtest=xtest.values.tolist()
    neighs=myknn(xtest,cxtrain,k,2) #returns the indices
    lbls=[]
    for i in range(len(neighs)):
        lbls.append(cytrain[neighs[i]])
    #print(lbls)
    scz=lbls.count(1)
    hel=lbls.count(0)
    if scz>hel:
        return 1
    else:
        return 0

def myscore(xtest,ytest,xtrain,ytrain,k):
    #thepred=predictit(xtest,xtrain.values.tolist(),ytrain.tolist())
    count=0
    results=[]
    for i in range(len(xtest)):
        lst=xtest.iloc[i].tolist() #because i wanna be passing a list 
        thepred=predictit(lst,xtrain.values.tolist(),ytrain.tolist(),k)
        results.append(thepred)
    for i in range(len(ytest)):
        if results[i]==ytest.iloc[i]:
            count+=1

    print(count/len(ytest))
    return count/len(ytest)



# AI KNN 
def bubble_sort(arr): 
 
    arr = arr.copy() 
 
    n = len(arr) 
 
    for i in range(n): 
 
        for j in range(0, n - i - 1): 
 
            if arr[j][0] > arr[j + 1][0]: 
                arr[j], arr[j + 1] = arr[j + 1], arr[j] 
 
    return arr 
 

def distance(v1, v2): 
 
    total = 0 
 
    for i in range(len(v1)): 
 
        total += (v1[i] - v2[i]) ** 2 
 
    return total ** 0.5 
 

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
 

def genai_knn(X_train, y_train, test_point, k, sorting_algorithm="merge"): 
    # Column 0 = subject 
    # Column 1 = label 
    # Columns 2 onwards = features 
 
    distances = [] 
    for i in range(len(X_train)): 
 
        d = distance(X_train.iloc[i].values, test_point) 
         # Store: 
        # distance, label, index 
        distances.append( 
            (d, y_train.iloc[i], i) 
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
 
        healthy_distance = 0 
        schizophrenia_distance = 0 
 
        for d, label, index in nearest: 
 
            if label == 0: 
                healthy_distance += d 
 
            else: 
                schizophrenia_distance += d 
 
        # Smaller total distance wins 
        if healthy_distance < schizophrenia_distance: 
 
            prediction = 0 
 
        else: 
 
            prediction = 1 
 
    return prediction

#knn prediction for self implemented
def my_knn_predict_all(X_train, y_train, X_test, k):
    xtrain = X_train.values.tolist()
    ytrain = y_train.tolist()
    predictions = []
    for i in range(len(X_test)):
        test = X_test.iloc[i].tolist()
        pred = predictit(test,xtrain,ytrain,k)
        predictions.append(pred)

    return predictions
def predict_genai(X_train,y_train,X_test,k):
    predictions = []
    for i in range(len(X_test)):
        test = X_test.iloc[i].values
        pred = genai_knn(X_train,y_train,test,k)
        predictions.append(pred)
    return predictions
def predict_sklearn(X_train, y_train, X_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    predictions = neigh.predict(X_test)
    return predictions

# IMPLEMENTATION OF ACCURACY, PRECISION, RECALL, F1


def calc(y_actual, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_actual)):
        actual = y_actual.iloc[i]
        predicted = y_pred[i]
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Precision
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    # Recall
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
   # F1-score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1
# MAIN A3 PERFORMANCE COMPARISON
if __name__ == "__main__":
    dataset = pd.read_csv("eeg_features.csv")
    X = dataset.drop(columns=["subject", "label"])
    Y = dataset["label"]
    k = 3
    runs = 10
    results = {"My KNN": [], "Scikit-Learn KNN": [], "GenAI KNN": []}
    # RUN 10 TIMES
    for run in range(runs):
        print("\n========================================")
        print("RUN", run + 1)
        print("========================================")
        # Same train/test split for all three versions
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=run,stratify=Y)
        #self implementation
        start = time.perf_counter()
        my_predictions = my_knn_predict_all( X_train, y_train,  X_test,k)
        end = time.perf_counter()
        my_time = end - start
        my_accuracy, my_precision, my_recall, my_f1 = (calc(y_test, my_predictions)        )
        results["My KNN"].append([my_accuracy,my_precision,my_recall,my_f1, my_time])
#SKLEARN IMPLEMENTATION
        start = time.perf_counter()
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train,y_train)
        sklearn_predictions = model.predict(X_test)
        end = time.perf_counter()
        sklearn_time = end - start
        sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1 = (calc(y_test,sklearn_predictions))
        results["Scikit-Learn KNN"].append([
            sklearn_accuracy,
            sklearn_precision,
            sklearn_recall,
            sklearn_f1,
            sklearn_time
        ])

        # ====================================================
        # 3. GENAI KNN
        # ====================================================

        start = time.perf_counter()

        genai_predictions = predict_genai(X_train, y_train, X_test,k)

        end = time.perf_counter()

        genai_time = end - start

        genai_accuracy, genai_precision, genai_recall, genai_f1 = (
            calc(
                y_test,
                genai_predictions
            )
        )

        results["GenAI KNN"].append([
            genai_accuracy,
            genai_precision,
            genai_recall,
            genai_f1,
            genai_time
        ])

        # ====================================================
        # PRINT CURRENT RUN
        # ====================================================

        print("\nMy KNN")
        print("Accuracy :", my_accuracy)
        print("Precision:", my_precision)
        print("Recall   :", my_recall)
        print("F1-score :", my_f1)
        print("Time     :", my_time)

        print("\nScikit-Learn KNN")
        print("Accuracy :", sklearn_accuracy)
        print("Precision:", sklearn_precision)
        print("Recall   :", sklearn_recall)
        print("F1-score :", sklearn_f1)
        print("Time     :", sklearn_time)

        print("\nGenAI KNN")
        print("Accuracy :", genai_accuracy)
        print("Precision:", genai_precision)
        print("Recall   :", genai_recall)
        print("F1-score :", genai_f1)
        print("Time     :", genai_time)


    # ========================================================
    # AVERAGE RESULTS OF 10 RUNS
    # ========================================================

    final_results = []

    for algorithm in results:

        values = np.array(
            results[algorithm]
        )

        final_results.append({
            "Algorithm": algorithm,
            "Accuracy": values[:, 0].mean(),
            "Precision": values[:, 1].mean(),
            "Recall": values[:, 2].mean(),
            "F1-score": values[:, 3].mean(),
            "Avg Time (s)": values[:, 4].mean()
        })


    # ========================================================
    # FINAL TABLE
    # ========================================================

    results_df = pd.DataFrame(
        final_results
    )

    print("\n\n========================================")
    print("       FINAL AVERAGE RESULTS")
    print("========================================")

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n========================================")
    print("       EXPERIMENT COMPLETED")
    print("========================================")