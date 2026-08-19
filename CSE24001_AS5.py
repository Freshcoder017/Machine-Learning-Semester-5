import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt
#def goodformat(df):
def bubble(arr): #[5,2,6,4]
    for k in range(len(arr)):
        for i in range (len(arr)-1):
            if arr[i]>arr[i+1]:
                arr[i],arr[i+1]=arr[i+1],arr[i]
    return arr

def select(arr):
    for i in range(len(arr)-1):
        temp=arr[i]
        for j in range(i+1,len(arr)):
            if arr[j]<temp:
                arr[i],arr[j]=arr[j],arr[i]
    print(arr)

def dataimput(data):  # THIS IS DATA IMPUTATION USING MEAN OF THAT COLUMN 
    nmv=data.notnull()
    #print(nmv)
    for i in nmv:
        for j in range(len(data[i])):
            if nmv[i][j]==False:
                data[i][j]=np.mean(nmv[i])
    print(data)


def minkymink(vec1,vec2,p):
    d=0
    for i in range(len(vec1)):
        d+=abs(vec1[i]-vec2[i])**p
    d=d**(1/p)
    return d


def mykmeans(data, k):
    clusters=[]
    for i in range (k):
        clusters.append([]) # now cluster list will look like this [[],[],[]]- empty
    datacopy=data.copy()
    centroids = []
    for i in range (k):
        luckyindex=random.randrange(0,len(datacopy))
        c1=datacopy[luckyindex]
        datacopy.pop(luckyindex)
        centroids.append(c1)
    centroids=data[:k]
    #print("centroids: ",centroids)
    curr_centroids=centroids.copy()
    iters=0     #this is only for that no of iterations vs k comparision graph
    while True:
        iters+=1
        clusters=[]
        for i in range (k):
            clusters.append([])
        for i in data:
            #print(i)
            dists=[]
            for j in curr_centroids:
                dists.append(minkymink(i,j,2))
            trucentroid=dists.index(min(dists))
            clusters[trucentroid].append(i) #putting it into that cluster with min distance

        

        for i in range(len(curr_centroids)):
            xdim=0
            ydim=0
            for j in range(len(clusters[i])):
                xdim+=clusters[i][j][0]
                ydim+=clusters[i][j][1]
            curr_centroids[i]=[xdim/len(clusters[i]),ydim/len(clusters[i])]

        if curr_centroids==centroids:
            break           
        centroids = [x.copy() for x in curr_centroids]
            
            #break
    '''for i in range(len(clusters)):
        print("-----------CLUSTER-------------")
        print("centroid: ",centroids[i])
        print(clusters[i])
        #print(dists)'''
    return clusters,centroids,iters   


def myknn(target,data,k,p):
    mydict={}
    for i in range(len(data)):
        dist=minkymink(target,data[i],p) # this is now a distance value from the target point to all other points
        mydict[i]=dist
    vals=mydict.items()
    vals=[i[1] for i in vals] 
    vals.sort()
    #print(vals)
    neighs=vals[:k]
    #print(neighs)
    datapts=[]
    for i in neighs:
        for j in mydict:
            if mydict[j]==i:
                datapts.append(j)
    #print(datapts)
    return datapts[:k]

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

# SCIKIT LEARN STUFF
def A3(data,x,y):
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
    #print("xtrain:\n",xtrain,"\nxtest:\n",xtest,"\nytrain:\n",ytrain,"\nytest:\n",ytest)
    #print(len(ytest))
    return xtrain,xtest,ytrain,ytest 

def A456(data,x,y,k,testvec=None):
    neigh=knn(n_neighbors=k)
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


def fitit(xtrain,ytrain):
    cxtrain=xtrain.values.tolist()
    cytrain=ytrain.tolist()
    return cxtrain,cytrain

#ok so ytest shouldnt be here because thats what we're predicting
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






    


#----------------main---------------------
'''
data = [

[1,2],[2,1],[2,2],[3,2],[2,3],[1,3],[3,1],[2,4],[4,2],[3,3],
[4,1],[1,4],[5,2],[4,3],[3,5],[5,3],[2,5],[4,4],[5,5],[6,4],
[6,3],[5,4],[6,5],[4,6],[5,6],
[50,50],[51,49],[52,50],[50,52],[53,51],[54,50],[52,52],[55,51],[56,50],[54,53],
[57,52],[58,51],[55,55],[56,54],[57,55],[58,56],[59,54],[60,55],[59,57],[61,56],
[62,55],[60,58],[61,59],[63,57],[64,58],
[100,10],[101,11],[102,10],[103,12],[104,11],[105,13],[106,12],[107,11],[108,13],[109,12],
[110,14],[111,13],[112,15],[113,14],[114,16],[115,15],[116,17],[117,16],[118,18],[119,17],
[120,19],[121,18],[122,20],[123,19],[124,21],
[20,100],[21,101],[22,99],[23,102],[24,101],[25,103],[26,102],[27,104],[28,103],[29,105],
[30,104],[31,106],[32,105],[33,107],[34,106],[35,108],[36,107],[37,109],[38,108],[39,110],
[40,109],[41,111],[42,110],[43,112],[44,111]
]

k=4
clusters=mykmeans(data,k)
clusters=clusters[0]
neighs=myknn([4.2,3.9],data,5,2)

print("----cluster1-----")
print(clusters[0][0])
print("clus2")
print(clusters[0][1])
for i in range(len(clusters)):
    print("-----Cluster ",i,"-----------------")
    print(clusters[i])
mems=membership(clusters,neighs)
superover(mems,[4.2,3.9],neighs)

#myknn([40,70],data,2)
'''
########### LOAD THE CSV ####################################
'''
dataset=pd.read_csv("eeg_features.csv")
#del dataset['subject_id']
#del dataset['label']
X=dataset.drop(columns=['subject_id','label'])
Y=dataset['label']
A3(dataset,X,Y)
#making a test vector
lst=[]
for i in X:
    #print(dataset[i].mean())
    lst.append(dataset[i].mean())

acc=A456(dataset,X,Y,lst,3)
#print(dataset)
#print(len(dataset.values.tolist()))
lstformat=dataset.values.tolist()
'''
#OWN IMPLEMENTATIONS#    (without sklearn)

df=pd.read_csv('eeg_features.csv')
X=df.drop(columns=['subject','label'])
Y=df['label']
xt,xtst,yt,ytst=A3(df,X,Y)
cp1,cp2=fitit(xt,yt)
lst=[]
for i in X:
    #print(dataset[i].mean())
    lst.append(df[i].mean())
k=3
lol=predictit(lst,cp1,cp2,k)
if lol==1:
    print("Test subject belongs to schizophrenia")
else:
    print("Healthy")
myscore(xtst,ytst,xt,yt,k)

# A8 COMPARISION #
# COMMON CODES
'''
dataset=pd.read_csv("eeg_features.csv")
#del dataset['subject_id']
#del dataset['label']
X=dataset.drop(columns=['subject','label'])
Y=dataset['label']
xt,xtst,yt,ytst=A3(dataset,X,Y)
# SKLEARN PART
skacc=[]
for i in range (1,6):
    acc=A456(dataset,X,Y,i)
    skacc.append(acc)
# MY PART
myacc=[]
cp1,cp2=fitit(xt,yt)
for i in range (1,6):
    #predictit(xtst,cp1,cp2,i)
    mac=myscore(xtst,ytst,xt,yt,i)
    myacc.append(mac)
plt.plot(skacc,marker='o',label='Sklearn')
plt.plot(myacc,marker='o',label='My KNN')
plt.legend()
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k")
plt.show()
'''
#simply running kmeans
'''
dataset=pd.read_csv("eeg_features.csv")
#del dataset['subject_id']
#del dataset['label']
X=dataset.drop(columns=['subject','label'])
print(X)
'''
#sortings
'''
arr=[5,6,2,1,7]
select(arr)
'''
#data imputation
'''
dataset=pd.read_csv("eeg_features.csv")
dataimput(dataset)
'''





