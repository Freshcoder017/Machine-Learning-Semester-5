#26/7/26
#I Used AI for k means and A10 and bit of help with A8
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def A4(vec1,vec2):
    if len(vec1) != len(vec2):
        return "Difference in dimensionality"
    msum=0
    for i in range (len(vec1)):
        msum+= abs(vec1[i]-vec2[i])
    print(msum)

    euc=0
    for i in range(len(vec1)):
        euc+=(vec1[i]-vec2[i])**2
    euc=euc**0.5
    print(euc)
    mink=0
    for i in range(len(vec1)):
        mink+=abs(vec1[i]-vec2[i])**len(vec1)
    mink=mink**(1/len(vec1))
    print(mink)
    return msum,euc,mink

#print(scipy.spatial.distance.minkowski([1,2],[4,6]))
def A7(vec1,vec2):
    if len(vec1) != len(vec2):
            return "Difference in dimensionality"
    dot=0
    for i in range(len(vec1)):
         dot+=vec1[i]*vec2[i]
    lv1=0
    lv2=0
    for i in range(len(vec1)):
         lv1+=abs(vec1[i])**len(vec1)
    lv1=lv1**(1/len(vec1))
    lv2=0
    for i in range(len(vec2)):
        lv2+=abs(vec2[i])**(len(vec2))
    lv2=lv2**(1/len(vec2))
    return lv1,lv2

def A8(lst):
    mean=sum(lst)/len(lst)
    v=0
    for i in range(len(lst)):
         v+=(lst[i]-mean)**2
    v=v/len(lst)
    sdiv=v**0.5
    return mean,v,sdiv
def A8matrix(mat):
    means=[]
    vars=[]
    sds=[]
    rows=len(mat)
    cols=len(mat[0])

    for j in range(cols):
        col=[]
        for i in range(rows):
            col.append(mat[i][j])

        m=sum(col)/len(col)

        v=0
        for x in col:
            v+=(x-m)**2
        v=v/len(col)

        s=v**0.5

        means.append(m)
        vars.append(v)
        sds.append(s)

    return means,vars,sds


def kmeans(data, k, max_iter=300, tol=1e-4):
    centroids = []  # this is gonna be my list of center points

    for i in range(k):
        first = data[i]
        centroids.append(first)  # just picking first k

    for iteration in range(max_iter):
        clusters = []
        for i in range(k):
            emclus = []  # basically an empty cluster list
            clusters.append(emclus)

        for i in range(len(data)):
            point = data[i]

            firstc = centroids[0]  # finding the dist from 1st centroid
            md = minkymink(point, firstc, 2)
            idx = 0

            for j in range(1, k):
                curr = centroids[j]
                d = minkymink(point, curr, 2)

                if d < md:  # if its closer
                    md = d
                    idx = j

            target = clusters[idx]
            target.append(point)  # point append karoo

        newcentroids = []  # new centers

        for cid in range(len(clusters)):
            cluster = clusters[cid]  # this is a cluster

            if len(cluster) == 0:
                empty_center = centroids[cid]  # retain previous center
                newcentroids.append(empty_center)
            else:
                c = []
                cols = len(cluster[0])  # no of columns

                for j in range(cols):
                    s = 0

                    for i in range(len(cluster)):
                        current_point = cluster[i]
                        val = current_point[j]
                        s = s + val

                    total_points = len(cluster)
                    mean_value = s / total_points  # col avg
                    c.append(mean_value)

                newcentroids.append(c)

        shift = max(
            minkymink(centroids[i], newcentroids[i], 2) for i in range(k)
        )
        centroids = newcentroids

        if shift < tol:  # CONVERGENCE CONDITIONN
            break

    return centroids, clusters
def A2lbl(df,col,mapping):
    try:
        df['Encoded']=df[col].map(mapping)
        return df
    except:
         print("Column probably doesnt exist lolol skill ishh")
def A2onehot(df,col):
    try:
        ucol=df[col].unique()

        for i in ucol:
            df[i]=0

        for i in range(len(df)):
            value=df.loc[i,col]
            df.loc[i,value]=1

        return df

    except:
        print("Column probably doesnt exist lol")

def minkymink(vec1,vec2,p):
    d=0
    for i in range(len(vec1)):
        d+=abs(vec1[i]-vec2[i])**p
    d=d**(1/p)
    return d


'''     
#main
#A2
data=pd.DataFrame({'Students':['Satya','Ishaan','Thejas'],'Clubs':['club01','club2','club3']})     
print(data)
mappings={'club01':1,'club2':2,'club3':3}
new=A2lbl(data,'Clubs',mappings)

print(new)
df2=pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
new=A2onehot(df2,"Education")
print(new.head())

#A4
man,eu,mink=A4([1,2,3],[4,5,6])
print("Manhattan dist: ",man)
print("Euclidiean dist: ",eu)
print("Minkowski dist: ",mink)
print("From library: ",scipy.spatial.distance.minkowski([1,2,3],[4,5,6]))
#A5
df3=df2.drop(columns=["Education","Marital_Status","Dt_Customer"])

v1=df3.iloc[0].tolist()
v2=df3.iloc[1].tolist()
for i in range(1,11):
    ans=minkymink(v1,v2,i)
    print("Minkowski distance for p=",i," : ",ans)

#A6
print("COMPARING WITH THE LIBRARY'S MINKYMINKY METHOD")
for i in range(1,11):
    ans=scipy.spatial.distance.minkowski(v1,v2,i)
    print("Minkowski distance for p=",i," : ",ans)


#A7
l1,l2=A7([1,2,3],[6,7,7])
print("Length of vec 1: ",l1)
print("Length of vec2: ",l2)

#A8
#A8
mat=df3.values.tolist()
means,vars,sds=A8matrix(mat)

print("Mean")
print(means)

print("Variance")
print(vars)

print("Standard deviation")
print(sds)


#A9
print("Mean from numpy")
print(np.mean(df3.values,axis=0))

print("Std from numpy")
print(np.std(df3.values,axis=0))


#A10
feature=df3.columns[0]

plt.hist(df3[feature],bins=10)
plt.title(feature)
plt.show()

print("Mean:",df3[feature].mean())
print("Variance:",df3[feature].var())


#A11
centroids,clusters=kmeans(mat,3)

print("Centroids")
for i in centroids:
    print(i)

for i in range(len(clusters)):
    print("Cluster",i+1,"has :",len(clusters[i]), "Elements")

# output was cluster 1- 2240, 2 and 3 had 0 elements so its gotta be bad clustering/ elbow point occurs at k=1????
'''

# =====AI GENERATED UNIT TEST CASES=======
# FUNCTION : A2LBL
# case 1
df = pd.DataFrame({"Club":["A","B","A"]})
mapping = {"A":1,"B":2}

print(A2lbl(df,"Club",mapping))
# case 2 - invalid col
df = pd.DataFrame({"Club":["A","B"]})
mapping={"A":1}

A2lbl(df,"WrongColumn",mapping)
# case 3: missing mapp
df=pd.DataFrame({"Club":["A","C"]})
mapping={"A":1}

print(A2lbl(df,"Club",mapping))

#FUNCTION : A2Onehot
#CASE 1:
df=pd.DataFrame({"Color":["Red","Blue","Red"]})
print(A2onehot(df,"Color"))
#case2:
df=pd.DataFrame({"Color":["Red","Red"]})
print(A2onehot(df,"Color"))
#case 3:
print(A2onehot(df,"ABC"))

#FUNCTION: A4
#case1:
print(A4([1,2,3],[4,5,6]))
#case2:
print(A4([1,2],[1,2]))
#case3:
print(A4([1,2],[1,2,3]))

#FUNCTION : MINKYMINK
print(minkymink([1,2],[4,6],1))  

# FUNCTION : A7
#case1:
print(A7([3,4],[5,12]))
#case 2:
print(A7([0,0],[0,0]))
#case 3:
print(A7([1,2],[1,2,3]))

# FUNCTION : A8:
#case 1:
print(A8([10,20,30,40,50]))
#case2:
print(A8([5,5,5,5]))
#case3:
print(A8([-2,-1,0,1,2]))

#FUNCTION: A8MATRIX
#cse1
print(A8matrix(mat=[
[1,2],
[3,4],
[5,6]
]))
#case2
print(A8matrix(mat=[
[5,5],
[5,5]
]))

#FUNCTION: KMEANS
print(kmeans([[1,1],[1,2],[8,8],[9,9]],k=2))
#case2
print(kmeans([[1,1],[2,2],[3,3]],1))
#case3
print(kmeans([[1,1],[2,2]],3))