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
    print(v,sdiv)
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


def centroid(cluster):
    if len(cluster)==0:
        return []

    c=[]
    cols=len(cluster[0]) # this is the no of columns like 2 for [2,3]

    for j in range(cols):
        s=0
        for i in range(len(cluster)):
            s+=cluster[i][j]
        c.append(s/len(cluster)) # finding the centroid 

    return c


def kmeans(data,k):
    centroids=[]
    for i in range(k):
        centroids.append(data[i])  #just picking first k

    while True:
        clusters=[]
        for i in range(k):
            clusters.append([]) # k clusters

        for point in data:
            md=minkymink(point,centroids[0],2)  # finding d from frst centroid
            idx=0

            for j in range(1,k):  # compare with the other centroids
                d=minkymink(point,centroids[j],2)
                if d<md:
                    md=d
                    idx=j

            clusters[idx].append(point)

        newcentroids=[]

        for cluster in clusters:
            if len(cluster)==0:
                newcentroids.append([])
            else:
                newcentroids.append(centroid(cluster))

        if newcentroids==centroids:
            break

        centroids=newcentroids

    return centroids,clusters    

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
    print("Cluster",i+1,"size:",len(clusters[i]))
         

