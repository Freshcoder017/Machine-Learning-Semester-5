import scipy 
import pandas as pd
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

def A2lbl(df,col,mapping):
    try:
        df['Encoded']=df[col].map(mapping)
        return df
    except:
         print("Column probably doesnt exist")

     
#main
#A2
data=pd.DataFrame({'Students':['Satya','Ishaan','Thejas'],'Clubs':['club01','club2','club3']})     
print(data)
mappings={'club01':1,'club2':2,'club3':3}
new=A2lbl(data,'Clubs',mappings)

print(new)

#A4
man,eu,mink=A4([1,2,3],[4,5,6])
print("Manhattan dist: ",man)
print("Euclidiean dist: ",eu)
print("Minkowski dist: ",mink)
print("From library: ",scipy.spatial.distance.minkowski([1,2,3],[4,5,6]))

#A7
l1,l2=A7([1,2,3],[6,7,7])
print("Length of vec 1: ",l1)
print("Length of vec2: ",l2)





#A8([2,4])
         

