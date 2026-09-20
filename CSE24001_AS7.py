import pandas as pd
import math

def givecls():
    df=pd.read_csv("eeg_features.csv")
    lst=df["subject"].values.tolist()
    lst=[i[0][0] for i in lst]
    return lst

def a12(lst):
    hc=lst.count('h')
    sc=lst.count('s')
    ph=hc/len(lst)
    ps=sc/len(lst)
    plst=[ph,ps]
    s=0
    gini=0
    for i in plst:
        s+=(i*math.log2(i))
        gini+=i**2
    s*=-1
    gini=1-gini
    #print(s)
    return s,gini



def makebin(lst):
    low=min(lst)
    hi=max(lst)
    wbin=(hi-low)/4
    bins=[[],[],[],[]]
    for i in lst:
        if i>=low and i <= low+wbin:
            bins[0].append(i)
        if i>low+wbin and i<=low+(2*wbin):
            bins[1].append(i)
        if i>low+(2*wbin) and i<=low+(3*wbin):
            bins[2].append(i)
        if i>low+(3*wbin) and i<=low+(4*wbin):
            bins[3].append(i)
    #now we gonna categorize them
    clst=[]
    for i in lst:
        if i in bins[0]:
            clst.append(0)
        if i in bins[1]:
            clst.append(1)
        if i in bins[2]:
            clst.append(2)
        if i in bins[3]:
            clst.append(3)
    #print(clst)
    return clst
def iggain(df,entropy):  # get labels,column as df here and categorical list from makebin (or can i jus call makebin here itself?)
    d = {}
    for i in range(len(df)):
        col = float(df.iloc[i, 1])
        label = int(df.iloc[i, 0])
        d[col] = label
    #print(d)
    #now making a bin to label mapping
    mybins=makebin([float(i) for i in df.iloc[:,1]])
    binlabel={0:[],1:[],2:[],3:[]}
    for i in range(len(mybins)):
        binlabel[mybins[i]].append(d[float(df.iloc[i,1])])
    #print(binlabel) # now i have a mapping of bin 0: [alll labels associated w it].. etc
    # now calculate the entropy
    hs=[0,0,0,0]
    for i in range(len(binlabel)):
        s=0
        if len(binlabel[i]) == 0:
            continue
        for j in range(2):
            pie=binlabel[i].count(j)/(len(binlabel[i]))
            if pie !=0:
                s+=(-1*(pie*math.log2(pie)))
        hs[i]=s
    #print(hs)
    #now i have entropies, make weighted entropies
    weighted=0
    for i in range(len(hs)):
        weighted+=(len(binlabel[i])/28)*hs[i]
    #print(weighted)
    ig=entropy-weighted
    #print(ig)
    return ig

def a3(df):
    results=[]
    for i in range(2,26):
        results.append(iggain(df.iloc[:,[1,i]],1))
    #print(results)
    print("Max IG GAIN: ",max(results))
    col=results.index(max(results))
    print("Root node: ",df.columns[col+2])
    
    




    

'''
ans=givecls()
entr,gin=a12(ans)
print("Entropy: ",entr)
print("Gini: ",gin)
'''
#df for infogain
df=pd.read_csv("eeg_features.csv")
#df=df.iloc[:,[1,2]]
#print(df)
makebin([
    0.123398,
    0.100862,
    0.105317,
    0.130813,
    0.155822,
    0.104638,
    0.131420,
    0.100781,
    0.086851,
    0.086450,
    0.131200,
    0.149107,
    0.123159,
    0.100054,
    0.090257,
    0.122495,
    0.084547,
    0.101712,
    0.090635,
    0.119286,
    0.115480,
    0.130702,
    0.097046,
    0.101485,
    0.104461,
    0.200731,
    0.096176,
    0.146005
])
#iggain(df,1)
a3(df)