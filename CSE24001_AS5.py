import random
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


def myknn(target,data,k):
    mydict={}
    for i in range(len(data)):
        dist=minkymink(target,data[i],2) # this is now a distance value from the target point to all other points
        mydict[i]=dist
    vals=mydict.items()
    vals=[i[1] for i in vals] 
    vals.sort()
    print(vals)
    neighs=vals[:k]
    print(neighs)
    datapts=[]
    for i in neighs:
        for j in mydict:
            if mydict[j]==i:
                datapts.append(data[j])
    #print(datapts)
    return datapts[:k]

def membership(clusters,chosen):
    #its a list of lists
    #chosen is a list of points like [20,100],[40,50]
    memb=[]
    print(chosen)
    for i in chosen:
        print("checkign for ",i)
        for j in range(len(clusters)):
            if i in clusters[j]:
                memb.append(j) # this gotta be the jth cluster so thats basically the name lol
    print(memb)



#----------------main---------------------
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
neighs=myknn([4,4],data,5)

print("----cluster1-----")
print(clusters[0][0])
print("clus2")
print(clusters[0][1])
for i in range(len(clusters)):
    print("-----Cluster ",i,"-----------------")
    print(clusters[i])
membership(clusters,neighs)

#myknn([40,70],data,2)
