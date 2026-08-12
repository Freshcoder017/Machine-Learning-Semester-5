import time
import math
import random
import matplotlib.pyplot as plt
def make_data(size):    # this function is to generate random data for a given size eg 100 means it will generate 100 random datapoints
    data = []

    for i in range(size):
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        data.append([x, y])

    return data
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

    
#AI implementation of kmeans:
    

    


def distance(p1, p2):
    s = 0
    for i in range(len(p1)):
        s += (p1[i] - p2[i]) ** 2
    return math.sqrt(s)


def centroid(cluster):
    if len(cluster) == 0:
        return []

    cols = len(cluster[0])
    c = []

    for j in range(cols):
        s = 0
        for i in range(len(cluster)):
            s += cluster[i][j]
        c.append(s / len(cluster))

    return c


def AIkmeans(data, k):

    # Initialize centroids
    centroids = data[:k]
    iters=0
    while True:
        iters+=1
        # Create empty clusters
        clusters = [[] for _ in range(k)]

        # Assign each point to nearest centroid
        for point in data:

            minDist = distance(point, centroids[0])
            index = 0

            for i in range(1, k):
                d = distance(point, centroids[i])
                if d < minDist:
                    minDist = d
                    index = i

            clusters[index].append(point)

        # Compute new centroids
        newCentroids = []

        for i in range(k):
            if len(clusters[i]) == 0:
                newCentroids.append(centroids[i])
            else:
                newCentroids.append(centroid(clusters[i]))

        # Stop if converged
        if newCentroids == centroids:
            break

        centroids = newCentroids

    return clusters, centroids,iters

#main function
# THIS SAMPLE DATA IS AI GENERATED - i wanted a large dataset
''' This section is just using data as the parameter and comparing runtime
between self and AI generated code. In the end it plots the cluster graph but only one
for self graph replace ans1 with ans in lines 181 and 182 and for ai make it ans1
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
print("*****####*****MY Kmeans algorithm*****####*****")
curr=time.time()
ans=mykmeans(data,4)
end=time.time()
print("My kmeans runtime: ",end-curr)
print("*****####*****AI Kmeans algorithm*****####*****")
curr1=time.time()
ans1=AIkmeans(data,4)
end1=time.time()
print("AI kmeans runtime: ",end1-curr1)

if (end1-curr1)<(end-curr):
    print("AI algorithm is better")
else:
    print("Mine is better")

#ans=(clusters,centroids) where clusters - [[],[],[]] and centroids []
myclusts=ans1[0]
mycents=ans1[1]
print(mycents)

xpoints=[]
ypoints=[]
for i in myclusts:
    for j in i:
        xpoints.append(j[0])
        ypoints.append(j[1])
plt.plot(xpoints,ypoints,"o")
plt.plot([x[0] for x in mycents],[y[1] for y in mycents],'*r')
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.show()


#BELOW CODE IS FOR RUNTIME VS DATASET SIZE COMPARISION:
#this uses tht makedata function 

'''
sizes = [100, 500, 1000, 5000, 10000, 20000]

my_times = []
ai_times = []

runs = 10

for size in sizes:

    testdata = make_data(size)

    my_total = 0
    ai_total = 0

    for i in range(runs):

        # My K-means
        start = time.perf_counter()
        mykmeans(testdata, 4)
        end = time.perf_counter()
        my_total += end - start

        # AI K-means
        start = time.perf_counter()
        AIkmeans(testdata, 4)
        end = time.perf_counter()
        ai_total += end - start

    my_times.append(my_total / runs)
    ai_times.append(ai_total / runs)

print("Data Size\tMy K-means\tAI K-means")

for i in range(len(sizes)):
    print(
        sizes[i],
        "\t\t",
        my_times[i],
        "\t",
        ai_times[i]
    )
plt.plot(sizes, my_times, marker="o", label="My K-means")
plt.plot(sizes, ai_times, marker="o", label="AI K-means")

plt.xlabel("Number of data points")
plt.ylabel("Average runtime (seconds)")
plt.title("Runtime vs Dataset Size")

plt.legend()
plt.grid()

plt.show()
'''
#BELOW CODE TESTS K VALUE VS NO OF ITERATIONS 
#this is why iter variable was used in the kmeans function- to record no of iterations
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
ans = mykmeans(data, 4)
ans1 = AIkmeans(data, 4)

print("My iterations:", ans[2])
print("AI iterations:", ans1[2])
for k in range(2, 8):

    ans = mykmeans(data, k)
    ans1 = AIkmeans(data, k)

    print("k =", k,"| My:", ans[2],"| AI:", ans1[2])

ks = range(2, 8)

my_iterations = []
ai_iterations = []

for k in ks:

    ans = mykmeans(data, k)
    ans1 = AIkmeans(data, k)

    my_iterations.append(ans[2])
    ai_iterations.append(ans1[2])

plt.plot(ks, my_iterations, marker='o', label="My K-means")
plt.plot(ks, ai_iterations, marker='o', label="AI K-means")

plt.xlabel("Number of clusters (k)")
plt.ylabel("Iterations to convergence")
plt.title("K-means Convergence vs Number of Clusters")
plt.legend()
plt.grid()
plt.show()
'''