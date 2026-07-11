# Q1
#lst=[5] it failed for this so i added the len(lst) condition
'''
def paircount(lst):
    pairs={}
    for i in lst:
        if (10-i in lst) and (10-i not in pairs) and (len(lst)>1):
            pairs[i]=10-i      
            print(i,10-i)
    print(pairs)
    print("no of pairs: ",len(pairs))
    
lst=[2,7,4,1,3,6]
paircount(lst)'''

#Q2
'''
ip=eval(input("Enter list of real numbers: "))

if len(ip)<3:
    print("Cannot determine range")
else:
    print("Range is ",max(ip)-min(ip))'''
#Q3
'''
def printmat(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end=" ")
        print()

def sqrmat(matrix, m):

    n = len(matrix)
    copy = []
    for i in range(n):
        copy.append(matrix[i][:])

    for a in range(m-1):
        result = []
        for i in range(n):
            result.append([0] * n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += matrix[i][k] * copy[k][j]
        matrix = result
    printmat(matrix)

sqrmat([[1,2,3],[4,5,6],[7,8,9]],2)'''
#4
'''
def countmost(ip):
    c=ip[0]
    val=ip.count(c)
    for i in range(1,len(ip)):
        if ip.count(ip[i])>val:
            c=ip[i]
            val=ip.count(ip[i])
    print(c," occurs the most and it occurs ",val," times")
    
ip=input("Enter the string: ")
countmost(ip)'''
#5
'''
def stats(lst):
    mean=sum(lst)/len(lst)
    mode=0
    app=0
    for i in lst:
        if lst.count(i)>app:
            mode=i
            app=lst.count(i)
    print("List: ",lst)
    lst.sort()
    median=lst[12]
    print("Mean: ",mean)
    print("Mode: ",mode)
    print("Median: ",median)


import random
lst=[]
for i in range(25):
    lst.append(random.randrange(1,10))
stats(lst)
'''

            
    
