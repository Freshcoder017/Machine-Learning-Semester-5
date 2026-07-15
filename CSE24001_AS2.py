#A1
from numpy import linalg
import openpyxl as xl
import numpy as np
from numpy.linalg import matrix_rank
import math
X=[]
y=[]
def A1():
    global X
    global y
    wb=xl.load_workbook("Lab Session Data.xlsx")
    sheet=wb["Purchase data"]
    i=0
    #X=[]
    #y=[]
    '''
    for row in sheet.iter_rows(max_col=5,values_only=True):    
        if i==0:
            X.append(list(row))
        else:
            y.append(list(row))
        i+=1'''
    for row in sheet.iter_rows(min_row=2, values_only=True):
        X.append(list(row[1:4]))   
        y.append(row[4])          

    X = np.array(X)
    rank=matrix_rank(X)
    #print(matrix_rank(X))
    #print(X)
    #print("loll")
    #print(y)

    X_pinv = np.linalg.pinv(X)
    cost = X_pinv @ y
    #print(X_pinv)
    #print("Cost of Candy :", cost[0])
    #print("Cost of Mango :", cost[1])
    #print("Cost of Milk  :", cost[2])
    return rank,cost[0],cost[1],cost[2]

def A2():
    global X,y
    classifier={}
    for i in range(len(y)):
        if y[i]>200:
            classifier["C_"+str(i+1)]="RICH"
        else:
            classifier["C_"+str(i+1)]="POOR"
    return classifier
def mean(lst):
    s=sum(lst)
    return s/len(lst)

def var(lst):
    res=0
    mn=mean(lst)
    for i in lst:
        res+= (i-mn)**2
    res=res/len(lst)
    return res


    
def A3():
    wb=xl.load_workbook("Lab Session Data.xlsx")
    sheet=wb["IRCTC Stock Price"]
    prices=[]
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if row[4] is not None:
            prices.append(row[4])          
    #print(mean(prices))
    #print(var(prices))
    #print(np.mean(prices))
    #print(np.var(prices))
    Wprices=[]
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if (row[4] is not None) and (row[2]=="Wed"):
            Wprices.append(row[4]) 
    #print(mean(Wprices))
    #print(var(Wprices))

    return mean(prices),var(prices),np.mean(prices),np.var(prices),mean(Wprices),var(Wprices)




R,a,b,c=A1()
classified=A2()
mmean,mvar,nmean,nvar,wmean,wvar=A3()
print("A1 QUESTION")
print("Rank: ",R)
print("Cost of Candy :",a)
print("Cost of Mango :",b)
print("Cost of Milk  :",c)
print("A2 QUESTION")
print(classified)
print("A3 QUESTION")
print("Mean from user def function: ",mmean)
print("variance from userdef function: ",mvar)
print("Numpy mean: ",nmean)
print("Numpy var",nvar)
print("Wednesday mean: ",wmean)
print("Wednesday var: ",wvar)




#A1()
#A2()
