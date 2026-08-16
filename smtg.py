
def dist(v1,v2):
    ans=0   #[4,5] [5,6]
    for i in range(len(v1)):
        ans+=(v1[i]-v2[i])**2
    return ans**0.5

ans=dist([0,0],[5,2])
print(ans)