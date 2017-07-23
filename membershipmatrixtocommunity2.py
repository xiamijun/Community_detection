import numpy as np
import copy
from collections import defaultdict

def read_randommat(filename,delimiter=None,nodetype=str):
    adj = {}
    k=0
    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        sum=0.0
        for l in L:
            sum=sum+float(l)
        adj[k]=sum
        k=k+1
    return adj
#根据最大隶属度划分得到的社区
def read_hmat(hmat):
    comms=defaultdict(set)
    hmatmatrix=np.zeros((len(hmat[0]),len(hmat)))
    for k,v in hmat.items():
        for i in range(len(v)):
             hmatmatrix[i][k]=v[i]
    for i in range(len(hmat[0])):
        max=0.0
        maxindex=0
        for j in range(len(hmat)):
             temp=float(hmatmatrix[i][j])

             if(temp>max):
                 max=hmatmatrix[i][j]
                 maxindex=j
        comms[maxindex].add(i+1)
    print(comms)
    return comms

if __name__=='__main__':
 adj=read_randommat(r'D:\试验\polblog74\randommat.txt', delimiter=' ')
 comms=[]
 hmat=defaultdict(list)
 commnum=0
 for line in open(r'D:\试验\polblog74\hmat.txt', 'U'):
    L=line[0:len(line)-1].split(' ')
    comm={}
    for i in range(len(L)):
       hmat[commnum].append(L[i])
       comm[i]=L[i];
    comms.append((sorted(comm.items(),key=lambda asd:asd[1],reverse=True)))
    commnum=commnum+1
 k=0
 newcomms=read_hmat(hmat)
 file=open(r'D:\试验\polblog74\mfcluster.txt','w')
 #从mfclust中读取数据根据社区隶属度进行社区划分
 for c in comms:
     edge=0.0
     sum=0.0
     for id in c:
         sum=sum+float(id[1])
     sum=sum*sum
     degree=0
     comm=[]
     temp=0;
     for id in c:
          degree=degree+adj[id[0]]
          if(abs((degree-sum))<abs((temp-sum))):
              comm.append(id[0]+1);
          else: break;
          temp=degree;
     #print(comm)
     '''
     for k,newcomm in newcomms.items():
              if(len(newcomm&set(comm))/len(newcomm)>0.2):
                         comm=newcomm|set(comm)
                         comm=list(comm)
                         for i in range(len(comm)-1):
                                         file.write(str(comm[i])+' ')
                         file.write(str(comm[len(comm)-1])+'\n')
     '''
     for i in range(len(comm)-1):
                            file.write(str(comm[i])+' ')
     file.write(str(comm[len(comm)-1])+'\n')




