import os
from io import StringIO
from itertools import combinations, chain
import numpy as np
file=open(r'D:\试验\polblog6\polblog.txt')
matrix=np.zeros((1400,1400))
while True:
    line=file.readline()
    if line:
        lines=line.split(' ')
        i=int(lines[0])
        j=int(lines[1])
        matrix[i-1][j-1]=1
        matrix[j-1][i-1]=1
    else:
        break
degree=matrix.sum(axis=1)
#print (degree)
'''
file=open(r'D:\OCD\DBLP\FCM\20_cluster.txt')
overlap=np.zeros(matrix.shape[0])
while True:
   line=file.readline()
   if line:
    lines=line.split(' ')
    lines=lines[0:len(lines)]
    for i in lines:
        overlap[int(i)-1]=overlap[int(i)-1]+1
   else:
      break
file.close()
'''
file=open(r'D:\试验\polblog6\mfcluster.txt')
degrees=matrix.sum()
summod=0
while True:
  line=file.readline()

  if line:
    lines=line[0:len(line)-1]
    lines=line.split(' ')
    lines=lines[0:len(lines)]
    for k in range(len(lines)):
        for l  in range (len(lines)):
          if k>=l:
              continue
          else:
             i=lines[k]
             j=lines[l]
             ki=degree[int(i)-1]
             kj=degree[int(j)-1]

             summod=summod+(matrix[int(i)-1][int(j)-1]-(ki*kj/degrees))


  else:
      break
summod=summod*2
summod=summod/matrix.sum()
print (summod)