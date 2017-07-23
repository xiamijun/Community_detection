import random
import numpy as np
import copy
from collections import defaultdict
from itertools import combinations, chain
import operator
import networkx as nx


def swap(a,b):
    if a > b:
        return b,a
    return a,b

def matrix_factorization(R,P,Q,K,steps=50,alpha=0.0002,beta=0.02):
  Q=Q.T
  length=len(R)
  for step in range(steps):
        #print(step)
        for i in range(length):

                    #RQ=np.dot(R,Q.T)
                    #WH=np.dot(np.dot(P,Q),Q.T)
                    for k in range(K):
                        print(k)
                        x=np.dot(R[i,:],Q[k,:])
                        y=np.dot(np.dot(P[i,:],Q),Q[k,:])
                        #P[i][k]=P[i,k]*(RQ[i,k]/WH[i,k])
                        P[i][k]=P[i][k]*(x/y)
                        Q[k][i]=P[i][k]
        '''
        e = 0
        for i in range(length):
            for j in range(length):
                #if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
        '''
  return P, Q.T


def normalize(A):
    column_sums = A.sum(axis=0)
    new_matrix = A / column_sums[np.newaxis, :]
    return new_matrix

def inflate(A, inflate_factor):
    return normalize(np.power(A, inflate_factor))

def expand(A, expand_factor):
    return np.linalg.matrix_power(A, expand_factor)

def add_diag(A, mult_factor):
    return A + mult_factor * np.identity(A.shape[0])


def mcl(M, expand_factor = 2, inflate_factor = 2, max_loop = 2 , mult_factor = 1):
    M = add_diag(M, mult_factor)
    M = normalize(M)

    for i in range(max_loop):

        M = inflate(M, inflate_factor)
        M = expand(M, expand_factor)
    return (M+M.T)

def read_edgelist_unweighted(filename,delimiter=None,nodetype=str):
    adj = defaultdict(set) # node to set of neighbors
    nodes=set()
    edges = set()
    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        ni,nj = nodetype(L[0]),nodetype(L[1]) # other columns ignored
        nodes.add(ni)
        nodes.add(nj)
        if ni != nj: # skip any self-loops...
            edges.add( swap(ni,nj) )
            adj[ni].add(nj)
            adj[nj].add(ni) # since undirected
    return (dict(adj),nodes,edges)

#使用PageRank计算节点value值
class PageRank:
    def __init__(self, graph, directed):
        self.graph = graph
        self.V = len(self.graph)
        self.d = 0.85
        self.directed = directed
        self.ranks = dict()

    def rank(self):
        for key, node in self.graph.nodes(data=True):
            if self.directed:
                self.ranks[key] = 1/float(self.V)
            else:
                self.ranks[key] = node.get('rank')

        for _ in range(10):
            for key, node in self.graph.nodes(data=True):
                rank_sum = 0
                curr_rank = node.get('rank')
                if self.directed:
                    neighbors = self.graph.out_edges(key)
                    for n in neighbors:
                        outlinks = len(self.graph.out_edges(n[1]))
                        if outlinks > 0:
                            rank_sum += (1 / float(outlinks)) * self.ranks[n[1]]
                else:
                    neighbors = self.graph[key]
                    for n in neighbors:
                        if self.ranks[n] is not None:
                            outlinks = len(self.graph.neighbors(n))
                            rank_sum += (1 / float(outlinks)) * self.ranks[n]

                # actual page rank compution
                self.ranks[key] = ((1 - float(self.d)) * (1/float(self.V))) + self.d*rank_sum

        return p

#对于节点排名
def rank(graph, node):
    #V
    nodes = graph.nodes()
    #|V|
    nodes_sz = len(nodes)
    #I
    neighbs = graph.neighbors(node)
    #d
    rand_jmp = random.uniform(0, 1)

    ranks = []
    ranks.append( (1/nodes_sz) )

    for n in nodes:
        rank = (1-rand_jmp) * (1/nodes_sz)
        trank = 0
        for nei in neighbs:
            trank += (1/len(neighbs)) * ranks[len(ranks)-1]
        rank = rank + (d * trank)
        ranks.append(rank)

#对于无向网络计算
def parse(filename, isDirected):
    f=open(filename, 'r')
    data=[]
    for line in f:
        line=line[0:len(line)-1]
        data.append(line)
    print ("Reading and parsing the data into memory...")
    if isDirected:
        return parse_directed(data)
    else:
        return parse_undirected(data)

#对于有向网络计算
def parse_undirected(data):
    G = nx.Graph()
    nodes=set()
    edges=[]
    for data in data:
        data=data.split(' ')
        nodes.add(data[0])
        edges.append((data[0],data[1]))
    print(nodes)
    print(edges)
    num_nodes = len(nodes)
    rank = 1/float(num_nodes)
    G.add_nodes_from(nodes, rank=rank)
    G.add_edges_from(edges)

    return G

def parse_directed(data):
    DG = nx.DiGraph()
    for data in  data:
        data=data.split(' ')
        node_a = data[0]
        node_b = data[1]
        val_a = 1
        val_b = 0

        DG.add_edge(node_a, node_b)
        if val_a >= val_b:
            DG.add_path([node_a, node_b])
        else:
            DG.add_path([node_b, node_a])

    return DG

def superadjmatrix(adj,edges):
    sedge=np.zeros((len(adj),len(adj)))
    for index in range(sedge.shape[0]):
        sedge[index,index]=0
    for edge in edges:
        i=int(edge[0])
        j=int(edge[1])
        sedge[int(i-1),int(j-1)]=1
        sedge[int(j-1),int(i-1)]=1
    return sedge

def write(cluster,filename):
    file=open(filename,'w')
    for c in cluster:
      if(len(c)>1):
        for i in range(len(c)-1):
            file.write(str(c[i])+' ')
        file.write(str(c[len(c)-1])+'\n')

def similarities(adj,nodes,influencevalue):
    sedge=np.zeros((len(nodes),len(nodes)))
    for index in range(sedge.shape[0]):
        sedge[index,index]=0
    i_adj = dict( (n,adj[n] | set([n])) for n in adj)
    for i,j in combinations(nodes,2):
                inc_ns_i,inc_ns_j = i_adj[i],i_adj[j]
                common=inc_ns_i&inc_ns_j
                commonsum=0.0
                for node in common:
                    commonsum=commonsum+influencevalue[node]
                sumi=0.0
                sumj=0.0
                for node in inc_ns_i:
                    sumi=sumi+influencevalue[node]
                for node in inc_ns_j:
                    sumj=sumj+influencevalue[node]
                S = 1.0 * commonsum / np.sqrt(sumi*sumj)
                sedge[int(i)-1,int(j)-1]=S
                sedge[int(j)-1,int(i)-1]=S
    return sedge

#属性相似度
def attrisimilarities(nodes,vector):
    sedge=np.zeros((len(nodes),len(nodes)))
    for index in range(sedge.shape[0]):
        sedge[index,index]=0
    for i,j in combinations(nodes,2):
            if(i!=j):
                vector_i=vector[i]
                vector_j=vector[j]
                S = len(vector_i&vector_j) / np.sqrt(len(vector_i)*len(vector_j))
                sedge[int(i)-1,int(j)-1]=S
                sedge[int(j)-1,int(i)-1]=S
    return sedge


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


def matrixtocommunity():
 adj=read_randommat(r'D:\试验\polblog74\randommat.txt', delimiter=' ')
 comms=[]
 for line in open(r'D:\试验\polblog74\hmat.txt', 'U'):
    L=line[0:len(line)-1].split(' ')
    comm={}
    for i in range(len(L)):
       comm[i]=L[i];
    comms.append((sorted(comm.items(),key=lambda asd:asd[1],reverse=True)))
 k=0
 file=open(r'D:\试验\polblog74\cluster.txt','w')
 for c in comms:
     edge=0.0
     sum=0.0
     for id in c:
         sum=sum+float(id[1])
     sum=sum*sum
     print(sum)
     degree=0
     comm=[]
     temp=0;
     for id in c:
          degree=degree+adj[id[0]]
          if(abs((degree-sum))<abs((temp-sum))):
              comm.append(id[0]+1);
          else: break;
          temp=degree;
     for i in range(len(comm)-1):
         file.write(str(comm[i])+' ')
     file.write(str(comm[len(comm)-1])+'\n')
     #file.write('\n')
     print(comm)
     k=k+1

def getModularity():
      file=open(r'D:\试验\polblog74\pol1.txt')
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

      file=open(r'D:\试验\polblog74\cluster.txt')
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
      k=0
      j=0
      for i in overlap:
            print (str(j)+' '+str(i))
            if i==0:
               k=k+1
            j=j+1
      print(k)
      file=open(r'D:\试验\polblog74\cluster.txt')
      degrees=matrix.sum()
      summod=0
      while True:
          line=file.readline()
          if line:
             lines=line[0:len(line)-1]
             lines=lines.split(' ')
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
                                    index=1/(overlap[int(i)-1]*overlap[int(j)-1])
                                    summod=summod+index*(matrix[int(i)-1][int(j)-1]-(ki*kj/degrees))
          else:
              break
      summod=summod*2
      summod=summod/matrix.sum()
      print (summod)
      return (summod)

if __name__=='__main__':
   isDirected = True
   networkfile=r'D:\试验\polblog74\pol1.txt'
   #attrifile=open(r'D:\试验\pol1_attri.txt')
   #communityfile='D:\OCD\exp1_cluster.txt'
   graph = parse(networkfile, isDirected)
   p = PageRank(graph, isDirected)
   p.rank()
   influencevalue={}
   sorted_r = sorted(p.ranks.items(), key=operator.itemgetter(1), reverse=True)
   for tup in sorted_r:
            #print(str(tup[0])+' '+str(tup[1]))
            influencevalue[str(tup[0])]=tup[1]
   adj,nodes,edges=read_edgelist_unweighted(networkfile, delimiter=' ')
   sedge=similarities(adj,nodes,influencevalue)
   '''
   vector = defaultdict(set)
   while True:
      line=attrifile.readline()
      if line:
         line=line[0:len(line)-1].split(' ')
         for i in range(1,len(line)):
                vector[line[0]].add(line[i])
      else:
          break
   attrisedge=attrisimilarities(nodes,vector)
   '''

   a=0.6
   b=0.009
   #ncrsedge=a*sedge+(1-a)*attrisedge
   ncrsedge=sedge
   dimension=len(adj)
   numIter=6
   adjedge=superadjmatrix(adj,edges)
   R=mcl(ncrsedge)+0.01*adjedge

   N=len(adj)
   M=len(adj)
   K=10
   P=np.random.rand(N,K)
   Q=np.random.rand(M,K)
   nP,nQ=matrix_factorization(R,P,Q,K)
   file=open(r'D:\试验\polblog74\hmat.txt','w')
   for i in range(nQ.T.shape[0]):
               for j in range(nQ.T.shape[1]-1):
                    file.write(str(round(nQ.T[i][j],3))+' ')
               file.write(str(round(nQ.T[i][nQ.T.shape[1]-1],3)))
               file.write('\n')
   file.close()
   file=open(r'D:\试验\polblog74\randommat.txt','w')
   for i in range(R.shape[0]):
                for j in range(R.shape[1]-1):
                     file.write(str(round(R[i,j],3))+' ')
                file.write(str(round(R[i,R.shape[1]-1],3))+'\n')
   file.close()
