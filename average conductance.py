from collections import defaultdict
def swap(a,b):
    if a > b:
        return b,a
    return a,b

def read_edgelist_unweighted(filename,delimiter=None,nodetype=str):
    adj = defaultdict(set)
    edges=set();
    nodes=set()
    i=1
    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        ni,nj = nodetype(L[0]),nodetype(L[1]) # other columns ignored
        nodes.add(ni)
        nodes.add(nj)
        edges.add(swap(ni,nj))
        adj[ni].add(nj)
        adj[nj].add(ni)
    return (dict(adj),edges,nodes)



if __name__=='__main__':
    adj,edges,nodes=read_edgelist_unweighted(r'D:\试验\polblog6\polblog.txt', delimiter=' ')
    file=open(r'D:\试验\polblog6\mfcluster.txt')
    AC=0
    sumnode=0
    while True:
        line=file.readline()
        if line:
            lines=line[0:len(line)-1]
            lines=lines.split(' ')
            num=len(lines)
            sumnode=sumnode+num
            cs=0
            vol=0
            vols=0
            for edge in edges:
                if(edge[0] in lines and edge[1] not in lines):
                    cs=cs+1
                if(edge[0] not in lines and edge[1] in lines):
                    cs=cs+1
            for v in lines:
                vol=vol+len(adj[v])
            nodevs=nodes-set(lines)
            for v in nodevs:
                vols=vols+len(adj[v])
            v=min(vol,vols)
            AC=AC+num*(cs/v)
        else:
            break
    AC=AC/(2*sumnode)
    file.close()
print(AC)