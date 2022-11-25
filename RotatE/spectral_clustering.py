from matplotlib.cbook import simple_linear_interpolation
import numpy as np
import torch
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.linalg import eigs

def laplaEigen(dataMat,k,t):
    m=dataMat.shape[0]
    n=dataMat.shape[1]
    dataMat=dataMat
    dataMat_T=dataMat.T
    # W=mat(zeros([m,m]))
    # D=mat(zeros([m,m]))
    neighbor=[]
    values=[]

    starttime=time.time()
    for i in range(0,m,100):
        diffMats=knn(dataMat[i:i+100],dataMat_T,k)
        for line in diffMats:
            sortedDistIndicies = np.argpartition(line, -100)[-100:]
            neighbor.extend(list(sortedDistIndicies))
            values.extend(list(line[sortedDistIndicies]))
        if (i)%10000==0:
            print(time.time()-starttime)
            starttime=time.time()
    values=np.array(values)
    neighbor=np.array(neighbor)
    save_array=np.concatenate([values[:,np.newaxis],neighbor[:,np.newaxis]],axis=-1)

    # np.save("WORK/sim_values_neighbor_gowalla.npy",save_array,allow_pickle=True)
    # save_array=np.load("WORK/sim_values_neighbor_gowalla.npy",allow_pickle=True)
    value_array=save_array[:,0].reshape(-1,100)
    line_array=save_array[:,1].reshape(-1,100)

    dict_value_line={}
    for i,line in enumerate(line_array):
        for ind,j in enumerate(line):

            if dict_value_line.__contains__((i,int(j))): # 如果 这个东西在 没有赋值之前就有这个值
                if dict_value_line[(i,int(j))]<value_array[i][ind]:
                    dict_value_line[(i,int(j))]=value_array[i][ind]
                    dict_value_line[(int(j),i)]=value_array[i][ind]

            else: # 在赋值之前没有这个值
                dict_value_line[(i,int(j))]=value_array[i][ind]
                dict_value_line[(int(j),i)]=value_array[i][ind]

        if i%10000==0:
            print(time.time())
    # np.save("WORK/dict_value_line_gowalla.npy",dict_value_line,allow_pickle=True)
    # dict_value_line=np.load("WORK/dict_value_line_gowalla.npy",allow_pickle=True).item()
    dict_key=dict_value_line.keys()
    dict_value=dict_value_line.values()
    dict_key_array=np.array(list(dict_key),dtype=np.int32)
    dict_value=np.array(list(dict_value),dtype=np.float32)
    row=[]
    col=[]
    value=[]
    for i,line in enumerate(dict_key_array):
        row.append(line[0])
        col.append(line[1])
        value.append(dict_value[i])
        if i%1000000==0:
            print(time.time())
    row=np.array(row)
    col=np.array(col)
    value_array=np.array(value)
    coo_m=coo_matrix((value_array, (row, col)),shape=(m, m)).tocsr()
    # sparse.save_npz(os.path.join('WORK/sim_array_sp_cluster_gowalla'),coo_m)

    # sim_array_sp_cluster=sparse.load_npz('WORK/sim_array_sp_cluster_gowalla.npz')
    sim_array_sp_cluster=coo_m
    print("dadhwaida")

    row=[]
    values=[]
    values_inv=[]
    for i,line in enumerate(sim_array_sp_cluster):
        values.append(line.sum())
        values_inv.append(1./line.sum())
        row.append(i)
        if i%10000==0:
            print(time.time())
    D_coo=coo_matrix((values, (row, row)),shape=(m, m)).tocsr()
    D_coo_inv=coo_matrix((values_inv, (row, row)),shape=(m, m)).tocsr()
    L_coo=(D_coo-sim_array_sp_cluster).tocoo().tocsr()

    # np.save("WORK/L_coo_1_gowalla.npy",L_coo,allow_pickle=True)
    # np.save("WORK/values_inv_gowalla.npy",np.array(values_inv),allow_pickle=True)

    # L_coo=np.load("WORK/L_coo_1_gowalla.npy",allow_pickle=True).item()
    # values_inv=np.load("WORK/values_inv_gowalla.npy",allow_pickle=True)

    values_inv=np.array(values_inv)
    for i,line in enumerate(L_coo):
        L_coo[i]=line.dot(values_inv[i])
        if i%2000==0:
            print(time.time())

    # np.save("WORK/D_coo.npy",D_coo,allow_pickle=True)
    # np.save("WORK/D_coo_inv.npy",D_coo_inv,allow_pickle=True)
    L_coo=L_coo.tocoo().tocsr()
    # np.save("WORK/L_coo_gowalla.npy",L_coo,allow_pickle=True)
    # L_coo=np.load("WORK/L_coo_gowalla.npy",allow_pickle=True).item()
    vals_p, vecs_p=eigs(L_coo, k=25,which='SM')
    # np.save("WORK/vals_gowalla",vals,allow_pickle=True)
    # np.save("WORK/vecs_gowalla",vecs,allow_pickle=True)

    # vals=np.load("WORK/vals_gowalla.npy",allow_pickle=True)
    # vecs=np.load("WORK/vecs_gowalla.npy",allow_pickle=True)
    # vals_p=np.load("WORK/vals.npy",allow_pickle=True)
    # vecs_p=np.load("WORK/vecs.npy",allow_pickle=True)

    vals=np.real(vals_p)
    vecs=np.real(vecs_p)
    index=np.argsort(np.abs(vals_p))
    vals = vals[index]

    j = 0
    while (vals[j]) < 1e-5:
        j+=1

    print("j: ", j)

    index = index[j:j+20]
    val_use = vals[j:j+20].copy()
    vecs_use = vecs[:,index].copy()
    # i=0

    # val_use=vals[i:i+20].copy()
    # vecs_use=vecs[:,i:i+20].copy()

    import faiss
    ncentroids = 200
    niter = 250
    verbose = True
    d = vecs_use.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(vecs_use)

    centroids=kmeans.centroids

    list_centroids=[[] for i in range(ncentroids)]
    list_number=[[] for i in range(ncentroids)]
    D,I = kmeans.index.search(vecs_use, 1)
    for i,index in enumerate(I.reshape(-1)):
        list_number[index].append(i)
        list_centroids[index].append(vecs_use[i])
    for i,list_c in enumerate(list_centroids):
        list_centroids[i]=np.stack(list_c)
        list_number[i]=np.array(list_number[i])

    min=1024
    max=0
    for line in list_centroids:
        min=min if min<len(line) else len(line)
        max=max if max>len(line) else len(line)

    print(min,max)
    # 通过这样的方式 我们可以构建出如下的内容
    # 首先 我们有了 loc_nums 个的 embedding表示 维度为106994x20
    # 其次 我们将其聚类，共分为200个类

    np.save("WORK/list_centroids",list_centroids,allow_pickle=True)
    np.save("WORK/vecs_use",vecs_use,allow_pickle=True)
    np.save("WORK/I",I,allow_pickle=True)
    np.save("WORK/list_number",list_number,allow_pickle=True)





    # print(i)
    # print(113234144241244414414142)
    # print(time.time())
    #     for j in range(k):
    #         sqDiffVector = dataMat[i,:]-dataMat[k_index[j],:]
    #         sqDiffVector=array(sqDiffVector)**2
    #         sqDistances = sqDiffVector.sum()
    #         W[i,k_index[j]]=math.exp(-sqDistances/t)
    #         D[i,i]+=W[i,k_index[j]]
    # L=D-W
    # Dinv=np.linalg.inv(D)
    # X=np.dot(D.I,L)
    # lamda,f=np.linalg.eig(X)
    # return lamda,f

def knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    diffMat = inX.dot(dataSet)
    diffMats=diffMat.toarray().squeeze()
    return diffMats

import time
import os
def normalize(graph):
    graph=graph.tolil()
    for i,line in enumerate(graph):
        graph[i]=(line.power(2)/line.power(2).sum()).sqrt()
        if i % 10000==0:
            print(time.time())
    return graph

def load_npz_data():

    loc2loc_npz=sparse.load_npz("WORK/coo_gowalla_neighbors.npz")

    i_114514=sp.identity(loc2loc_npz.shape[0], format='coo')
    graph=loc2loc_npz+i_114514*0.01

    graph=normalize(graph).tocsr()
    sparse.save_npz(os.path.join('WORK/graph.npz'),graph)

    graph=sparse.load_npz("WORK/graph.npz")
    X_ndim = laplaEigen(graph, 100, t = 20)

load_npz_data()


# if __name__ == '__main__':
#     X, Y = make_swiss_roll(n_samples = 2000)
#     X_ndim = le(X, n_neighbors = 5, t = 20)

#     fig = plt.figure(figsize=(12,6))
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c = Y)

#     ax2 = fig.add_subplot(122)
#     ax2.scatter(X_ndim[:, 0], X_ndim[:, 1], c = Y)
#     plt.show()

#     X = load_digits().data
#     y = load_digits().target

#     dist = cal_pairwise_dist(X)
#     max_dist = np.max(dist)
#     print("max_dist", max_dist)
#     X_ndim = le(X, n_neighbors = 20, t = max_dist*0.1)
#     plt.scatter(X_ndim[:, 0], X_ndim[:, 1], c = y)
#     plt.savefig("LE2.png")
#     plt.show()
#     print("what?")
