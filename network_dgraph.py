import torch
import torch.nn as nn
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import faiss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DyGraph(nn.Module):
    def __init__(self, input_size, user_count, hidden_size,hidden_dim):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = 20
        self.hidden_dim=hidden_dim
        self.list_centroids=list(np.load("WORK/list_centroids.npy",allow_pickle=True))  #每个聚类中含有的元素(存的不是序号，是array)
        num_centroids=len(self.list_centroids)
        self.vecs_use=torch.tensor(np.load("WORK/vecs_use.npy",allow_pickle=True),dtype=torch.float32).cuda()  #初始向量
        vecs_emblength=self.vecs_use.shape[1]
        self.I_array=np.load("WORK/I.npy",allow_pickle=True)  # 每个向量对应的聚类
        self.list_number=np.load("WORK/list_number.npy",allow_pickle=True)
        self.index_list=[]

        for i in range(num_centroids):
            index_i=faiss.IndexFlatL2(vecs_emblength)
            index_i.add(self.list_centroids[i])
            self.index_list.append(index_i)

        self.seq_embedding_layer1=nn.Linear(20*5,20)
        self.seq_embedding_layer2=nn.Linear(20,10)

        self.vec_embedding_layer1=nn.Linear(20,20)
        self.vec_embedding_layer2=nn.Linear(20,10)

        # self.score_layer_1=nn.Linear(20,20)
        # self.score_layer_2=nn.Linear(20,1)

    def Loss_l2(self):
        base_params = dict(self.named_parameters())
        loss_l2=0.
        count=0
        for key, value in base_params.items():
            if 'bias' not in key and 'pre_model' not in key:
                loss_l2+=torch.sum(value**2)
                count+=value.nelement()
        return loss_l2

    def prepare_train(self):
        #input_tensorlist=torch.tensor(np.array([i for i in range(self.input_size)]),dtype=int).cuda()
        x_embedding_dy=self.vec_embedding_layer1(self.vecs_use)
        x_embedding_dy=F.relu(x_embedding_dy)
        self.x_embedding_dy=self.vec_embedding_layer2(x_embedding_dy)


    def emb_with_neighbor_output(self,x,x_embedding_network):

        seq_len, user_len = x.size()
        x_vecs = []
        loc_vecs_use=self.vecs_use

        x_view=torch.reshape(x,(-1,))
        x_emb=torch.index_select(loc_vecs_use,0,x_view)
        x_emb=torch.reshape(x_emb,(seq_len,user_len,-1))  #x.view(-1).cpu().numpy()

        # x_emb_history=x_emb
        x_emb_history=x_emb
        x_emb_history=x_emb_history.reshape(-1,20)

        x_emb_history_now=x_emb.reshape(-1,20)
        x_emb_history_neg_1=torch.cat((x_emb[0:1],x_emb[0:seq_len-1]),dim=0).reshape(-1,20)
        x_emb_history_neg_2=torch.cat((x_emb[0:2],x_emb[0:seq_len-2]),dim=0).reshape(-1,20)
        x_emb_history_neg_3=torch.cat((x_emb[0:3],x_emb[0:seq_len-3]),dim=0).reshape(-1,20)
        x_emb_history_neg_4=torch.cat((x_emb[0:4],x_emb[0:seq_len-4]),dim=0).reshape(-1,20)

        x_emb_history_concat=torch.cat((x_emb_history_neg_4,x_emb_history_neg_3,x_emb_history_neg_2,x_emb_history_neg_1,x_emb_history_now),dim=-1)

        x_emb_history_concat=self.seq_embedding_layer1(x_emb_history_concat)
        x_emb_history_concat=F.relu(x_emb_history_concat)
        x_emb_history_concat=self.seq_embedding_layer2(x_emb_history_concat)

        index_x_centroids=self.I_array[x.view(-1).cpu().numpy()].squeeze()  #获取每个向量所在的聚类

        # 要注意的是 x_emb_history 是 加权均值
        # 而 index_x_centroids 则是 每个元素的 聚类编号
        # index_list 中存放的 是 用于找topk nei 的 faiss类型参数 其个数等于 聚类个数
        # list_centroids 是一个存放多个 array的 列表 其中存放的就是
        # I_array 存放的也是每个元素所在的聚类编号
        # self.vecs_use 指的是所有的 向量初始值

        list_return=[]
        indexofcentroids=[[]for i in range(len(self.list_number))]

        time_start=time.time()

        x_emb_history=self.vec_embedding_layer1(x_emb_history)
        x_emb_history=F.relu(x_emb_history)
        x_emb_history=self.vec_embedding_layer2(x_emb_history)

        for i in range(x_emb_history.shape[0]):
            indexofcentroids[index_x_centroids[i]].append(i)

        return_x_emb=torch.zeros((x_emb_history.shape[0],self.hidden_dim)).cuda()

        for i,line in enumerate(indexofcentroids):
            # 如果聚类没有元素 直接跳过
            if len(line)!=0:
                list_nei=self.list_number[i] # 位于这个聚类内的所有元素
                candidate_number=torch.tensor(list_nei,dtype=int).cuda()
                now_self=x_view[line]

                line_cuda=torch.tensor(np.array(line),dtype=int).cuda()
                now_self_emb=x_emb_history_concat[line] # n x 10
                neighbor_candidate_emb=torch.index_select(self.x_embedding_dy,0,candidate_number)  #neigh x 10

                now_self_emb=now_self_emb.unsqueeze(1).expand(-1,neighbor_candidate_emb.shape[0],-1)
                neighbor_candidate_emb=neighbor_candidate_emb.unsqueeze(0).expand(now_self_emb.shape[0],-1,-1)

                distance=torch.norm(now_self_emb-neighbor_candidate_emb,p=2,dim=-1)

                # indices=np.random.choice(len(list_nei),(len(line),min(10,distance.shape[-1])),replace=True)
                score=torch.exp(-distance*0.02)
                values,indices=torch.topk(score,k=min(10,score.shape[-1]),dim=-1,largest=True)
                indices=torch.reshape(indices,(-1,))

                index_embedding_neigh=torch.index_select(candidate_number,0,indices)

                embedding_neigh=x_embedding_network[index_embedding_neigh].reshape(values.shape[0],values.shape[1],x_embedding_network.shape[-1])

                self_embedding=x_embedding_network[now_self].unsqueeze(1)
                embedding_neigh=torch.concat((embedding_neigh,self_embedding),dim=1)
                # values=torch.zeros(values.shape).cuda()
                one_value=torch.tensor(np.array([1.]*len(line)).reshape(-1,1),dtype=torch.float32).cuda()
                values=torch.concat((values,one_value),dim=-1)

                score_new=torch.nn.functional.softmax(values,dim=-1)
                embedding_return=embedding_neigh*score_new.unsqueeze(-1)
                embedding_return=embedding_return.sum(1)
                return_x_emb[line,:]=embedding_return
                #print(1)
        time_end=time.time()
        #print((time_end-time_start))
        #return_x_emb=(x_embedding_network[x_view]+return_x_emb*1.0)/2.0
        return return_x_emb

    def forward(self, x,y):
        seq_len, user_len = x.size()
        loc_vecs_use=self.vecs_use  # 特征向量的poi表示

        x_vecs = []
        x_view=torch.reshape(x,(-1,))
        y_view=torch.reshape(y,(-1,))

        x_emb=torch.index_select(loc_vecs_use,0,x_view)
        x_emb=torch.reshape(x_emb,(seq_len,user_len,-1))  #x.view(-1).cpu().numpy()

        y_emb=torch.index_select(loc_vecs_use,0,y_view)
        y_emb=torch.reshape(y_emb,(seq_len,user_len,-1))


        index_x_centroids=self.I_array[x.view(-1).cpu().numpy()].squeeze()  #获取每个向量所在的聚类
        time_start=time.time()


        x_emb_history=x_emb
        x_emb_history=x_emb_history.reshape(-1,20)

        x_emb_history_now=x_emb.reshape(-1,20)
        x_emb_history_neg_1=torch.cat((x_emb[0:1],x_emb[0:seq_len-1]),dim=0).reshape(-1,20)
        x_emb_history_neg_2=torch.cat((x_emb[0:2],x_emb[0:seq_len-2]),dim=0).reshape(-1,20)
        x_emb_history_neg_3=torch.cat((x_emb[0:3],x_emb[0:seq_len-3]),dim=0).reshape(-1,20)
        x_emb_history_neg_4=torch.cat((x_emb[0:4],x_emb[0:seq_len-4]),dim=0).reshape(-1,20)

        x_emb_history_concat=torch.cat((x_emb_history_neg_4,x_emb_history_neg_3,x_emb_history_neg_2,x_emb_history_neg_1,x_emb_history_now),dim=-1)


        # 要注意的是 x_emb_history 是 加权均值
        # 而 index_x_centroids 则是 每个元素的 聚类编号
        # index_list 中存放的 是 用于找topk nei 的 faiss类型参数 其个数等于 聚类个数
        # list_centroids 是一个存放多个 array的 列表 其中存放的就是
        # I_array 存放的也是每个元素所在的聚类编号
        # self.vecs_use 指的是所有的 向量初始值

        pos_sample=[]
        neg_sample=[]
        self_sample=[]

        indexofcentroids=[[]for i in range(len(self.list_number))]

        time_start=time.time()

        x_emb_history_concat=self.seq_embedding_layer1(x_emb_history_concat)
        x_emb_history_concat=F.relu(x_emb_history_concat)
        x_emb_history_concat=self.seq_embedding_layer2(x_emb_history_concat)

        vec_output=self.vec_embedding_layer1(self.vecs_use)
        vec_output=F.relu(vec_output)
        vec_output=self.vec_embedding_layer2(vec_output)

        faiss_x_emb_history=x_emb_history.cpu().numpy()

        #self.prepare_train()

        for i in range(x_emb_history.shape[0]):
            indexofcentroids[index_x_centroids[i]].append(i)

        for i,line in enumerate(indexofcentroids):
            if len(line)!=0:
                list_nei=self.list_number[i] # 位于这个聚类内的所有元素
                # candidate_number=torch.tensor(list_nei,dtype=int).cuda()
                # i 既是聚类的编号
                index=self.index_list[i]
                index_len=index.ntotal
                #pos_item=np.random.choice(list_nei,size=(len(line),min(30,index_len)),replace=True)
                # Distance,neighborindex=index.search(faiss_x_emb_history[line],min(index_len,10)) # len(line) x 10
                # neighborindex=neighborindex.reshape(-1)
                #pos_sample.append(self.list_number[i][neighborindex])

                #now_self=x_view[line]
                line_array=np.array(line)
                line_array=np.expand_dims(line_array,axis=1).repeat(min(index_len,10),axis=1)  #扩展多倍
                self_sample.append(line_array.reshape(-1))

                pos_line=np.array(y_view[line].cpu())
                pos_line=np.expand_dims(pos_line,axis=1).repeat(min(index_len,10),axis=1)  #扩展多倍
                pos_sample.append(pos_line.reshape(-1))

                num_neg=np.random.randint(0,self.input_size,size=(min(index_len,10)+2)*len(line))
                index_neg=np.where(self.I_array[num_neg]!=i)[0] # 不在一个聚类里面就是负样本
                now_neg_num=len(index_neg)
                neg_list=[]
                neg_list.extend(num_neg[index_neg])
                while now_neg_num<min(index_len,10)*len(line):
                    num_neg=np.random.randint(0,self.input_size,size=(min(index_len,10)+2)*len(line))
                    index_neg=np.where(self.I_array[num_neg]!=i)[0] # 不在一个聚类里面就是负样本
                    now_neg_num+=len(index_neg)
                    neg_list.extend(num_neg[index_neg])
                neg_list=neg_list[:min(index_len,10)*len(line)]  #这样我们有了10个正样本 10个负样本
                neg_sample.append(np.array(neg_list))


        pos_sample=torch.tensor(np.concatenate(pos_sample),dtype=int).cuda()
        neg_sample=torch.tensor(np.concatenate(neg_sample),dtype=int).cuda()
        self_sample=torch.tensor(np.concatenate(self_sample),dtype=int).cuda()


        pos_output=torch.index_select(vec_output,0,pos_sample)
        neg_output=torch.index_select(vec_output,0,neg_sample)
        self_output=torch.index_select(x_emb_history_concat,0,self_sample)



        # pos_score=self.score_layer_1(torch.cat((self_output,pos_output),dim=-1)).mean()
        # neg_score=self.score_layer_1(torch.cat((self_output,neg_output),dim=-1)).mean()


        pos_score=torch.norm(pos_output-self_output,p=2,dim=-1)
        neg_score=torch.norm(neg_output-self_output,p=2,dim=-1)

        loss_function=nn.LogSigmoid()
        loss=-1.*loss_function(neg_score-pos_score).mean()
        time_end=time.time()
        #print(time_end-time_start)
        return loss


