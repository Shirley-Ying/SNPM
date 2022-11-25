import numpy as np
import torch
from KGEModel import KGEModel
import TrainDataset
from torch.utils.data import DataLoader

from TrainDataset import TrainDataset
from TrainDataset import BidirectionalOneShotIterator
class Graph_Trainer:
    def __init__(self,entity_path,relation_path,relation_tuple_path):

        self.hidden_dim=200
        self.gamma=6
        self.batch_size=4096
        self.max_steps=200000
        self.negative_sample_size=64
        self.learning_rate=0.0002

        self.warm_up_steps = self.max_steps // 2

        self.entity_data=np.load(entity_path,allow_pickle=True)
        self.relation_data=np.load(relation_path,allow_pickle=True).item()
        relation_tuple_data=np.load(relation_tuple_path,allow_pickle=True)



        self.count=None
        self.true_tail=None

        nentity = len(self.entity_data)
        nrelation = len(self.relation_data)

        self.kge_model = KGEModel(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=self.hidden_dim,
            gamma=self.gamma,
            double_entity_embedding=True,
            double_relation_embedding=False,
        ).cuda()


        self.train_dataloader_tail = DataLoader(
            TrainDataset(relation_tuple_data, nentity, nrelation, self.negative_sample_size, 'tail-batch',self.count,self.true_tail),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=5,
            collate_fn=TrainDataset.collate_fn
        )

        self.train_iterator = BidirectionalOneShotIterator(self.train_dataloader_tail)

        # Set training configuration
        current_learning_rate = self.learning_rate
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),  # filter 过滤掉不需要梯度的参数
            lr=current_learning_rate
        )

        init_step=0
        step = init_step
        training_logs = []
        import time
        start=time.time()


        self.count=np.load("WORK/count_dict_gowalla.npy",allow_pickle=True).item()
        self.true_tail=np.load("WORK/true_tail_dict_gowalla.npy",allow_pickle=True).item()
        #self.kge_model=np.load("LogicData/kge_model-loc2loc.npy",allow_pickle=True).item()

        #self.evaluate(flag=False)

        print(time.time()-start)
        for step in range(init_step, self.max_steps):
            loss = self.kge_model.train_step(self.kge_model, self.optimizer, self.train_iterator)
            if (step+1)%100==0:
                print("step:",step+1,"loss:",loss.item())
                print(time.time()-start)
                start=time.time()
            if (step+1)>=1500 and (step+1) % 100==0:
                self.evaluate(flag=False)
            if step+1==2000:
                self.evaluate(flag=True)
                np.save("WORK/kge_model-gowalla",self.kge_model)
                break
            if (step+1)%8000==0 and (step+1)<=20000:
                self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.kge_model.parameters()),  # filter 过滤掉不需要梯度的参数
                lr=self.learning_rate/2)
                self.learning_rate/=2


    def evaluate(self,flag):

        self.poi_len=len(self.entity_data)
        with torch.no_grad():
            self.create_loc2loc_graph(flag)

    def create_loc2loc_graph(self,flag):
        count=0
        all_count=0
        row=[]
        col=[]
        value_array=[]
        for i in range(self.poi_len):

            start=np.array([i]*self.poi_len)
            relation=np.array([self.relation_data['pre_and_sub_and_self']]*self.poi_len,dtype=int)
            end=np.array([i for i in range(self.poi_len)])

            relation_tuple=torch.tensor(np.array([start,relation,end]).transpose(1,0)).cuda()
            score=self.kge_model.evaluate_step(relation_tuple) # score 的索引和 distance_se的索引一致 依然是loc 索引

            real_tail=self.true_tail.get((int(i),self.relation_data['pre_and_sub_and_self']),np.array([]))
            values,indices=torch.topk(score,k=100,dim=0)
            indices=indices.cpu().numpy()
            indices_x=end[indices]

            for ind in indices_x:
                if int(ind) in real_tail:
                    count+=1
            #indices=distance_se[indices] # 获取 loc 索引
            all_count+=min(100,len(real_tail))
            row.extend([i]*indices.shape[0])
            col.extend(list(indices))
            value_array.extend(list(values.cpu().detach().numpy()))

            if (i+1)%3000==0:
                print(count/all_count)
                if not flag:
                    break
        if flag:
            from scipy.sparse import coo_matrix
            from scipy import sparse
            import os
            row=np.array(row)
            col=np.array(col)
            value_array=np.array(value_array)
            coo_m=coo_matrix((value_array, (row, col)),shape=(self.poi_len, self.poi_len))
            sparse.save_npz(os.path.join('WORK/coo_gowalla_neighbors.npz'),coo_m)

    def f_s(self,lng1,lat1,lng2,lat2): # long 是 经度 lat 是维度
        lng1=np.deg2rad(lng1)
        lat1=np.deg2rad(lat1)
        lng2=np.deg2rad(lng2)
        lat2=np.deg2rad(lat2)

        dlon=lng2-lng1
        dlat=lat2-lat1

        a=np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        distance=2*np.arcsin(np.sqrt(a))*6371*1000 # 地球平均半径，6371km
        distance=np.around(distance/1000,decimals=3)
        distance=torch.tensor(distance,dtype=torch.float32)
        return distance

G_T=Graph_Trainer("WORK/entity_list_gowalla.npy","WORK/relation_dict_gowalla.npy","WORK/relation_only_pre_and_sub_gowalla.npy")
