import numpy as np
import torch

import os
from datetime import datetime
import time
import pytz

from datetime import timezone
from tzwhere import tzwhere
from timezonefinder import TimezoneFinder
class Graph_Embedding:
    def __init__(self,checkin_file_path,Pdata_path,count_path):
        self.checkin_path=checkin_file_path
        self.Pdata_path=Pdata_path
        self.count_path=count_path


    def create_entity_file(self):
        self.user2id={}
        self.poi2id={}
        self.Pdata=np.load(self.Pdata_path,allow_pickle=True)
        self.Countdata=np.load(self.count_path,allow_pickle=True)
        for line in self.Pdata:
            for loc in line[4]:
                if loc not in self.poi2id:
                    self.poi2id[loc]=loc
        poilist=list(self.poi2id.values())
        np.save("WORK/entity_list_gowalla",np.array(poilist,dtype=np.int32),allow_pickle=True)
        print("entity is Done")

    def create_relation_file(self):
        self.relation_dict={"pre_and_sub_and_self":0}
        np.save("WORK/relation_dict_gowalla",self.relation_dict,allow_pickle=True)
        print("relation is Done")

    def create_tuplerelations_file(self):
        relation_pre_and_sub=self.precursor_and_subsequent_relations()  # 1,2
        relation_list=relation_pre_and_sub
        relation_array=np.array(relation_list,dtype=np.int32)

        np.save("WORK/relation_only_pre_and_sub_gowalla",relation_array,allow_pickle=True)

    def precursor_and_subsequent_relations(self):
        precursor_dict={}
        list_relations=[]
        infx=np.load("data/Pdata_gowalla.npy",allow_pickle=True)
        for userdata in infx:
            loclen=int(len(userdata[4])*0.8)
            for i in range(loclen-1):
                start=userdata[4][i]
                end=userdata[4][i+1]
                if precursor_dict.get((int(start),int(end))) is None:
                    precursor_dict[(int(start),int(end))]=0
                if precursor_dict.get((int(end),int(start))) is None:
                    precursor_dict[(int(end),int(start))]=0

        for i in range(self.Countdata[1]):
            if precursor_dict.get((i,i)) is None:
                precursor_dict[(i,i)]=0

        for key in precursor_dict:
            list_relations.append([key[0],self.relation_dict["pre_and_sub_and_self"],key[1]])
        return list_relations




g_e=Graph_Embedding("data/checkins-gowalla.txt","data/Pdata_gowalla.npy","data/Count_gowalla.npy")
g_e.create_entity_file()
g_e.create_relation_file()
g_e.create_tuplerelations_file()
