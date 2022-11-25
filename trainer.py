import time

import torch
import torch.nn as nn
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, use_graph_user, use_spatial_graph, interact_graph):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        # self.spatial_graph = spatial_graph
        # self.friend_graph = friend_graph
        self.interact_graph = interact_graph

    def __str__(self):
        return 'Use flashback training.'

    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device,model_pre):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay

        # exp decay  2个functions
        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))

        # def f_s(lng1,lat1,lng2,lat2): #return torch.exp(-(delta_s * self.lambda_s))

        #     lng1=torch.deg2rad(lng1)
        #     lat1=torch.deg2rad(lat1)
        #     lng2=torch.deg2rad(lng2)
        #     lat2=torch.deg2rad(lat2)

        #     dlon=lng2-lng1
        #     dlat=lat2-lat1
        #     a=torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        #     distance=2*torch.asin(torch.sqrt(a))*6371*1000 # 地球平均半径，6371km
        #     distance=torch.round(distance/1000,decimals=3)
        #     return torch.exp(-(distance * self.lambda_s))


        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, hidden_size, f_t, f_s, gru_factory, self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph,model_pre).to(device)

    def evaluate(self, x, t, t_slot, s,y, h, active_users):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        """

        self.model.eval()
        # (seq_len, user_len, loc_count)
        out, h = self.model(x, t, t_slot, s,y, h, active_users)

        # seq_len, user_len, loc_count = out.shape
        # out = out.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)
        # out = out.t()  # (loc_count, seq_len * batch_size)
        # graph = sparse_matrix_to_tensor(self.graph).to(x.device)  # (loc_count, loc_count)
        # graph = graph.t()
        # out = torch.sparse.mm(graph, out)  # (loc_count, seq_len * batch_size)
        # out = out.t()  # (seq_len * batch_size, loc_count)
        # out = torch.reshape(out, (seq_len, user_len, loc_count))

        out_t = out.transpose(0, 1)
        return out_t, h  # model outputs logits

    def loss(self, x, t, t_slot, s, y, h, active_users):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model.train()
        out, h = self.model(x, t, t_slot, s, y, h, active_users)
        out = out.view(-1, self.loc_count)

        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)

        return l,h.detach()
