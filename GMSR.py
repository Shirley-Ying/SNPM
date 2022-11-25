import torch
import numpy as np
from torch import nn, Tensor

class GMSRModel(nn.Module):
    def __init__(self,k,h,input_size):
        super(GMSRModel, self).__init__()

        self.hidden_size = h
        self.pre_v = k

        self.W = nn.Parameter(torch.randn(k*h,k*h), requires_grad=True)
        self.B = nn.Parameter(torch.randn(1,k*h), requires_grad=True)
        self.R = nn.Parameter(torch.randn(1,k*h//2), requires_grad=True)

        self.FC_W=nn.Parameter(torch.randn(input_size,h), requires_grad=True)
        self.FC_B=nn.Parameter(torch.randn(1,h), requires_grad=True)
        self.FC_W2=nn.Parameter(torch.randn(input_size,h), requires_grad=True)
        self.FC_B2=nn.Parameter(torch.randn(1,h), requires_grad=True)


    #def step(self,preH,hiden_states):
    def forward(self, inputs: Tensor, hidden_states): # hidden states  k x batch x hidden_size
        # seq x batch x input_size     input_size x h
        h_return=[]
        seq_len=inputs.shape[0]
        batch=inputs.shape[1]
        hidden_size=inputs.shape[2]

        inputs_emb=torch.matmul(inputs,self.FC_W)+self.FC_B
        inputs_emb=torch.nn.functional.relu(inputs_emb)
        inputs_emb=torch.matmul(inputs_emb,self.FC_W2)+self.FC_B2
        #inputs_emb=torch.tanh(inputs_emb)

        preH = torch.concat([hidden_states[i] for i in range(self.pre_v)],dim=-1)  # batch x (3Xhidden_size)
        preH = torch.tanh(preH)
        for i in range(seq_len):
            cosR=torch.cos(self.R).reshape(1,self.pre_v,-1)
            sinR=torch.sin(self.R).reshape(1,self.pre_v,-1)
            cos_sin_R=torch.concat((cosR,sinR),dim=-1).reshape(1,-1)
            preH = preH*cos_sin_R
            preH_attention=torch.matmul(preH,self.W)+self.B
            preH = torch.reshape(preH,(batch,self.pre_v,-1))  # 分成 三份
            preH_attention=torch.reshape(preH_attention,(batch,self.pre_v,-1))
            attention=torch.nn.functional.softmax(torch.abs(preH_attention/2.),dim=1)
            preH_input=(preH*attention).sum(1)

            inputs_input=inputs_emb[i]/8
            h_output=inputs_input+preH_input
            h_output=torch.tanh(h_output)
            h_return.append(h_output)
            preH=torch.concat((preH[:,1:,:].view(batch,-1),h_output),dim=-1)

        H=torch.concat(h_return[-self.pre_v:],dim=-1).reshape(batch,self.pre_v,-1).swapaxes(0,1)
        h_return=torch.stack(h_return)
        return h_return,H





#     def forward(self, inputs: Tensor, supports: Tensor, hidden_states) -> Tuple[Tensor, Tensor]:
#         bs, k, n, d = hidden_states.size()
#         _, _, f = inputs.size()
#         preH = hidden_states[:, -1:]
#         for i in range(1, self.pre_v):
#             preH = torch.cat([preH, hidden_states[:, -(i + 1):-i]], -1)
#         preH = preH.reshape(bs, n, d * self.pre_v)
#         convInput = F.leaky_relu_(self.evolution(torch.cat([inputs, preH], -1), supports))
#         new_states = hidden_states + self.R.unsqueeze(0)
#         output = torch.matmul(convInput, self.W) + self.b.unsqueeze(0) + self.attention(new_states)
#         output = output.reshape(bs, 1, n, d)
#         x = hidden_states[:, 1:k]
#         hidden_states = torch.cat([x, output], dim=1)
#         output = output.reshape(bs, n, d)
#         return output, hidden_states

#     def attention(self, inputs: Tensor):
#         bs, k, n, d = inputs.size()
#         x = inputs.reshape(bs, k, -1)
#         out = self.attlinear(x)
#         weight = F.softmax(out, dim=1)
#         outputs = (x * weight).sum(dim=1).reshape(bs, n, d)
#         return outputs

# def __init__(self, input_dim: int, hidden_size: int, pre_k: int, pre_v:int, num_nodes: int, n_supports: int,
#                  k_hop: int, e_layer: int, n_dim: int):
#         super(MSDRCell, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_nodes = num_nodes
#         self.pre_v = pre_v
#         self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size), requires_grad=True)
#         self.b = nn.Parameter(torch.zeros(num_nodes, hidden_size), requires_grad=True)
#         self.R = nn.Parameter(torch.zeros(pre_k, num_nodes, hidden_size), requires_grad=True)
#         self.attlinear = nn.Linear(num_nodes * hidden_size, 1)
#         self.evolution = EvolutionCell(input_dim + hidden_size * pre_v, hidden_size, num_nodes, n_supports, k_hop,
#                                        e_layer, n_dim)