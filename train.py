import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
import network_dgraph
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix
import os
'''
Main train script to invoke from commandline.
'''

if __name__ == '__main__':
    setting = Setting()
    setting.parse()
    log = open(setting.log_file, 'w')
    # log_string(log, setting)

    print(setting)

    log_string(log, 'log_file: ' + setting.log_file)
    #log_string(log, 'user_file: ' + setting.trans_user_file)
    log_string(log, 'loc_temporal_file: ' + setting.trans_loc_file)
    #log_string(log, 'loc_spatial_file: ' + setting.trans_loc_spatial_file)
    log_string(log, 'interact_file: ' + setting.trans_interact_file)

    log_string(log, str(setting.lambda_user))
    log_string(log, str(setting.lambda_loc))

    log_string(log, 'W in AXW: ' + str(setting.use_weight))
    log_string(log, 'GCN in user: ' + str(setting.use_graph_user))
    log_string(log, 'spatial graph: ' + str(setting.use_spatial_graph))


    def readPdata(spath=None):
        if not spath:
            spath=os.path.join(setting.workpath,'Pdata_gowalla.npy')
        Userdata=np.load(spath,allow_pickle=True)
        return Userdata

    def readCountdata(path=None):
        if not path:
            path=os.path.join(setting.workpath,'Count_gowalla.npy')
        countdata=np.load(path,allow_pickle=True)
        usercount=countdata[0]
        locscount=countdata[1]
        return usercount,locscount


    Userdata=readPdata()  # user time time_slots coords locs
    usercount,locscount=readCountdata()

    poi_loader = PoiDataloader(
        setting.max_users, setting.min_checkins,setting.work_length,Userdata,usercount,locscount,None)  # 0， 5*20+1
    # poi_loader.read(setting.dataset_file)
    # print('Active POI number: ', poi_loader.locations())  # 18737 106994
    # print('Active User number: ', poi_loader.user_count())  # 32510 7768
    # print('Total Checkins number: ', poi_loader.checkins_count())  # 1823598

    log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
    log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
    log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

    dataset = poi_loader.create_dataset(
        setting.sequence_length, setting.batch_size, Split.TRAIN)  # 20, 200 or 1024, 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataset_test = poi_loader.create_dataset(
        setting.sequence_length, setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    assert setting.batch_size < poi_loader.user_count(
    ), 'batch size must be lower than the amount of available users'



    model_pre=network_dgraph.DyGraph(poi_loader.locations(),poi_loader.user_count(),20,setting.hidden_dim)
    model_pre.load_state_dict(torch.load('WORK/dyn_network_60.pth'))
    model_pre.cuda()

    # for name,params in model_pre.named_parameters():
    #     print(name,params.requires_grad)

    trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
                            setting.use_weight, None, setting.use_graph_user,
                            setting.use_spatial_graph, None)  # 0.01, 100 or 1000

    h0_strategy = create_h0_strategy(
        setting.hidden_dim, setting.is_lstm)  # 10 True or False
    trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory,
                    setting.device,model_pre)

    evaluation_test = Evaluation(dataset_test, dataloader_test,
                                poi_loader.user_count(), h0_strategy, trainer, setting, log)
    print('{} {}'.format(trainer, setting.rnn_factory))

    #  training loop filter(lambda p: p.requires_grad, self.kge_model.parameters())
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.parameters(
    # )), lr=setting.learning_rate, weight_decay=setting.weight_decay)

    params_dict_1=[]
    params_dict_2=[]
    for name,params in trainer.model.named_parameters():
        # if 'pre_model' in name:
        #     pass
        #     #params_dict_2.append({'params': params, 'lr': 0.002})
        # else:
        params_dict_1.append({'params': params, 'lr': setting.learning_rate})
    optimizer = torch.optim.Adam(params_dict_1, weight_decay=setting.weight_decay)
    #optimizer2 = torch.optim.Adam(params_dict_2, weight_decay=setting.weight_decay)
    # params_dict = [{'params': trainer.parameters(), 'lr': 0.1},
    #          {'params': model.layer2.parameters(), 'lr': 0.2}]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[20, 40, 60, 80], gamma=0.2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20,30,40,50], gamma=0.2)

    bar = tqdm(total=setting.epochs)
    bar.set_description('Training')

    #evaluation_test.evaluate()

    for e in range(setting.epochs):  # 100
        h = h0_strategy.on_init(setting.batch_size, setting.device)
        dataset.shuffle_users()  # shuffle users before each epoch!
        losses = []
        epoch_start = time.time()
        for i, (x, t, t_slot, s, y, reset_h, active_users) in enumerate(dataloader):
            # reset hidden states for newly added users
            for j, reset in enumerate(reset_h):
                if reset:
                    h[:, j] = h0_strategy.on_reset(active_users[0][j])

            x = x.squeeze().to(setting.device)
            t = t.squeeze().to(setting.device)
            t_slot = t_slot.squeeze().to(setting.device)
            s = s.squeeze().to(setting.device)

            y = y.squeeze().to(setting.device)
            active_users = active_users.to(setting.device)

            #optimizer2.zero_grad()
            forward_start = time.time()
            loss,hp = trainer.loss(x, t, t_slot, s, y, h, active_users)

            loss2=trainer.model.Loss_l2()*1.5e-6
            lossx=loss+loss2
            start = time.time()
            optimizer.zero_grad()
            lossx.backward()
            # torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5)
            end = time.time()
            # print('反向传播需要{}s'.format(end - start))
            losses.append(loss.item())
            optimizer.step()
            #optimizer2.step()

        # schedule learning rate:
        scheduler.step()
        bar.update(1)
        epoch_end = time.time()
        log_string(log, 'One training need {:.2f}s'.format(
            epoch_end - epoch_start))
        # statistics:
        if (e + 1) % 1 == 0:
            epoch_loss = np.mean(losses)
            # print(f'Epoch: {e + 1}/{setting.epochs}')
            # print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
            # print(f'Avg Loss: {epoch_loss}')
            log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
            log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
            log_string(log, f'Avg Loss: {epoch_loss}')

        if (e+1)%5==0 and (e+1)>=35 and (e+1)!=30:
            log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
            # print(f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
            evl_start = time.time()
            evaluation_test.evaluate()
            evl_end = time.time()
            # print('评估需要{:.2f}'.format(evl_end - evl_start))
            log_string(log, 'One evaluate need {:.2f}s'.format(
                evl_end - evl_start))

    bar.close()
