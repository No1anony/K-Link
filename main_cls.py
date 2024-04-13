import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import Model
from clip import clip
import matplotlib.pyplot as plt
import random
from Model_Base import *
from torch.utils.tensorboard import SummaryWriter

from data_loader_HAR import data_generator as data_generator_HAR
from data_loader_ISRUC import data_generator as data_generator_SSC

from utils_ import sensor_names_HAR, sensor_names_SSC

class Train():
    def __init__(self, args):

        self.device = 'cuda:0'
        self.args = args
        self.clip_model, _ = clip.load("ViT-B/32")
        self.clip_model.eval().to(self.device)

        if args.data_name == 'HAR':
            self.train, self.valid, self.test = data_generator_HAR('./HAR/', args=args)
            self.sensor_prompt = sensor_names_HAR(args.index_sensor)
            self.sensor_feature_prompt = sensor_prompt_HAR(self.clip_model,args.num_windows,self.sensor_prompt, self.device)
        elif args.data_name == 'SSC':
            self.train, self.valid, self.test = data_generator_SSC('./ISRUC/', args)
            self.sensor_prompt = sensor_names_SSC(args.index_sensor)
            self.sensor_feature_prompt = sensor_prompt_SSC(self.clip_model,args.num_windows,self.sensor_prompt, self.device)

        self.net = Model.K_Link_cls(args,self.device)

        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.loss_function = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        loss_mse_item = 0
        loss_sensor_item = 0
        loss_edge_item = 0

        c1 = self.args.sensor_coeff
        c2 = self.args.edge_coeff
        c3 = self.args.label_coeff

        for data, label in self.train:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            if args.data_name == 'HAR':
                prompt = label_prompt_HAR(self.clip_model, args.num_windows, self.sensor_prompt,
                                                        label, self.device)
            elif args.data_name == 'SSC':
                prompt = label_prompt_SSC(self.clip_model, args.num_windows, self.sensor_prompt,
                                                        label, self.device)
            self.optim.zero_grad()

            if self.args.prompt:
                prediction,sensor_feature_loss, sensor_edge_loss, label_feature_loss = self.net(data,prompt,
                                                                            self.sensor_feature_prompt)

                loss_mse = self.loss_function(prediction, label)
                loss = loss_mse+c1*sensor_feature_loss+ c2*sensor_edge_loss + c3*label_feature_loss

                loss_mse_item = loss_mse_item + loss_mse.item()
                loss_sensor_item = loss_sensor_item + sensor_feature_loss.item()
                loss_edge_item = loss_edge_item + sensor_edge_loss.item()

            else:
                prediction = self.net(data)
                loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()

        return loss_, loss_mse_item, loss_sensor_item, loss_edge_item

    def Train_model(self):
        epoch = self.args.epoch
        best_accu = 0
        test_accu_ = []
        prediction_ = []
        real_ = []

        for i in range(epoch):
            loss = self.Train_batch()
            if i % self.args.show_interval == 0:

                test_accu, prediction, real = self.Prediction()
                if test_accu > best_accu:
                    best_accu = test_accu
                print('In the {}th epoch, TESTING accuracy is {}%'.format(i, np.round(best_accu, 3)))
                test_accu_.append(test_accu)
                prediction_.append(prediction)
                real_.append(real)

        np.save('./experiment/{}.npy'.format(self.args.save_name), [test_accu_, prediction_, real_])

    def cuda_(self, x):
        x = tr.Tensor(x)

        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x


    def data_preprocess_transpose(self, data, ops):
        '''

        :param data: size is [bs, time_length, dimension, Num_nodes]
        :return: size is [bs, time_length, Num_nodes, dimension]
        '''

        data = tr.transpose(data,2,3)
        ops = tr.transpose(ops,2,3)

        return data, ops

    def Cross_validation(self):
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.valid:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_,0)
        real_ = tr.cat(real_,0)

        prediction_ = tr.argmax(prediction_,-1)
        # print(prediction_)
        # print(real_)

        accu = self.accu_(prediction_, real_)
        # print(accu)
        return accu

    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.test:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction,_,_,_ = self.net(data, training = False)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_, 0)
        real_ = tr.cat(real_, 0)

        prediction_ = tr.argmax(prediction_, -1)
        accu = self.accu_(prediction_, real_)
        return accu, prediction_, real_


    def accu_(self, predicted, real):
        num = predicted.size(0)
        real_num = 0
        for i in range(num):
            if predicted[i] == real[i]:
                real_num+=1
        return 100*real_num/num




    def visualization(self, prediction, real):
        fig = plt.figure()
        sub = fig.add_subplot(1, 1, 1)

        sub.plot(prediction, color = 'red', label = 'Predicted Labels')
        sub.plot(real, 'black', label = 'Real Labels')
        sub.legend()
        plt.show()


if __name__ == '__main__':
    from args import args

    seed_cus = 0
    random.seed(seed_cus)
    np.random.seed(seed_cus)
    torch.manual_seed(seed_cus)
    torch.cuda.manual_seed(seed_cus)
    torch.cuda.manual_seed_all(seed_cus)
    args = args()

    def args_config_cls(args, data_name):
        args.data_name = data_name
        args.epoch = 81
        args.k = 1
        args.decay = 0.7
        args.pool_choice = 'mean'
        args.moving_window = 2
        args.stride = 2
        args.conv_time_CNN = 6

        args.batch_size = 100
        args.temperature = 0.005

        args.index_sensor = False
        args.prompt = True
        if data_name == 'HAR':
            args.window_sample = 128

            args.sensor_coeff = 0.001
            args.edge_coeff = 0.01
            args.label_coeff = 0.01
            args.lr = 9e-4
            args.conv_kernel = 6
            args.patch_size = 64
            args.time_denpen_len = int(args.window_sample / args.patch_size)
            args.conv_out = 20
            args.num_windows = 1
            args.num_sensor = 9
            args.n_class = 6
            args.lstmout_dim = 18
            args.hidden_dim = 36
            args.lstmhidden_dim = 64

        elif data_name == 'SSC':
            args.window_sample = 300

            args.sensor_coeff = 0.001
            args.edge_coeff = 0.001
            args.label_coeff = 0.01
            args.lr = 1e-3
            args.conv_kernel = 5
            args.patch_size = 75
            args.time_denpen_len = int(args.window_sample / args.patch_size)
            args.conv_out = 24
            args.num_windows = 2
            args.dropout = 0.4
            args.num_sensor = 10
            args.n_class = 5
            args.lstmout_dim = 64
            args.lstmhidden_dim = 64
            args.hidden_dim = 72



        return args

    datasub = 'HAR'
    args = args_config_cls(args, datasub)


    argsDict = args.__dict__
    args_save_name = 'HAR'
    with open(f'./experiment/{args_save_name}_args.txt', 'w') as f:
        f.writelines('------------------ start saving args------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    args.save_name = args_save_name
    train = Train(args)
    train.Train_model()
