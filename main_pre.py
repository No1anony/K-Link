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
from data_loader_RUL import CMPDataIter_graph
import matplotlib.pyplot as plt
import random
from Model_Base import *
from torch.utils.tensorboard import SummaryWriter
from utils_ import sensor_names_RUL

class Train():
    def __init__(self, args):


        data_iter = CMPDataIter_graph('./',
                                data_set='FD00{}'.format(str(args.data_sub)),
                                max_rul=args.max_rul,
                                seq_len=args.patch_size,
                                time_denpen_len=args.time_denpen_len,
                                window_sample=args.window_sample,
                                net_name=1)

        self.device = 'cuda:0'

        self.sensor_prompt = sensor_names_RUL(args.index_sensor)

        self.args = args
        self.train_data = self.cuda_(data_iter.out_x)
        self.train_ops = self.cuda_(data_iter.out_ops)
        self.train_label = self.cuda_(data_iter.out_y)

        self.val_data = self.cuda_(data_iter.cross_val_x)
        self.val_ops = self.cuda_(data_iter.cross_val_ops)
        self.val_label = self.cuda_(data_iter.cross_val_y)

        self.test_data = self.cuda_(data_iter.test_x)
        self.test_ops = self.cuda_(data_iter.test_ops)
        self.test_label = self.cuda_(data_iter.test_y)

        self.train_data, self.train_ops = self.data_preprocess_transpose(self.train_data, self.train_ops)
        self.test_data, self.test_ops = self.data_preprocess_transpose(self.test_data, self.test_ops)
        self.val_data, self.val_ops = self.data_preprocess_transpose(self.val_data, self.val_ops)

        self.clip_model, _ = clip.load("ViT-B/32")
        self.clip_model.eval().to(self.device)
        self.sensor_feature_prompt = sensor_prompt_RUL(self.clip_model,args.num_windows,self.sensor_prompt, self.device)


        self.net = Model.K_Link_pre(args,self.device)

        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.loss_function = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        loss_mse_item = 0
        loss_sensor_item = 0
        loss_edge_item = 0
        batch_size = self.args.batch_size
        iter = int(self.train_data.size(0) / batch_size)
        remain = self.train_data.size(0) - iter * batch_size

        # c1 = 0.01

        c1 = self.args.sensor_coeff
        c2 = self.args.edge_coeff
        c3 = self.args.label_coeff

        for i in range(iter):

            data = self.train_data[i * batch_size:(i + 1) * batch_size]
            label = self.train_label[i * batch_size:(i + 1) * batch_size]
            prompt = label_prompt_RUL(self.clip_model, args.num_windows, self.sensor_prompt,
                                                    args.max_rul * label, self.device)
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

        if remain != 0:

            data = self.train_data[-remain:]
            label = self.train_label[-remain:]
            prompt = label_prompt_RUL(self.clip_model, args.num_windows, self.sensor_prompt,
                                                    args.max_rul * label, self.device)

            self.optim.zero_grad()
            if self.args.prompt:
                prediction, sensor_feature_loss, sensor_edge_loss, RUL_feature_loss = self.net(data,prompt,
                                                                            self.sensor_feature_prompt)

                loss_mse = self.loss_function(prediction, label)
                loss = loss_mse + c1 * sensor_feature_loss + c2 * sensor_edge_loss + c3 * RUL_feature_loss
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
        best_RMSE = np.inf
        best_Score = np.inf

        test_RMSE = []
        test_score = []
        RUL_predicted = []
        RUL_real = []


        for i in range(epoch):
            _ = self.Train_batch()
            print(f'training {self.args.save_name} in the {i} epoch')
            if i%self.args.show_interval == 0:


                test_RMSE_, test_score_, test_result_predicted, test_result_real = self.Prediction()
                if test_RMSE_ < best_RMSE:
                    best_RMSE= test_RMSE_
                if test_score_[0] < best_Score:
                    best_Score= test_score_[0]
                print('In the {}th epoch, TESTING RMSE is {}, TESTING Score is {}'.format(i, best_RMSE, best_Score))
                test_RMSE.append(test_RMSE_)
                test_score.append(test_score_[0])
                RUL_predicted.append(test_result_predicted)
                RUL_real.append(test_result_real)


        test_RMSE = np.stack(test_RMSE,0)
        test_score = np.stack(test_score,0)

        test_results = np.stack([test_RMSE, test_score],0)
        np.save('./experiment/{}.npy'.format(self.args.save_name),test_results)

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
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.val_data.size(0) / batch_size)
        remain = self.val_data.size(0) - iter * batch_size
        for i in range(iter):
            data = self.val_data[i * batch_size:(i + 1) * batch_size]
            label = self.val_label[i * batch_size:(i + 1) * batch_size]

            if self.args.prompt:
                prediction,sensor_feature_loss, sensor_edge_loss = self.net(data, training=False)
                loss_mse = self.loss_function(prediction, label)
                loss = loss_mse+sensor_feature_loss+ sensor_edge_loss
            else:
                prediction = self.net(data)
                loss = self.loss_function(prediction, label)

            loss_ = loss_ + loss.item()

        if remain != 0:
            data = self.val_data[-remain:]
            label = self.val_label[-remain:]
            if self.args.prompt:
                prediction,sensor_feature_loss, sensor_edge_loss = self.net(data, training=False)
                loss_mse = self.loss_function(prediction, label)
                loss = loss_mse+sensor_feature_loss+ sensor_edge_loss
            else:
                prediction = self.net(data)
                loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()
        # print(loss_)
        return loss_

    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        if self.args.prompt:
            prediction,_,_,_ = self.net(self.test_data, training=False)
        else:
            prediction = self.net(self.test_data)


        predicted_RUL = prediction
        real_RUL = self.test_label
        MSE = self.loss_function(predicted_RUL, real_RUL)

        RMSE = tr.sqrt(MSE)*self.args.max_rul
        score = self.scoring_function(predicted_RUL, real_RUL)
        return RMSE.detach().cpu().numpy(),\
               score.detach().cpu().numpy(), \
               predicted_RUL.detach().cpu().numpy(), \
               real_RUL.detach().cpu().numpy()

    def Prediction_training(self):
        self.net.eval()
        sample_idx = random.sample(range(len(self.train_data)), 100)
        train_data_sample = self.train_data[sample_idx]
        train_label_sample = self.train_label[sample_idx]
        prediction = self.net(train_data_sample)
        MSE = self.loss_function(prediction, train_label_sample)
        RMSE = tr.sqrt(MSE) * self.args.max_rul
        score = self.scoring_function(prediction, train_label_sample)
        return RMSE.detach().cpu().numpy(), \
               score.detach().cpu().numpy(), \
               prediction.detach().cpu().numpy(), \
               train_label_sample.detach().cpu().numpy()




    def visualization(self, prediction, real):
        fig = plt.figure()
        sub = fig.add_subplot(1, 1, 1)

        sub.plot(prediction, color = 'red', label = 'Predicted Labels')
        sub.plot(real, 'black', label = 'Real Labels')
        sub.legend()
        plt.show()

    def scoring_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):

            if real[i] > predicted[i]:
                score = score+ (tr.exp((real[i]*self.args.max_rul-predicted[i]*self.args.max_rul)/13)-1)

            elif real[i]<= predicted[i]:
                score = score + (tr.exp((predicted[i]*self.args.max_rul-real[i]*self.args.max_rul)/10)-1)

        return score


if __name__ == '__main__':
    from args import args

    seed_cus = 0
    random.seed(seed_cus)
    np.random.seed(seed_cus)
    torch.manual_seed(seed_cus)
    torch.cuda.manual_seed(seed_cus)
    torch.cuda.manual_seed_all(seed_cus)

    args = args()


    def args_config(data_sub, args):

        args.epoch = 51
        args.k = 1
        args.conv_kernel = 2
        args.moving_window = 2
        args.stride = 2
        args.pool_choice = 'mean'
        args.decay = 0.7
        args.sensor_coeff = 0.01
        args.edge_coeff = 0.1
        args.label_coeff = 0.1
        args.batch_size = 300
        args.temperature = 0.0005
        args.index_sensor = False
        args.prompt = True
        if data_sub == 2:
            args.data_sub = 2
            args.patch_size = 5
            args.time_denpen_len = 10
            args.conv_out = 7
            args.num_windows = 5
            args.lstmout_dim = 48
            args.hidden_dim = 56
            args.window_sample = 50
            args.conv_time_CNN = 10
            args.lstmhidden_dim = 64
            args.lr = 4e-3

        if data_sub == 4:
            args.data_sub = 4
            args.patch_size = 5
            args.time_denpen_len = 10
            args.conv_out = 7
            args.num_windows = 5
            args.lstmout_dim = 16
            args.hidden_dim = 36
            args.window_sample = 50
            args.conv_time_CNN = 10
            args.lstmhidden_dim = 56
            args.lr = 2e-3

        return args

    datasub = 2
    args = args_config(datasub, args)


    argsDict = args.__dict__
    args_save_name = 'FD002'
    with open(f'./experiment/{args_save_name}_args.txt', 'w') as f:
        f.writelines('------------------ start saving args------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    args.save_name = args_save_name

    train = Train(args)
    train.Train_model()

