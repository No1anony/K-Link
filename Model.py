import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict
from Model_Base import *

from clip import clip




class K_Link_pre(nn.Module):
    def __init__(self,args, device):
        super(K_Link_pre, self).__init__()

        Conv_out = args.conv_out
        lstmhidden_dim = args.lstmhidden_dim
        lstmout_dim = args.lstmout_dim
        conv_kernel = args.conv_kernel
        hidden_dim = args.hidden_dim
        time_length = args.conv_time_CNN
        num_node = args.num_sensor
        num_windows = args.num_windows
        moving_window = args.moving_window
        stride = args.stride
        decay = args.decay
        pooling_choice = args.pool_choice
        n_class = 1
        self.window_length = args.num_windows
        self.device = device

        ### Prompt side
        self.mapping1 = nn.Linear(512,int(hidden_dim))
        self.mapping2 = nn.Linear(512,int(hidden_dim))
        self.mapping_sensor = nn.Linear(int(hidden_dim),int(hidden_dim))


        ### Data side
        self.nonlin_map = Feature_extractor_1DCNN_pre(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )
        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)
        self.MPNN1 = GraphConvpoolMPNN_block(2*hidden_dim, hidden_dim, num_node, time_length,
                                                time_window_size=moving_window, stride=stride, decay = decay,
                                                pool_choice=pooling_choice)
        if args.data_sub == 2:
            layer_config = [2,2,1]
        elif args.data_sub == 4:
            layer_config = [1,1,1,1,1,1]
        output_layers= []

        for layer_i in range(len(layer_config)):
            if layer_i == 0:
                output_layers.append(nn.Linear(hidden_dim * num_windows * num_node, hidden_dim*layer_config[layer_i]))
                output_layers.append(nn.ReLU(inplace=True))
            else:
                output_layers.append(nn.Linear(hidden_dim * layer_config[layer_i-1], hidden_dim * layer_config[layer_i]))
                output_layers.append(nn.ReLU(inplace=True))
        output_layers.append(nn.Linear(hidden_dim * layer_config[-1], n_class))
        self.fc = nn.ModuleList(output_layers)

        ## Alignment side
        self.alignment1 = contrastive_alignment_sensor
        self.alignment2 = MSE_alignment
        self.alignment3 = contrastive_alignment_label

    def forward(self, X, label_prompt_features=None,sensor_feature_prompt = None,training=True):
        bs, tlen, num_node, dimension = X.size()
        ### Prompt side


        ### Data side
        ### Graph Generation
        A_input = tr.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = tr.reshape(A_input_, [bs, tlen,num_node,-1])


        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = tr.transpose(X_,1,2)
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)
        A_input_ = X_


        MPNN_output1, Adj = self.MPNN1(A_input_)

        if training:
            ###
            label_prompt_embedding = tr.reshape(label_prompt_features, [bs, self.window_length * num_node, -1])
            label_prompt_embedding = self.mapping1(label_prompt_embedding)

            sensor_text_embedding = tr.cat([sensor_feature_prompt.unsqueeze(0)] * bs, 0)
            sensor_text_embedding = tr.reshape(sensor_text_embedding, [bs, self.window_length * num_node, -1])
            sensor_text_embedding = self.mapping2(sensor_text_embedding)

            sensor_text_embedding_ = tr.cat((label_prompt_embedding, sensor_text_embedding), -1)

            edge_prompt = Dot_Graph_Construction(sensor_text_embedding_)
            edge_prompt = tr.reshape(edge_prompt, [bs, self.window_length * num_node, self.window_length * num_node])
            MPNN_output = tr.reshape(MPNN_output1, [bs,self.window_length*num_node,-1])
            MPNN_output = self.mapping_sensor(MPNN_output)

            Adj =Dot_Graph_Construction(MPNN_output)
            Adj = tr.reshape(Adj, [bs,self.window_length*num_node,self.window_length*num_node])


            sensor_feature_loss = self.alignment1(MPNN_output,sensor_text_embedding,self.device)
            sensor_edge_loss = self.alignment2(Adj,edge_prompt,self.device)
            label_feature_loss = self.alignment3(MPNN_output,label_prompt_embedding,self.device)
        else:
            sensor_feature_loss = 0
            sensor_edge_loss = 0
            label_feature_loss = 0

        features1 = tr.reshape(MPNN_output1, [bs, -1])
        x=  features1
        for net in self.fc:
            x = net(x)

        features = x

        return features, sensor_feature_loss, sensor_edge_loss, label_feature_loss




class K_Link_cls(nn.Module):
    def __init__(self,args, device,text_dim=512):
        super().__init__()
        # graph_construction_type = args.graph_construction_type

        Conv_out = args.conv_out
        lstmhidden_dim = args.lstmhidden_dim
        lstmout_dim = args.lstmout_dim
        conv_kernel = args.conv_kernel
        hidden_dim = args.hidden_dim
        time_length = args.conv_time_CNN
        num_node = args.num_sensor
        num_windows = args.num_windows
        moving_window = args.moving_window
        stride = args.stride
        decay = args.decay
        pooling_choice = args.pool_choice
        n_class = args.n_class
        self.window_length = args.num_windows
        self.device = device
        self.temperature = args.temperature

        ### Prompt side
        self.mapping1 = nn.Linear(text_dim,int(hidden_dim))
        self.mapping2 = nn.Linear(text_dim,int(hidden_dim))

        self.mapping_sensor = nn.Linear(int(hidden_dim),int(hidden_dim))


        ### Data side
        self.nonlin_map = Feature_extractor_1DCNN_cls(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block(2*hidden_dim, hidden_dim, num_node, time_length,
                                             time_window_size=moving_window, stride=stride, decay = decay,
                                             pool_choice=pooling_choice)
        if args.data_name == 'HAR':
            self.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_dim * num_windows * num_node, n_class))
            ]))
        elif args.data_name == 'SSC':
            self.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2 * hidden_dim)),
                ('relu1', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(2 * hidden_dim, 2 * hidden_dim)),
                ('relu2', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(2 * hidden_dim, hidden_dim)),
                ('relu3', nn.ReLU(inplace=True)),
                ('fc4', nn.Linear(hidden_dim, n_class)),

            ]))
        self.alignment1 = contrastive_alignment_sensor
        self.alignment2 = MSE_alignment
        self.alignment3 = contrastive_alignment_label

    def forward(self, X, label_prompt_features=None,sensor_feature_prompt = None,training=True):
        # print(X.size())
        bs, tlen, num_node, dimension = X.size() ### tlen = 1

        ### Data side
        ### Graph Generation
        A_input = tr.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = tr.reshape(A_input_, [bs, tlen,num_node,-1])


        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = tr.transpose(X_,1,2)
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)
        A_input_ = X_

        ## positional encoding before mapping ending

        MPNN_output1, Adj = self.MPNN1(A_input_)

        if training:
            # print('winodw length should be', MPNN_output1.size(1))
            label_prompt_embedding = tr.reshape(label_prompt_features,[bs,self.window_length*num_node,-1])
            label_prompt_embedding = self.mapping1(label_prompt_embedding)

            sensor_text_embedding = tr.cat([sensor_feature_prompt.unsqueeze(0)]*bs,0)
            sensor_text_embedding = tr.reshape(sensor_text_embedding,[bs,self.window_length*num_node,-1])
            sensor_text_embedding = self.mapping2(sensor_text_embedding)
            sensor_text_embedding_ = tr.cat((label_prompt_embedding,sensor_text_embedding),-1)

            edge_prompt =Dot_Graph_Construction(sensor_text_embedding_)
            edge_prompt = tr.reshape(edge_prompt, [bs,self.window_length*num_node,self.window_length*num_node])


            MPNN_output = tr.reshape(MPNN_output1, [bs,self.window_length*num_node,-1])
            MPNN_output = self.mapping_sensor(MPNN_output)
            Adj =Dot_Graph_Construction(MPNN_output)
            Adj = tr.reshape(Adj, [bs,self.window_length*num_node,self.window_length*num_node])

            sensor_feature_loss = self.alignment1(MPNN_output,sensor_text_embedding,self.device, temperature=self.temperature)
            sensor_edge_loss = self.alignment2(Adj,edge_prompt,self.device, temperature=self.temperature)
            label_feature_loss = self.alignment3(MPNN_output,label_prompt_embedding,self.device, temperature=self.temperature)

        else:
            sensor_feature_loss = 0
            sensor_edge_loss = 0
            label_feature_loss = 0

        features1 = tr.reshape(MPNN_output1, [bs, -1])


        features = self.fc(features1)

        return features, sensor_feature_loss, sensor_edge_loss, label_feature_loss

