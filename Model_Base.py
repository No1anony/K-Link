import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch
_tokenizer = _Tokenizer()


class Feature_extractor_1DCNN_pre(nn.Module):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size = 8, stride = 1, dropout = 0):
        super(Feature_extractor_1DCNN_pre, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )


    def forward(self, x_in):
        ### input dim is (bs, tlen, feature_dim)
        x = tr.transpose(x_in, -1,-2)

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x


class Feature_extractor_1DCNN_cls(nn.Module):
    def __init__(self, input_channels, num_hidden,embedding_dimension, kernel_size = 3, stride = 1, dropout = 0):
        super(Feature_extractor_1DCNN_cls, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
        )


    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = tr.transpose(x_in, -1,-2)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        return x
def Dot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = tr.transpose(node_features, 1, 2)

    Adj = tr.bmm(node_features, node_features_1)

    eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
    eyes_like_inf = eyes_like*1e8
    Adj = F.leaky_relu(Adj-eyes_like_inf)
    Adj = F.softmax(Adj, dim = -1)
    # print(Adj[0])
    Adj = Adj+eyes_like
    # print(Adj[0])
    # if prior:


    return Adj

class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)

        Adj = tr.bmm(node_features, node_features_1)

        eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj

class Dot_Graph_Construction_weights_woi(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)

        Adj = tr.bmm(node_features, node_features_1)

        # eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        # eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        # Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj
def softmax_eye(edges, device):
    ### input size is (bs, num_nodes, num_nodes)
    bs, N, N = edges.size()


    eyes_like = tr.eye(N).repeat(bs, 1, 1).to(device)
    eyes_like_inf = eyes_like * 1e8
    Adj = F.leaky_relu(edges - eyes_like_inf)
    Adj = F.softmax(Adj, dim=-1)
    Adj = Adj + eyes_like


    return Adj


class Dot_Graph_Construction_weights_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)

        Adj = tr.bmm(node_features, node_features_1)

        eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj



def iDot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    ##

    node_features_1 = tr.transpose(node_features, 1, 2)

    Adj = tr.bmm(node_features, node_features_1)

    eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
    eyes_like_inf = eyes_like*1e8
    Adj = F.leaky_relu(Adj-eyes_like_inf)
    Adj = F.softmax(Adj, dim = -1)
    Adj = Adj+eyes_like

    return Adj



class MPNN_mk(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = tr.bmm(A_,A)
            out_k = self.theta[kk](tr.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        return F.leaky_relu(GCN_output_)


class MPNN_mk_v2(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(outpuut_dinmension)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = tr.bmm(A_,A)
            out_k = self.theta[kk](tr.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = tr.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = tr.transpose(GCN_output_, -1, -2)

        return F.leaky_relu(GCN_output_)

def Graph_regularization_loss(X, Adj, gamma):
    ### X size is (bs, N, dimension)
    ### Adj size is (bs, N, N)
    X_0 = X.unsqueeze(-3)
    X_1 = X.unsqueeze(-2)

    X_distance = tr.sum((X_0 - X_1)**2, -1)

    Loss_GL_0 = X_distance*Adj
    Loss_GL_0 = tr.mean(Loss_GL_0)

    Loss_GL_1 = tr.sqrt(tr.mean(Adj**2))
    # print('Loss GL 0 is {}'.format(Loss_GL_0))
    # print('Loss GL 1 is {}'.format(Loss_GL_1))


    Loss_GL = Loss_GL_0 + gamma*Loss_GL_1


    return Loss_GL



class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = tr.zeros(max_len, d_model).cuda()
        position = tr.arange(0, max_len).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2) *
                             -(math.log(100.0) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)
        pe[:, 1::2] = tr.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x = x + torch.Tensor(self.pe[:, :x.size(1)],
        #                  requires_grad=False)
        # print(self.pe[0, :x.size(1),2:5])
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x

def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = tr.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride)

    y_ = tr.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = tr.transpose(y_, 1,-1)

    return y_

def Conv_GraphST_pad(input, time_window_size, stride, padding):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = tr.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride, padding=[0,padding])

    y_ = tr.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = tr.transpose(y_, 1,-1)

    return y_
def Mask_Matrix(num_node, time_length, decay_rate):
    Adj = tr.ones(num_node * time_length, num_node * time_length).cuda()
    for i in range(time_length):
        v = 0
        for r_i in range(i,time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
        v=0
        for r_i in range(i+1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i-r_i) * num_node
            idx_e_col = (i-r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1

    return Adj




class GraphConvpoolMPNN_block(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)

        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors,time_window_size,decay)

        self.pool_choice = pool_choice
    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = tr.transpose(input_con, 2,3)
        input_con_ = tr.reshape(input_con_, [bs*num_windows, time_window_size*num_sensors, feature_dim])

        A_input_ = self.graph_construction(input_con_)
        A_input = A_input_*self.pre_relation


        input_con_ = tr.transpose(input_con_, -1, -2)
        input_con_ = self.BN(input_con_)
        input_con_ = tr.transpose(input_con_, -1, -2)
        X_output = self.MPNN(input_con_, A_input)


        X_output = tr.reshape(X_output, [bs, num_windows, time_window_size,num_sensors, self.output_dim])

        X_output = tr.mean(X_output, 2)

        return X_output,A_input_





def sensor_prompt_RUL(model, time_length, sensors_name, device):
    token_STGraph = []
    for timestamp in range(time_length):
        for sensor_name in sensors_name:
            prompt_ = f"When predicting the remaining useful life of an aircraft turbine engine, a sensor of {sensor_name} in the {(timestamp)*time_length} timestamp"
            token_STGraph.append(prompt_)
            # print(prompt_)

    text_tokens = clip.tokenize(token_STGraph).to(device)

    with tr.no_grad():
        text_features = model.encode_text(text_tokens).float()
    return tr.reshape(text_features,[time_length,len(sensors_name), -1])


def sensor_prompt_HAR(model, time_length, sensors_name, device):
    token_STGraph = []
    for timestamp in range(time_length):
        for sensor_name in sensors_name:
            prompt_ = f"When classifying human activities, a sensor of {sensor_name} in the {(timestamp)*time_length} timestamp"
            token_STGraph.append(prompt_)
            # print(prompt_)

    text_tokens = clip.tokenize(token_STGraph).to(device)

    with tr.no_grad():
        text_features = model.encode_text(text_tokens).float()
    return tr.reshape(text_features,[time_length,len(sensors_name), -1])

def sensor_prompt_SSC(model, time_length, sensors_name, device):
    token_STGraph = []
    for timestamp in range(time_length):
        for sensor_name in sensors_name:
            prompt_ = f"When classifying sleep stages by placing electrodes based on 10-20 system, a sensor of {sensor_name}, in the {(timestamp)*time_length} timestamp"
            token_STGraph.append(prompt_)
            # print(prompt_)

    text_tokens = clip.tokenize(token_STGraph).to(device)

    with tr.no_grad():
        text_features = model.encode_text(text_tokens).float()
    return tr.reshape(text_features,[time_length,len(sensors_name), -1])


def label_prompt_RUL(model, time_length, sensors_name, labels, device):

    token_STGraph = []
    num_samples = len(labels)
    for label in labels:
        prompt_ =f"the aircraft turbine engine with remaining useful life as {int(label)} cycles"
        token_STGraph.append(prompt_)

    text_tokens = clip.tokenize(token_STGraph).to(device)

    with tr.no_grad():
        text_features = model.encode_text(text_tokens).float()

    text_features = tr.cat([text_features.unsqueeze(1)]*time_length,1)
    text_features = tr.cat([text_features.unsqueeze(2)]*len(sensors_name),2)

    return text_features



def label_prompt_HAR(model, time_length, sensors_name, labels, device):
    names = ['walking', 'walking upstairs', 'walking downstairs', 'sitting', 'standing', 'laying']
    idx2names = {}
    for i,v in enumerate(names):
        idx2names[i] = v
    token_STGraph = []
    num_samples = len(labels)
    for label in labels:
        prompt_ =f"This is the human activity of {idx2names[int(label)]}"
        token_STGraph.append(prompt_)

    text_tokens = clip.tokenize(token_STGraph).to(device)

    with tr.no_grad():
        text_features = model.encode_text(text_tokens).float()

    text_features = tr.cat([text_features.unsqueeze(1)]*time_length,1)
    text_features = tr.cat([text_features.unsqueeze(2)]*len(sensors_name),2)

    return text_features


def label_prompt_SSC(model, time_length, sensors_name, labels, device):
    names = ['wake',
             'N1, Non-REM stage 1, the lightest stage of sleep',
             'N2, Non-REM stage 2, the slightly deeper stage of sleep',
             'N3, Non-REM stage 3, the deepest stage of sleep',
             'REM, Rapid Eye Movement']
    # names = ['walking', 'walking upstairs', 'walking downstairs', 'sitting', 'standing', 'laying']
    # 0-Wake, 1-N1, 2-N2, 3-N3, 4 correspond to REM, 5-REM
    idx2names = {}
    for i,v in enumerate(names):
        idx2names[i] = v
    token_STGraph = []
    num_samples = len(labels)
    for label in labels:
        prompt_ =f"This is the sleep stages of {idx2names[int(label)]}"
        token_STGraph.append(prompt_)

    text_tokens = clip.tokenize(token_STGraph).to(device)

    with tr.no_grad():
        text_features = model.encode_text(text_tokens).float()

    text_features = tr.cat([text_features.unsqueeze(1)]*time_length,1)
    text_features = tr.cat([text_features.unsqueeze(2)]*len(sensors_name),2)

    return text_features



def contrastive_alignment_sensor(TS_features, text_features, device,temperature=0.005):
    def _get_correlated_mask_node_cross_view(num_node, batch):
        diag = np.eye(num_node)

        diag = np.repeat(np.expand_dims(diag, 0), batch, axis=0)

        mask = tr.from_numpy(diag)
        mask = (1 - mask).type(tr.bool)
        return mask
    ### input size is (bs, num_sensors, dimension)
    temperature = temperature
    criterion = tr.nn.CrossEntropyLoss(reduction="sum")

    bs, num_sensors,_ = TS_features.size()
    # print(TS_features.size())
    # print(text_features.size())
    sim = TS_features @ tr.transpose(text_features, -1, -2)
    mask = _get_correlated_mask_node_cross_view(num_sensors, bs).type(tr.bool)

    pos = tr.diagonal(sim,offset=0,dim1=-1, dim2=-2).unsqueeze(-1)
    neg = sim[mask].reshape([bs,num_sensors,-1])

    logits_node = tr.cat((pos, neg), dim=-1)
    logits_node /= temperature


    labels_node = tr.zeros(bs*num_sensors).to(device).long()
    logits_node = tr.reshape(logits_node, [bs*num_sensors,-1])

    loss_node = criterion(logits_node, labels_node)/(2*bs*num_sensors)
    return loss_node



def contrastive_alignment_label(TS_features, text_features, device,temperature=1):
    def _get_correlated_mask_node_cross_view(num_node):
        diag = np.eye(num_node)

        # diag = np.repeat(np.expand_dims(diag, 0), batch, axis=0)

        mask = tr.from_numpy(diag)
        mask = (1 - mask).type(tr.bool)
        return mask
    ### input size is (bs, num_sensors, dimension)
    # temperature = temperature
    criterion = tr.nn.CrossEntropyLoss(reduction="sum")
    bs, num_sensors,_ = TS_features.size()

    TS_features_batch = tr.reshape(TS_features, [bs,-1])
    Text_features_batch = tr.reshape(text_features, [bs,-1])

    # print('batch')

    sim = TS_features_batch @ tr.transpose(Text_features_batch, -1, -2)
    mask = _get_correlated_mask_node_cross_view(bs).type(tr.bool)
    # print(sim)
    # print(mask)

    pos = tr.diagonal(sim,offset=0,dim1=0, dim2=1).unsqueeze(-1)
    # print(pos.size())

    neg = sim[mask].reshape([bs,-1])
    # print(neg.size())

    logits_node = tr.cat((pos, neg), dim=-1)
    logits_node /= temperature


    labels_node = tr.zeros(bs).to(device).long()
    logits_node = tr.reshape(logits_node, [bs,-1])

    loss_node = criterion(logits_node, labels_node)
    loss_node = loss_node/(2*bs)

    return loss_node



def MSE_alignment(TS_features, text_features, device=None,temperature = None):

    ### input size is (bs, num_sensors, dimension)

    bs, num_sensors,_ = TS_features.size()
    criterion = tr.nn.MSELoss()

    TS_features_batch = tr.reshape(TS_features, [bs*num_sensors,-1])
    Text_features_batch = tr.reshape(text_features, [bs*num_sensors,-1])

    loss_node = criterion(TS_features_batch, Text_features_batch)

    return loss_node



