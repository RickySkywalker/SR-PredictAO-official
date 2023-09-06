#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

from deep_neural_decision_forests import NeuralDecisionForest as NRF

cuda_device = 0

def layer_normalization(x):
    x = x - torch.mean(x, -1).unsqueeze(-1)
    norm_x = torch.sqrt(torch.sum(x**2, -1)).unsqueeze(-1)
    y = x / norm_x
    return y

class SGNN(Module):
    def __init__(self, hidden_size, step=1):
        super(SGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def ave_pooling(self, hidden, graph_mask):
        length = torch.sum(graph_mask, 1)
        hidden = hidden * graph_mask.unsqueeze(-1).float()
        output = torch.sum(hidden, 1) / length.unsqueeze(-1).float()
        return output

    def att_pooling(self, hidden, star_node, graph_mask):
        sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze()
        sim = torch.exp(sim)
        sim_mask = sim * graph_mask.float()
        sim_each = torch.sum(sim_mask, -1).unsqueeze(-1) + 1e-24
        sim = sim_mask/sim_each
        output = torch.sum(sim.unsqueeze(-1) * hidden, 1)
        return output

    def forward(self, A, hidden, graph_mask):
        star_node = self.ave_pooling(hidden, graph_mask)        
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
            sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_size)
            alpha = torch.sigmoid(sim).unsqueeze(-1)
            bs, item_num = hidden.shape[0], hidden.shape[1]
            star_node_repeat = star_node.repeat(1, item_num).view(bs, item_num, self.hidden_size)
            hidden = (1-alpha) * hidden + alpha * star_node_repeat
            star_node = self.att_pooling(hidden, star_node, graph_mask)
        return hidden, star_node

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.tau = opt.tau
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.pos_embedding = nn.Embedding(opt.cutnum, self.hidden_size)
        self.gnn = SGNN(self.hidden_size, step=opt.step)
        self.linear_hn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_four = nn.Linear(self.hidden_size, 1, bias=False)

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


        # para list for RF
        num_trees = 128
        depth = 5
        num_features = self.hidden_size * 11
        used_feature_rate = 0.25
        num_classes = self.n_node - 1

        self.RF = NRF(num_trees,
                      depth,
                      num_features,
                      used_feature_rate,
                      num_classes,
                      device=torch.device("cuda:" + str(cuda_device))
                      )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, seq_hidden, hidden, star_node, mask, graph_mask):
        # Pos Embedding
        bs, item_num = seq_hidden.shape[0], seq_hidden.shape[1]
        index = torch.arange(item_num).unsqueeze(0)
        pos_index = index.repeat(bs, 1).view(bs, item_num)
        pos_index = trans_to_cuda(torch.Tensor(pos_index.float()).long())
        pos_hidden = self.pos_embedding(pos_index)
        seq_hidden = seq_hidden + pos_hidden

        # Last Item
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(star_node).view(star_node.shape[0], 1, star_node.shape[1])  # batch_size x 1 x latent_size
        q3 = self.linear_three(seq_hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_four(torch.sigmoid(q1 + q2 + q3))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        a = layer_normalization(a)
        b = layer_normalization(b)
        scores = torch.matmul(a, b.transpose(1, 0))
        scores *= self.tau
        return scores

    def forward(self, inputs, A, graph_mask):
        hidden = self.embedding(inputs)
        hidden_update, star_node = self.gnn(A, hidden, graph_mask)
        hidden_concat = torch.cat([hidden, hidden_update], -1) # bs * item_num * (2*emb_dim)
        alpha = self.linear_hn(hidden_concat) # bs * item_num * emb_dim
        alpha = torch.sigmoid(alpha)
        output = alpha * hidden + (1-alpha) * hidden_update
        return output, star_node


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(cuda_device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice_gnn(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    graph_mask = torch.sign(items)
    hidden, star_node = model(items, A, graph_mask)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])



    # Get the base module's result
    base_model_result = model.compute_scores(seq_hidden, hidden, star_node, mask, graph_mask)

    # Get the latent variable
    batch_size = seq_hidden.shape[0]
    RF_input_size = (batch_size, -1)
    RF_input = torch.cat([hidden[:, :5, :].reshape(RF_input_size),
                          star_node.reshape(RF_input_size),
                          seq_hidden[:, :5, :].reshape(RF_input_size)], dim=1)

    # Get the RF's result
    RF_result = model.RF(RF_input)


    # Using var-based signal selection to combine results
    base_model_var = torch.exp(torch.var(base_model_result, dim=1).reshape((-1, 1)))
    RF_var = torch.exp(torch.var(RF_result, dim=1).reshape((-1, 1)))
    score = (base_model_result * base_model_var + RF_result * RF_var)/(base_model_var + RF_var)

    return targets, score


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
