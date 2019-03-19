# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2019-02-25 11:23:00
# @Last Modified time: 2019-03-19 10:13:25
import torch
from torch.autograd import Variable
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class myNet(nn.Module):

    def __init__(self, instance_number, encoder_dimension):
        super(myNet, self).__init__()
        self.source_variable = nn.Parameter(torch.Tensor(
            instance_number, encoder_dimension).to(device))

        torch.nn.init.xavier_uniform_(
            self.source_variable, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self):
        sim = torch.matmul(self.source_variable,
                           torch.transpose(self.source_variable, 0, 1))
        sim = F.softmax(sim, dim=1)
        return sim


class Verse():

    def __init__(self, instance_number, dimension):
        self.dimension = dimension
        self.instance_number = instance_number
        self.net = myNet(instance_number, dimension).to(device)

    def read_similarity(self, similarity_file):
        df_sim = pd.read_csv(similarity_file)
        df_mat = df_sim.pivot(index='source', columns='target')
        df_mat = df_mat.fillna(0)

        self.node_list = df_mat.index.values.tolist()

        df_mat_value = df_mat.values
        df_mat_value_max_row = df_mat_value.max(axis=1)
        df_mat_value_max_row.shape = (df_mat.shape[0], 1)

        df_mat_value_max_row[df_mat_value_max_row == 0] = 1
        df_mat_value = df_mat_value / (df_mat_value_max_row * 1.0)

        return df_mat_value

    def embedding(self, similarity_file, epoches, log=False):
        df_similarity = self.read_similarity(similarity_file)

        var_similarity = Variable(torch.from_numpy(
            df_similarity).type(torch.FloatTensor).to(device))

        lr = 0.01
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        for epoch in range(epoches):
            similarity_gen = self.net()

            lossMat = -var_similarity * \
                torch.log(similarity_gen + 10**-10)

            loss = torch.sum(lossMat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log:
                print(epoch, loss.item())

    def save_embedding(self, save_file):
        v = self.net.source_variable.cpu().detach().numpy()
        df_v = pd.DataFrame(v, index=self.node_list)
        df_v.index.name = 'nodeID'
        df_v.to_csv(save_file)
