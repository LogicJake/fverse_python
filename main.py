# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2019-03-19 10:13:05
# @Last Modified time: 2019-03-19 10:14:17
from verse import Verse

if __name__ == '__main__':
    epoches = 10
    dimension = 16
    node_num = 2110
    similarity_file = 'citeseer_CN.csv'

    verse = Verse(node_num, dimension)
    verse.embedding(similarity_file, epoches, log=True)
    verse.save_embedding('embedding.csv')
